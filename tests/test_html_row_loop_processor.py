# SPDX-License-Identifier: Apache-2.0
"""Tests for make_html_row_loop_processor, penalty context forwarding, and ModelSettings fields."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.model_profiles import UNIVERSAL_PROFILE_FIELDS
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.request import SamplingParams
from omlx.scheduler import Scheduler
from omlx.server import _no_repeat_ngram_kwargs, _resolve_repetition_context_size
from omlx.utils.sampling import make_html_row_loop_processor


class _GreedyTokenizer:
    """Minimal greedy longest-match tokenizer for HTML tokenization tests."""

    def __init__(self, pieces: list[str]):
        self.id_to_piece = list(pieces)
        self.piece_to_id = {p: i for i, p in enumerate(self.id_to_piece)}
        # Match longer tokens before shorter prefixes.
        self._sorted_pieces = sorted(
            self.id_to_piece, key=lambda s: (-len(s), self.piece_to_id[s])
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        pos = 0
        while pos < len(text):
            for piece in self._sorted_pieces:
                if piece and text.startswith(piece, pos):
                    ids.append(self.piece_to_id[piece])
                    pos += len(piece)
                    break
            else:
                raise ValueError(f"Unmappable text at {pos}: {text[pos : pos + 20]!r}")
        return ids

    def id_of(self, piece: str) -> int:
        return self.piece_to_id[piece]


def _bpe_merge_tokenizer() -> _GreedyTokenizer:
    # Mimics Qwen3-VL / Infinity-Parser2 where '</tr><tr>' merges into '><'.
    return _GreedyTokenizer(
        [
            "<unused>",
            "<tr",
            "tr",
            "><",
            "td",
            "></",
            ">",
            "\n",
            " <",
            "th",
            "</",
            "table",
            "tbody",
            "thead",
            "colA",
            "colB",
            " ",
            "  ",
        ]
    )


def _indented_tokenizer() -> _GreedyTokenizer:
    # Mimics Qwen2.5-VL / OXR-1.0 indented table rows (' <', 'tr', '>\n', ...).
    return _GreedyTokenizer(
        [
            "<unused>",
            " <",
            "tr",
            ">\n",
            "  ",
            "td",
            "></",
            " </",
            "<table>\n",
            "</table>",
            "<thead>\n",
            "</thead>\n",
            "<tbody>\n",
            "</tbody>",
            "<th>",
            "</th>\n",
            "A",
            "B",
            "\n",
            "<td>",
            "</td>",
            "</tr>",
            "<tr",
            ">",
            "<",
            "/",
            " ",
        ]
    )


def _special_token_tokenizer() -> _GreedyTokenizer:
    # Mimics Unlimited-OCR-2 atomic HTML tag tokens.
    return _GreedyTokenizer(
        [
            "<unused>",
            "<tr>",
            "<td>",
            "</td>",
            "</tr>",
            "<table>",
            "</table>",
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<th>",
            "</th>",
            "val0",
            "val1",
            " ",
            "\n",
        ]
    )


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
def test_bpe_merging_tokenizer_bans_reopen_after_20_rows_only_at_boundary(dtype):
    tok = _bpe_merge_tokenizer()
    proc = make_html_row_loop_processor(tok)
    assert proc is not None

    open_id = tok.id_of("><")
    close_id = tok.id_of("></")
    # Explicit blank row matching Infinity-Parser2 structure:
    row_tokens = [
        tok.id_of(p)
        for p in ("><", "tr", "><", "td", "></", "td", "></", "tr")
    ]
    assert row_tokens[-3:] == [tok.id_of("td"), tok.id_of("></"), tok.id_of("tr")]

    # L0 = 0 (preceded by 'tbody') and L0 = 3 (header ends with td, ></, tr)
    prefixes = [
        [tok.id_of("tbody"), tok.id_of(">")],
        [tok.id_of("th"), tok.id_of("></"), tok.id_of("td"), tok.id_of("></"), tok.id_of("tr")],
    ]

    for prefix in prefixes:
        vocab = 32
        logits = mx.zeros((1, vocab), dtype=mx.bfloat16)

        # 19 blank rows -> must NOT ban anything yet.
        seq_19 = mx.array(prefix + row_tokens * 19, dtype=dtype)
        out_19 = proc(seq_19, logits)
        mx.eval(out_19)
        assert out_19.dtype == mx.bfloat16
        assert float(out_19[0, open_id].item()) == 0.0

        # Mid-row after 20 blank rows -> must NOT ban anything (not at </tr> boundary).
        seq_mid = mx.array(prefix + row_tokens * 20 + row_tokens[:-1], dtype=dtype)
        out_mid = proc(seq_mid, logits)
        mx.eval(out_mid)
        assert not mx.isinf(out_mid).any().item()

        # Exactly after 20th blank row ('td', '></', 'tr') -> MUST ban only '><'.
        seq_20 = mx.array(prefix + row_tokens * 20, dtype=dtype)
        out_20 = proc(seq_20, logits)
        mx.eval(out_20)
        assert float(out_20[0, open_id].item()) == float("-inf")
        assert float(out_20[0, close_id].item()) == 0.0
        assert mx.isinf(out_20).sum().item() == 1


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
def test_indented_tokenizer_handles_overlap_L0_5(dtype):
    tok = _indented_tokenizer()
    proc = make_html_row_loop_processor(tok)
    assert proc is not None

    open_id = tok.id_of(" <")
    close_table_id = tok.id_of("</table>")
    row_tokens = tok.encode(" <tr>\n  <td></td>\n  <td></td>\n </tr>\n")
    # L0 = 5 overlap before Row 1: ends with '  ', 'td', ' </', 'tr', '>\n' matching row tail.
    header_l0_5 = (
        tok.encode(" <tr>\n  <th>A</th>\n </tr>\n")[:-5]
        + row_tokens[-5:]
    )

    logits = mx.ones((1, 40), dtype=mx.float32)

    # At row 19 end -> no ban
    out_19 = proc(mx.array(header_l0_5 + row_tokens * 19, dtype=dtype), logits)
    mx.eval(out_19)
    assert float(out_19[0, open_id].item()) == 1.0

    # At row 20 + 2 tokens (mid-row) -> no ban
    out_mid = proc(mx.array(header_l0_5 + row_tokens * 20 + row_tokens[:2], dtype=dtype), logits)
    mx.eval(out_mid)
    assert float(out_mid[0, open_id].item()) == 1.0

    # At row 20 end -> bans only ' <', keeps '</table>'
    out_20 = proc(mx.array(header_l0_5 + row_tokens * 20, dtype=dtype), logits)
    mx.eval(out_20)
    assert float(out_20[0, open_id].item()) == float("-inf")
    assert float(out_20[0, close_table_id].item()) == 1.0


@pytest.mark.parametrize("dtype", [mx.int32, mx.uint32])
def test_special_token_tokenizer_bans_tr_after_20_rows_only_at_end(dtype):
    tok = _special_token_tokenizer()
    proc = make_html_row_loop_processor(tok)
    assert proc is not None

    tr_id = tok.id_of("<tr>")
    table_end_id = tok.id_of("</table>")
    row_tokens = tok.encode("<tr><td></td><td></td></tr>")

    # L0 = 1: preceding header row ends with </tr>.
    header = tok.encode("<tr><th></th></tr>")
    logits = mx.zeros((1, 32), dtype=mx.float32)

    out_mid = proc(mx.array(header + row_tokens * 20 + row_tokens[:-1], dtype=dtype), logits)
    mx.eval(out_mid)
    assert float(out_mid[0, tr_id].item()) == 0.0

    out_20 = proc(mx.array(header + row_tokens * 20, dtype=dtype), logits)
    mx.eval(out_20)
    assert float(out_20[0, tr_id].item()) == float("-inf")
    assert float(out_20[0, table_end_id].item()) == 0.0


def test_non_identical_rows_never_trigger():
    tok = _special_token_tokenizer()
    proc = make_html_row_loop_processor(tok)
    assert proc is not None

    tokens = []
    for i in range(25):
        val = "val0" if i % 2 == 0 else "val1"
        tokens.extend(tok.encode(f"<tr><td>{val}</td><td></td></tr>"))
    logits = mx.zeros((1, 32), dtype=mx.float32)
    out = proc(mx.array(tokens, dtype=mx.int32), logits)
    mx.eval(out)
    assert not mx.isinf(out).any().item()


def test_make_html_row_loop_processor_invalid_tokenizers():
    assert make_html_row_loop_processor(None) is None
    assert make_html_row_loop_processor(SimpleNamespace()) is None
    assert make_html_row_loop_processor(SimpleNamespace(encode=lambda *_a, **_k: "bad")) is None
    assert make_html_row_loop_processor(SimpleNamespace(encode=lambda *_a, **_k: [1, 2])) is None


def test_scheduler_attaches_html_loop_processor_and_forwards_context_sizes(mock_model):
    tok = _special_token_tokenizer()
    scheduler = Scheduler(model=mock_model, tokenizer=tok)

    # Neutral penalties -> no HTML loop processor attached.
    _, procs_neutral = scheduler._build_sampler_and_processors(SamplingParams())
    assert procs_neutral == []

    # Active repetition_penalty -> attaches HTML row loop processor and forwards context sizes.
    params = SamplingParams(repetition_penalty=1.05, repetition_context_size=128)
    with patch(
        "omlx.scheduler.make_logits_processors", return_value=[]
    ) as mock_make_lp:
        _, procs = scheduler._build_sampler_and_processors(params)

    mock_make_lp.assert_called_once_with(
        repetition_penalty=1.05,
        presence_penalty=None,
        frequency_penalty=None,
        repetition_context_size=128,
        presence_context_size=128,
        frequency_context_size=128,
    )
    assert len(procs) == 1
    assert procs[0] is scheduler._html_row_loop_processor


def test_scheduler_context_size_fallback_on_narrow_mock(mock_model, mock_tokenizer):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    params = SamplingParams(repetition_penalty=1.05, repetition_context_size=64)

    calls = []

    def _narrow_stub(*, repetition_penalty=None, presence_penalty=None, frequency_penalty=None, repetition_context_size=None):
        calls.append(repetition_context_size)
        return []

    with patch("omlx.scheduler.make_logits_processors", side_effect=_narrow_stub):
        scheduler._build_sampler_and_processors(params)

    assert calls == [64]


def test_model_settings_and_server_fallbacks(tmp_path):
    assert "repetition_context_size" in UNIVERSAL_PROFILE_FIELDS
    assert "no_repeat_ngram_size" in UNIVERSAL_PROFILE_FIELDS
    assert "ngram_window" in UNIVERSAL_PROFILE_FIELDS

    manager = ModelSettingsManager(tmp_path)
    ms = ModelSettings(
        repetition_context_size=256,
        no_repeat_ngram_size=8,
        ngram_window=64,
    )
    restored = ModelSettings.from_dict(ms.to_dict())
    assert restored.repetition_context_size == 256
    assert restored.no_repeat_ngram_size == 8
    assert restored.ngram_window == 64

    manager.set_settings("ocr-model", ms)

    with patch("omlx.server.get_model_settings_for_request", side_effect=lambda mid: manager.get_settings(mid)):
        req_default = SimpleNamespace(
            model="ocr-model",
            repetition_context_size=None,
            no_repeat_ngram_size=None,
            ngram_window=None,
        )
        assert _resolve_repetition_context_size(req_default, "ocr-model") == 256
        assert _no_repeat_ngram_kwargs(req_default, "ocr-model") == {
            "no_repeat_ngram_size": 8,
            "ngram_window": 64,
        }

        req_override = SimpleNamespace(
            model="ocr-model",
            repetition_context_size=512,
            no_repeat_ngram_size=12,
            ngram_window=128,
        )
        assert _resolve_repetition_context_size(req_override, "ocr-model") == 512
        assert _no_repeat_ngram_kwargs(req_override, "ocr-model") == {
            "no_repeat_ngram_size": 12,
            "ngram_window": 128,
        }
