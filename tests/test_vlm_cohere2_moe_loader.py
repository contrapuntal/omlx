# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cohere2 MoE mlx-vlm text-only load path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")

from omlx.engine import vlm as vlm_module
from omlx.engine.vlm import (
    VLMBatchedEngine,
    _load_vlm_native_text_model,
)
from omlx.exceptions import InvalidRequestError


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    eos_token_ids = None
    pad_token = None


class _FakeDetokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeStoppingCriteria:
    def __init__(self, eos_token_ids, tokenizer):
        self.eos_token_ids = eos_token_ids
        self.tokenizer = tokenizer


def test_cohere2_moe_loader_uses_upstream_processor(monkeypatch, tmp_path):
    import mlx_vlm.utils as vlm_utils

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[2]))
    processor = object()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(vlm_utils, "load_processor", lambda *a, **k: processor)

    loaded_model, loaded_processor = _load_vlm_native_text_model("cohere")

    assert loaded_model is model
    assert loaded_processor is processor


def test_cohere2_moe_loader_falls_back_to_tokenizer(monkeypatch, tmp_path):
    import mlx_vlm.tokenizer_utils as tokenizer_utils
    import mlx_vlm.utils as vlm_utils
    import transformers

    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[7]))
    tokenizer = _FakeTokenizer()

    monkeypatch.setattr(vlm_utils, "get_model_path", lambda model_name: tmp_path)
    monkeypatch.setattr(vlm_utils, "load_model", lambda *a, **k: model)

    def fail_processor(*args, **kwargs):
        raise ValueError("no processor")

    monkeypatch.setattr(vlm_utils, "load_processor", fail_processor)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: tokenizer,
    )
    monkeypatch.setattr(
        tokenizer_utils,
        "load_tokenizer",
        lambda *a, **k: _FakeDetokenizer,
    )
    monkeypatch.setattr(vlm_utils, "StoppingCriteria", _FakeStoppingCriteria)

    loaded_model, loaded_processor = _load_vlm_native_text_model("cohere")

    assert loaded_model is model
    assert loaded_processor is tokenizer
    assert tokenizer.pad_token == "<eos>"
    assert isinstance(tokenizer.detokenizer, _FakeDetokenizer)
    assert isinstance(tokenizer.stopping_criteria, _FakeStoppingCriteria)
    assert tokenizer.stopping_criteria.eos_token_ids == [7]


def test_cohere2_moe_rejects_image_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "describe"}],
            images=[object()],
        )


def test_cohere2_moe_rejects_audio_input():
    engine = VLMBatchedEngine("cohere")
    engine._vlm_model = SimpleNamespace(
        config=SimpleNamespace(model_type=vlm_module.COHERE2_MOE_MODEL_TYPE)
    )

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "transcribe"}],
            images=[],
            audio=[("samples", 16000)],
        )


def test_laguna_rejects_image_input():
    """Native text models other than Cohere2 (e.g. Laguna) also reject images."""
    engine = VLMBatchedEngine("laguna")
    engine._vlm_model = SimpleNamespace(config=SimpleNamespace(model_type="laguna"))

    with pytest.raises(InvalidRequestError, match="text-only"):
        engine._prepare_vision_inputs(
            [{"role": "user", "content": "describe"}],
            images=[object()],
        )


def test_force_sanitize_strips_mlx_format_within_model_dir(tmp_path):
    """``_force_sanitize_for_mlx_format`` hides ``format=mlx`` for in-dir files.

    mlx-vlm's ``load_model`` keys ``is_mlx_format`` off the safetensors
    ``format`` metadata and skips ``Model.sanitize`` when it is ``"mlx"``.
    The context manager must drop that key for files under the model dir
    (so sanitize runs) while leaving other metadata and outside files intact.
    """
    import numpy as np
    import safetensors
    import safetensors.numpy as stnp

    from omlx.engine.vlm import _force_sanitize_for_mlx_format

    inside = tmp_path / "model" / "weights.safetensors"
    inside.parent.mkdir()
    outside = tmp_path / "other.safetensors"
    tensors = {"w": np.zeros((2, 2), dtype=np.float32)}
    stnp.save_file(tensors, str(inside), metadata={"format": "mlx", "keep": "1"})
    stnp.save_file(tensors, str(outside), metadata={"format": "mlx"})

    with _force_sanitize_for_mlx_format(inside.parent):
        with safetensors.safe_open(str(inside), framework="np") as f:
            meta = f.metadata()
            assert "format" not in meta  # dropped → sanitize will run
            assert meta.get("keep") == "1"  # other metadata preserved
        with safetensors.safe_open(str(outside), framework="np") as f:
            assert f.metadata().get("format") == "mlx"  # outside dir untouched

    # Restored after the context exits.
    with safetensors.safe_open(str(inside), framework="np") as f:
        assert f.metadata().get("format") == "mlx"
