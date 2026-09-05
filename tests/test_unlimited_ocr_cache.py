"""The OCR decode ring must survive oMLX cache boundaries unchanged."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cache.type_registry import CacheTypeRegistry
from omlx.models.vlm import VLMModelAdapter
from omlx.scheduler import Scheduler, SchedulerConfig, _patched_merge_caches


def adapter():
    from omlx.patches.mlx_vlm_unlimited_ocr_compat import (
        apply_mlx_vlm_unlimited_ocr_compat_patch,
    )

    apply_mlx_vlm_unlimited_ocr_compat_patch()
    from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

    return VLMModelAdapter(
        SimpleNamespace(
            config=SimpleNamespace(model_type="unlimited-ocr"),
            language_model=SimpleNamespace(make_cache=lambda: [RingSlidingKVCache(2)]),
        )
    )


def append(cache, values):
    tensor = mx.array(values, dtype=mx.float32).reshape(1, 1, -1, 1)
    return cache.update_and_fetch(tensor, tensor)[0].reshape(-1).tolist()


def make_filled_cache():
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    for token in range(4, 12):
        append(cache, [token])
    return cache


def test_singleton_merge_keeps_native_ring():
    cache = make_filled_cache()
    merged = _patched_merge_caches([[cache]])[0]
    assert append(merged, [12]) == [0, 1, 2, 3, 12, 11]
    assert merged.offset == 13


def test_unsafe_multirow_merge_is_rejected():
    with pytest.raises(ValueError, match="serial"):
        _patched_merge_caches([[make_filled_cache()], [make_filled_cache()]])


def test_ring_prefix_serialization_does_not_export_overwritten_slots():
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = handler.serialize_state(cache)
    assert state[0].reshape(-1).tolist() == [0, 1, 2, 3]
    assert state[1].shape == (1, 1, 4, 1)


def test_restored_partial_prefix_reestablishes_ring_boundary():
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = {"keys": cache.keys[:, :, :2], "values": cache.values[:, :, :2]}
    restored = handler.reconstruct_cache(state, handler.serialize_meta_state(cache))
    append(restored, [2, 3])
    for token in range(4, 12):
        result = append(restored, [token])
    assert result == [0, 1, 2, 3, 10, 11]
    assert restored.offset == 12


@pytest.mark.parametrize(
    "meta",
    [
        None,
        (),
        ("0", "-1", "4", "0"),
        ("2", "4", "99", "0"),
        ("2", "4", "4", "0", "extra"),
    ],
)
def test_bad_ring_metadata_rejected(meta):
    cache = make_filled_cache()
    handler = CacheTypeRegistry.get_handler_for_object(cache)
    state = {"keys": cache.keys[:, :, :4], "values": cache.values[:, :, :4]}
    with pytest.raises(ValueError):
        handler.reconstruct_cache(state, meta)


def test_scheduler_serializes_only_ring_model(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer, SchedulerConfig(max_num_seqs=8))
    assert scheduler._effective_max_num_seqs() == 8
    scheduler.model = adapter()
    assert scheduler._effective_max_num_seqs() == 1


def test_scheduler_exports_only_intact_prefix(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([make_filled_cache()])
    assert payload[0]["state"][0].reshape(-1).tolist() == [0, 1, 2, 3]
    handler = CacheTypeRegistry.get_handler_by_class_name(
        config.layer_configs[0].class_name
    )
    restored = handler.reconstruct_cache(
        {"keys": payload[0]["state"][0], "values": payload[0]["state"][1]},
        payload[0]["meta_state"],
    )
    for token in range(4, 12):
        values = append(restored, [token])
    assert values == [0, 1, 2, 3, 10, 11]


def test_ring_is_not_turboquant_convertible(mock_model, mock_tokenizer):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    assert not scheduler._turboquant_eligible([make_filled_cache()])


def test_stale_plain_kv_prefix_is_incompatible():
    from mlx_lm.models.cache import KVCache

    model = adapter()
    assert not model.is_prefix_cache_compatible([KVCache()])
    assert model.is_prefix_cache_compatible(model.make_cache())


def test_singleton_protocol_and_trim():
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    assert cache.to_batch([0]) is cache
    assert cache.extract(0) is cache
    cache.filter([0])
    assert cache.is_trimmable()
    assert cache.trim(1) == 1
    assert cache.offset == 3
    append(cache, [3])
    assert not cache.is_trimmable()
    with pytest.raises(ValueError, match="trim"):
        cache.trim(1)
    with pytest.raises(ValueError, match="serial"):
        cache.to_batch([1])
    with pytest.raises(ValueError, match="serial"):
        cache.extend(cache)
    cache.filter([])
    assert cache.offset == 0 and cache.keys is None


@pytest.mark.parametrize("block_size", [2, 3])
def test_ssd_store_and_restore_only_prompt_blocks(
    tmp_path, mock_model, mock_tokenizer, caplog, block_size
):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path,
        max_size_bytes=1024**2,
        expected_model_name="ocr",
        expected_num_layers=1,
        expected_block_size=block_size,
    )
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=32, initial_blocks=32, model_name="ocr"
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(layers=[None]),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
    )
    scheduler = Scheduler(mock_model, mock_tokenizer)
    payload, config = scheduler._extract_cache_states([make_filled_cache()])
    try:
        table = prefix.store_cache(
            "cold", list(range(12)), payload, model_cache_config=config
        )
        assert table is not None
        assert table.num_tokens == 4 // block_size * block_size
        assert "Rejecting block" not in caplog.text
        restored = prefix.reconstruct_cache(table)[0]
        if table.num_tokens < 4:
            # Append a multi-token suffix so it remains prefill, not decode.
            append(restored, list(range(table.num_tokens, 5)))
            expected_prefix = list(range(5))
        else:
            expected_prefix = list(range(4))
        for token in range(len(expected_prefix), 12):
            values = append(restored, [token])
        assert values[: len(expected_prefix)] == expected_prefix
        assert sorted(values[len(expected_prefix) :]) == [10, 11]

        # The first blocks retain the first capture's metadata after dedup.
        extended = adapter().make_cache()[0]
        append(extended, list(range(8)))
        payload, config = scheduler._extract_cache_states([extended])
        table = prefix.store_cache(
            "extended", list(range(8)), payload, model_cache_config=config
        )
        restored = prefix.reconstruct_cache(table)
        assert restored is not None
        assert restored[0].offset == 8 // block_size * block_size
    finally:
        ssd.close()


def test_scheduler_discards_legacy_ocr_prefix(mock_model, mock_tokenizer):
    from unittest.mock import MagicMock

    from mlx_lm.models.cache import KVCache

    from omlx.cache.paged_cache import BlockTable
    from omlx.request import Request, SamplingParams

    scheduler = Scheduler(mock_model, mock_tokenizer)
    scheduler.model = adapter()
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = MagicMock()
    table = BlockTable(request_id="legacy", block_ids=[1], num_tokens=2)
    scheduler.block_aware_cache.fetch_cache.return_value = (table, [12, 13])
    scheduler.block_aware_cache.reconstruct_cache.return_value = [KVCache()]
    request = Request(
        request_id="legacy",
        prompt=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=4),
    )
    scheduler.add_request(request)
    scheduler._prepare_prefix_cache_for_request(request)
    assert request.cached_tokens == 0 and request.prompt_cache is None
    assert request.remaining_tokens == [10, 11, 12, 13]
    scheduler.paged_cache_manager.delete_block_table.assert_called_once_with("legacy")


def make_ssd_stack(directory, block_size):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    ssd = PagedSSDCacheManager(
        cache_dir=directory,
        max_size_bytes=1024**2,
        hot_cache_max_bytes=0,
        expected_model_name="ocr",
        expected_num_layers=1,
        expected_block_size=block_size,
    )
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=32, initial_blocks=32, model_name="ocr"
    )
    paged.set_paged_ssd_cache_manager(ssd)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(layers=[None]),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
    )
    return prefix, paged, ssd


class TinyRingLanguageModel:
    """Weight-free model exercising real scheduler prefill and EOS completion."""

    layers = [None]

    def __init__(self):
        self.seen_tokens = []

    def make_cache(self):
        from mlx_vlm.models.unlimited_ocr.language import RingSlidingKVCache

        return [RingSlidingKVCache(2)]

    def __call__(self, input_ids, cache=None, **kwargs):
        self.seen_tokens.extend(input_ids.reshape(-1).tolist())
        tensor = input_ids.astype(mx.float32)[:, None, :, None]
        for layer in cache or []:
            layer.update_and_fetch(tensor, tensor)
        # The tokenizer fixture's EOS is 2. Return real logits, not a mocked
        # scheduler/BatchGenerator response, so fallback must actually run.
        return mx.broadcast_to(mx.array([0.0, 0.0, 10.0, 0.0]), (*input_ids.shape, 4))


def test_guard_refuses_native_ring_until_registration(
    monkeypatch, mock_model, mock_tokenizer
):
    model = adapter()
    native_model = model._language_model
    with monkeypatch.context() as unregistered:
        for name in ("RingSlidingKVCache", "OMLXRingSlidingKVCache"):
            unregistered.delitem(CacheTypeRegistry._class_name_map, name)
        before = Scheduler(mock_model, mock_tokenizer)
        before.model = native_model
        assert before._model_has_unreconstructible_cache()
    after = Scheduler(mock_model, mock_tokenizer)
    after.model = model
    assert not after._model_has_unreconstructible_cache()


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("legacy_layout", ["native-decode", "plain-kv"])
def test_persisted_legacy_entries_release_real_references_after_restart(
    tmp_path, mock_model, mock_tokenizer, block_size, legacy_layout, monkeypatch
):
    from mlx_lm.models.cache import KVCache

    from omlx.cache.hybrid_cache import ModelCacheConfig
    from omlx.request import Request, SamplingParams

    # Capture the old writer's raw state/meta contract, before the ring handler
    # exported prompt-only tensors. No production image/hash salt is involved.
    native = adapter()._language_model.make_cache()[0]
    append(native, [0, 1, 2, 3])
    for token in range(4, 12):
        append(native, [token])
    if legacy_layout == "plain-kv":
        old = KVCache()
        old.state = native.state
    else:
        old = native
        assert old.meta_state == ("2", "4", "12", "0")
    payload = [
        {
            "state": old.state,
            "meta_state": old.meta_state,
            "class_name": type(old).__name__,
            "cache_type": "KVCache",
        }
    ]
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        with monkeypatch.context() as unregistered:
            for name in ("RingSlidingKVCache", "OMLXRingSlidingKVCache"):
                unregistered.delitem(CacheTypeRegistry._class_name_map, name)
            config = ModelCacheConfig.from_cache_list([old])
            table = prefix.store_cache(
                "old-writer", list(range(12)), payload, model_cache_config=config
            )
        assert table is not None and table.num_tokens == 6
    finally:
        ssd.close()
    assert list(tmp_path.rglob("*.safetensors")), "fixture must reach disk"

    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    scheduler = Scheduler(mock_model, mock_tokenizer)
    scheduler.model = adapter()
    language_model = TinyRingLanguageModel()
    scheduler.model._language_model = language_model
    scheduler.block_aware_cache = prefix
    scheduler.paged_cache_manager = paged
    try:
        # Prove the unchanged identity actually hits the old persisted chain.
        table, remaining = prefix.fetch_cache("probe", list(range(8)))
        assert table is not None and table.num_tokens == 6
        assert remaining == [6, 7]
        if legacy_layout == "native-decode":
            # Exercise handler metadata refusal, not just the adapter's
            # different-class check for a plain-KV reconstruction.
            assert prefix.reconstruct_cache(table) is None
        else:
            assert type(prefix.reconstruct_cache(table)[0]) is KVCache
        old_block_ids = tuple(table.block_ids)
        paged.delete_block_table("probe")
        for attempt in range(3):
            request = Request(
                request_id=f"legacy-{attempt}",
                prompt=list(range(8)),
                sampling_params=SamplingParams(max_tokens=4),
            )
            scheduler.add_request(request)
            scheduler._prepare_prefix_cache_for_request(request)
            if attempt == 0:
                assert request.cached_tokens == 0
                assert request.prompt_cache is None and request.block_table is None
                assert request.remaining_tokens == list(range(8))
                assert paged.get_block_table(request.request_id) is None
                assert all(paged.blocks[i].ref_count == 0 for i in old_block_ids)
                assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
                assert paged.free_block_queue.num_free_blocks == 31
                expected_prefill = list(range(8))
            else:
                # Completion rewrites the rejected entries with safe prompt
                # captures. Subsequent requests must reuse those, not keep
                # missing forever or inherit the old full-KV attention.
                assert request.cached_tokens == 6
                assert scheduler.model.is_prefix_cache_compatible(request.prompt_cache)
                expected_prefill = [6, 7]
            language_model.seen_tokens.clear()
            for _ in range(10):
                scheduler.step()
                if request.is_finished():
                    break
            assert request.is_finished()
            assert request.get_finish_reason() == "stop"
            assert (
                language_model.seen_tokens[: len(expected_prefill)] == expected_prefill
            )
            assert all(
                token == 2
                for token in language_model.seen_tokens[len(expected_prefill) :]
            )
            assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("block_size", [2, 3])
def test_guarded_reuse_after_restart_preserves_ring(
    tmp_path, mock_model, mock_tokenizer, block_size
):
    from omlx.request import Request, SamplingParams

    scheduler = Scheduler(mock_model, mock_tokenizer)
    cache = adapter().make_cache()[0]
    append(cache, list(range(8)))
    payload, config = scheduler._extract_cache_states([cache])
    prefix, _, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        table = prefix.store_cache(
            "cold", list(range(8)), payload, model_cache_config=config
        )
        assert table is not None
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    scheduler.model = adapter()
    scheduler.block_aware_cache = prefix
    scheduler.paged_cache_manager = paged
    try:
        request = Request("warm", list(range(10)), SamplingParams(max_tokens=4))
        scheduler.add_request(request)
        scheduler._prepare_prefix_cache_for_request(request)
        assert request.cached_tokens == (8 if block_size == 2 else 6)
        restored = request.prompt_cache[0]
        append(restored, request.remaining_tokens)
        for token in range(10, 16):
            values = append(restored, [token])
        assert values == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15]
        paged.delete_block_table(request.request_id)
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


@pytest.mark.parametrize("block_size", [2, 3])
def test_deduplicated_legacy_prefill_cannot_hide_decoded_tail(
    tmp_path, monkeypatch, block_size
):
    from omlx.cache.hybrid_cache import ModelCacheConfig

    native = adapter()._language_model.make_cache()[0]
    append(native, [0, 1, 2, 3])
    prefix, _, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        with monkeypatch.context() as unregistered:
            unregistered.delitem(
                CacheTypeRegistry._class_name_map, "RingSlidingKVCache"
            )
            config = ModelCacheConfig.from_cache_list([native])
            first = [
                {
                    "state": native.state,
                    "meta_state": native.meta_state,
                    "class_name": "RingSlidingKVCache",
                    "cache_type": "KVCache",
                }
            ]
            table = prefix.store_cache(
                "old-prefill", list(range(4)), first, model_cache_config=config
            )
            assert table is not None
            for token in range(4, 12):
                append(native, [token])
            decoded = [
                {
                    "state": native.state,
                    "meta_state": native.meta_state,
                    "class_name": "RingSlidingKVCache",
                    "cache_type": "KVCache",
                }
            ]
            table = prefix.store_cache(
                "old-decode", list(range(12)), decoded, model_cache_config=config
            )
            assert table is not None and table.num_tokens == 6
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, block_size)
    try:
        table, _ = prefix.fetch_cache("warm-mixed", list(range(8)))
        assert table is not None and table.num_tokens == 6
        # The first block's metadata is valid prefill; a later block contains
        # overwritten decode slots [10, 11] falsely labeled as prompt [4, 5].
        assert prefix.reconstruct_cache(table) is None
        paged.delete_block_table("warm-mixed")
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()


def test_ring_block_tensor_length_must_match_token_count(
    tmp_path, mock_model, mock_tokenizer
):
    scheduler = Scheduler(mock_model, mock_tokenizer)
    cache = adapter().make_cache()[0]
    append(cache, [0, 1, 2, 3])
    payload, config = scheduler._extract_cache_states([cache])
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table = prefix.store_cache(
            "cold", [0, 1, 2, 3], payload, model_cache_config=config
        )
        tail_hash = paged.blocks[table.block_ids[-1]].block_hash
    finally:
        ssd.close()
    prefix, _, ssd = make_ssd_stack(tmp_path, 2)
    try:
        short = mx.array([2.0]).reshape(1, 1, 1, 1)
        assert ssd.save_block(
            tail_hash,
            [(short, short)],
            token_count=2,
            model_name="ocr",
            layer_cache_types=["OMLXRingSlidingKVCache"],
            layer_meta_states=[("2", "-1", "4", "0")],
            replace_existing=True,
        )
    finally:
        ssd.close()
    prefix, paged, ssd = make_ssd_stack(tmp_path, 2)
    try:
        table, _ = prefix.fetch_cache("warm-short", [0, 1, 2, 3, 4, 5])
        assert table is not None and table.num_tokens == 4
        assert prefix.reconstruct_cache(table) is None
        paged.delete_block_table("warm-short")
        assert all(b.ref_count == 0 for b in paged.blocks if not b.is_null)
    finally:
        ssd.close()
