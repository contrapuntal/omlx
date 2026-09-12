"""TeleOCR math and checkpoint contracts, using an independent NumPy oracle."""

import copy
import importlib
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.models.cache import KVCache


def adapter():
    from omlx.patches.mlx_vlm_mlx0322_compat import (
        apply_mlx_vlm_mlx0322_compat_patch,
    )

    # Match oMLX pre-load ordering: inherited vision methods must be imported
    # after the MLX scalar-argument compatibility hook is installed.
    apply_mlx_vlm_mlx0322_compat_patch()
    return importlib.import_module("omlx.patches.mlx_vlm_teleocr_compat")


def tiny_config():
    return dict(
        model_type="qwen2_5_vl",
        auto_map={
            "AutoModelForImageTextToText": "modeling_naviocr.Qwen2_5_VLForConditionalGeneration"
        },
        text_config=dict(
            model_type="qwen2_5_vl",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=6,
            rms_norm_eps=1e-6,
            vocab_size=20,
            tie_word_embeddings=True,
            rope_theta=10000.0,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 1]},
        ),
        vision_config=dict(
            depth=1,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
            out_hidden_size=8,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=1,
            window_size=4,
            fullatt_block_indexes=[0],
        ),
    )


def test_selection_is_metadata_scoped_and_idempotent():
    a = adapter()
    import mlx_vlm.utils as utils

    ordinary = utils.get_model_and_args({"model_type": "qwen2_5_vl"})[0]
    assert a.apply_mlx_vlm_teleocr_compat_patch()
    wrapper = utils.get_model_and_args
    assert a.apply_mlx_vlm_teleocr_compat_patch()
    assert utils.get_model_and_args is wrapper
    assert utils.get_model_and_args(tiny_config())[0].Model is a.Model
    assert utils.get_model_and_args({"model_type": "qwen2_5_vl"})[0] is ordinary
    assert not a.is_teleocr_config(
        {"model_type": "qwen2_5_vl", "_name_or_path": "TeleOCR"}
    )
    assert a.is_teleocr_config({"model_type": "qwen2_5_vl", "omlx_adapter": "teleocr"})


def test_config_preserves_nested_dimensions_and_root_head_ownership():
    raw = tiny_config()
    original = copy.deepcopy(raw)
    cfg = adapter().ModelConfig.from_dict(raw)
    assert raw == original
    assert cfg.text_config.head_dim == 6
    assert cfg.text_config.tie_word_embeddings
    raw["tie_word_embeddings"] = False
    assert not adapter().ModelConfig.from_dict(raw).text_config.tie_word_embeddings
    flat = {**raw, **raw.pop("text_config")}
    assert adapter().ModelConfig.from_dict(flat).text_config.head_dim == 6


@pytest.mark.parametrize("layout", ["nested", "flat", "renamed"])
def test_unknown_geometry_is_refused_instead_of_certified_as_teleocr(layout):
    raw = tiny_config()
    raw.pop("auto_map")
    if layout == "flat":
        raw.update(raw.pop("text_config"))
    elif layout == "renamed":
        raw["auto_map"] = {"AutoModel": "modeling_teleocr.TeleOCRModel"}
    a = adapter()
    assert not a.is_teleocr_config(raw)
    a.apply_mlx_vlm_teleocr_compat_patch()
    from mlx_vlm.utils import get_model_and_args

    with pytest.raises(ValueError, match="Unrecognized.*geometry"):
        a.reject_unrecognized_geometry(raw)
    # The process-wide hook must leave unrelated loaders' policy alone.
    assert get_model_and_args(raw)[0].__name__ == "mlx_vlm.models.qwen2_5_vl"


def test_teleocr_binding_rejects_incomplete_weights_even_when_strict_is_false():
    a = adapter()
    model = a.Model(a.ModelConfig.from_dict(tiny_config()))
    weights = dict(tree_flatten(model.parameters()))
    weights.pop("language_model.model.layers.0.self_attn.q_norm.weight")
    with pytest.raises(ValueError, match="q_norm"):
        model.load_weights(list(weights.items()), strict=False)


@pytest.mark.parametrize("tied", [True, False])
def test_sanitize_accepts_model_prefixed_output_head(tied):
    a = adapter()
    config = tiny_config()
    config["tie_word_embeddings"] = tied
    model = a.Model(a.ModelConfig.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    head = weights.pop(
        "language_model.lm_head.weight",
        weights["language_model.model.embed_tokens.weight"],
    )
    weights["model.lm_head.weight"] = head
    sanitized = model.sanitize(weights)
    model.load_weights(list(sanitized.items()), strict=True)
    assert ("language_model.lm_head.weight" in sanitized) is (not tied)


def test_vision_sanitize_drops_only_position_id_buffers():
    a = adapter()
    vision = a.VisionModel(a.ModelConfig.from_dict(tiny_config()).vision_config)
    value = mx.ones((1,))
    weights = {
        "vision_tower.position_ids": value,
        "vision_tower.position_ids_projection.weight": value,
    }
    assert set(vision.sanitize(weights)) == {
        "vision_tower.position_ids_projection.weight"
    }


@pytest.mark.parametrize(
    "sliding",
    [{"use_sliding_window": True}, {"layer_types": ["sliding_attention"] * 2}],
)
def test_sliding_window_variants_are_rejected(sliding):
    raw = tiny_config()
    raw["text_config"].update(sliding)
    with pytest.raises(ValueError, match="full attention"):
        adapter().ModelConfig.from_dict(raw)


@pytest.mark.parametrize("head_dim,heads", [(None, 2), (4, 2), (6, 0), ("6", 2)])
def test_geometry_detection_does_not_claim_stock_or_invalid_configs(head_dim, heads):
    raw = tiny_config()
    raw.pop("auto_map")
    raw["text_config"].update(head_dim=head_dim, num_attention_heads=heads)
    assert not adapter().is_teleocr_config(raw)


def test_constructor_preserves_parent_initialization_and_custom_components(monkeypatch):
    from mlx_vlm.models.qwen2_5_vl.qwen2_5_vl import Model as QwenModel

    a = adapter()
    original = QwenModel.__init__
    upstream_state = object()

    def initialize(self, config):
        original(self, config)
        self.upstream_state = upstream_state

    monkeypatch.setattr(QwenModel, "__init__", initialize)
    model = a.Model(a.ModelConfig.from_dict(tiny_config()))
    assert model.upstream_state is upstream_state
    assert isinstance(model.vision_tower, a.VisionModel)
    assert isinstance(model.language_model, a.LanguageModel)
    assert model.layers[0].self_attn.q_proj.weight.shape == (12, 8)
    assert "bias" not in model.layers[0].self_attn.q_proj


def test_attention_replacement_preserves_other_decoder_layer_state(monkeypatch):
    from mlx_vlm.models.qwen2_5_vl.language import LanguageModel as QwenLanguageModel

    original = QwenLanguageModel.__init__
    inherited_state = []

    def initialize(self, args, config):
        original(self, args, config)
        inherited_state.clear()
        for layer in self.model.layers:
            layer.upstream_state = object()
            inherited_state.append(
                (
                    {k: v for k, v in layer.items() if k != "self_attn"},
                    layer.upstream_state,
                )
            )

    monkeypatch.setattr(QwenLanguageModel, "__init__", initialize)
    a = adapter()
    model = a.Model(a.ModelConfig.from_dict(tiny_config()))
    for layer, (state, extra) in zip(model.layers, inherited_state, strict=True):
        assert layer.upstream_state is extra
        assert {k for k in layer if k != "self_attn"} == set(state)
        assert all(layer[key] is value for key, value in state.items())


def test_upstream_attention_surface_requires_review_on_pin_drift():
    # The adapter replaces this module. A new upstream state field (e.g. a
    # window policy) must prompt a compatibility audit before another pin lands.
    from mlx_vlm.models.qwen2_5_vl.language import Attention as QwenAttention

    a = adapter()
    upstream = QwenAttention(a.ModelConfig.from_dict(tiny_config()).text_config)
    public_state = set(upstream) | {
        name for name in vars(upstream) if not name.startswith("_")
    }
    assert public_state == {
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "scale",
        "rope_scaling",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "rotary_emb",
    }


def test_concurrent_install_keeps_one_dispatcher_and_stock_routing(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import mlx_vlm.utils as utils

    a = adapter()
    original = utils.get_model_and_args
    while getattr(original, "_omlx_teleocr_compat", False):
        original = original.__wrapped__
    monkeypatch.setattr(utils, "get_model_and_args", original)
    monkeypatch.setattr(a, "_DISPATCHER_INSTALLED", False, raising=False)
    barrier = Barrier(8)

    def install(_):
        barrier.wait()
        a.apply_mlx_vlm_teleocr_compat_patch()
        return utils.get_model_and_args

    with ThreadPoolExecutor(8) as executor:
        wrappers = list(executor.map(install, range(8)))
    assert all(wrapper is wrappers[0] for wrapper in wrappers)
    assert wrappers[0].__wrapped__ is original
    ordinary = {"model_type": "qwen2_5_vl"}
    assert wrappers[0](ordinary) == original(ordinary)


def test_later_opaque_wrapper_is_preserved_without_reinstallation(monkeypatch):
    import mlx_vlm.utils as utils

    a = adapter()
    a.apply_mlx_vlm_teleocr_compat_patch()
    installed = utils.get_model_and_args

    def later_wrapper(config):
        return installed(config)

    monkeypatch.setattr(utils, "get_model_and_args", later_wrapper)
    for _ in range(3):
        a.apply_mlx_vlm_teleocr_compat_patch()
    assert utils.get_model_and_args is later_wrapper
    assert utils.get_model_and_args(tiny_config())[0].Model is a.Model


@pytest.mark.parametrize("chunks", [[3, 3], [2, 2, 2]])
def test_cached_array_masks_cover_prefix_and_match_full_attention(chunks):
    a = adapter()
    attn = a.Attention(a.ModelConfig.from_dict(tiny_config()).text_config)
    rng = np.random.default_rng(13)
    x = mx.array(rng.normal(size=(1, 6, 8)).astype(np.float32))
    full = attn(x, mask="causal")
    cache = KVCache()
    outputs = []
    offset = 0
    for length in chunks:
        # The installed cache API includes cached keys in array masks.
        mask = cache.make_mask(length, return_array=True, window_size=None)
        assert mask.shape[-1] == offset + length
        outputs.append(attn(x[:, offset : offset + length], mask=mask, cache=cache))
        offset += length
    np.testing.assert_allclose(
        np.array(mx.concatenate(outputs, axis=1)), np.array(full), rtol=2e-5, atol=2e-5
    )


def test_attention_matches_independent_numpy_mrope_reference():
    a = adapter()
    attn = a.Attention(a.ModelConfig.from_dict(tiny_config()).text_config)
    rng = np.random.default_rng(42)
    weights = {
        name: mx.array(rng.normal(0, 0.2, w.shape).astype(np.float32))
        for name, w in tree_flatten(attn.parameters())
    }
    attn.load_weights(list(weights.items()))
    x = rng.normal(size=(1, 4, 8)).astype(np.float32)
    positions = np.array([[[0, 1, 3, 4]], [[0, 2, 2, 5]], [[0, 3, 1, 6]]])

    def projection(name, heads):
        return (
            (x @ np.array(weights[name + ".weight"]).T)
            .reshape(1, 4, heads, 6)
            .transpose(0, 2, 1, 3)
        )

    q, k, v = projection("q_proj", 2), projection("k_proj", 1), projection("v_proj", 1)

    def norm(z, name):
        return (
            z
            / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + 1e-6)
            * np.array(weights[name + ".weight"])
        )

    q, k = norm(q, "q_norm"), norm(k, "k_norm")
    frequencies = 10000.0 ** (-np.arange(0, 6, 2) / 6)
    phase = np.stack([positions[i] * frequencies[i] for i in range(3)], axis=-1)
    phase = np.concatenate([phase, phase], axis=-1)

    def rotate(z):
        rotated = np.concatenate([-z[..., 3:], z[..., :3]], axis=-1)
        return z * np.cos(phase) + rotated * np.sin(phase)

    q, k = rotate(q), rotate(k)
    scores = q @ np.repeat(k, 2, axis=1).swapaxes(-1, -2) / np.sqrt(6.0)
    scores = np.where(np.tril(np.ones((4, 4), dtype=bool)), scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = (probs @ np.repeat(v, 2, axis=1)).transpose(0, 2, 1, 3).reshape(1, 4, 12)
    expected = expected @ np.array(weights["o_proj.weight"]).T
    actual = attn(mx.array(x), mask="causal", position_ids=mx.array(positions))
    np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-5)


def test_full_and_chunked_cached_logits_agree_and_strict_raw_weights_reload():
    a = adapter()
    raw_config = tiny_config()
    raw_config["tie_word_embeddings"] = False
    cfg = a.ModelConfig.from_dict(raw_config)
    model = a.Model(cfg)
    weights = dict(tree_flatten(model.parameters()))
    assert "language_model.lm_head.weight" in weights
    raw = {}
    for key, value in weights.items():
        if key.startswith("language_model.model."):
            key = key.replace("language_model.model.", "model.language_model.", 1)
        elif key.startswith("language_model.lm_head."):
            key = key.replace("language_model.lm_head.", "lm_head.", 1)
        elif key.startswith("vision_tower."):
            key = key.replace("vision_tower.", "model.visual.", 1)
            if "patch_embed.proj.weight" in key:
                value = value.transpose(0, 4, 1, 2, 3)
        raw[key] = value
    normalized = a.Model.sanitize(SimpleNamespace(config=cfg), raw)
    normalized = a.VisionModel.sanitize(
        SimpleNamespace(config=cfg.vision_config), normalized
    )
    again = a.Model.sanitize(SimpleNamespace(config=cfg), normalized)
    again = a.VisionModel.sanitize(SimpleNamespace(config=cfg.vision_config), again)
    assert set(again) == set(weights)
    for key in weights:
        np.testing.assert_array_equal(np.array(again[key]), np.array(weights[key]))
    clone = a.Model(cfg)
    clone.load_weights(list(again.items()), strict=True)
    ids = mx.array([[1, 2, 3, 4, 5]])
    full = model(ids).logits
    cache = [KVCache() for _ in clone.layers]
    first = clone(ids[:, :3], cache=cache).logits
    second = clone(ids[:, 3:4], cache=cache).logits
    last = clone(ids[:, 4:], cache=cache).logits
    np.testing.assert_allclose(
        np.array(mx.concatenate([first, second, last], axis=1)),
        np.array(full),
        rtol=2e-5,
        atol=2e-5,
    )
    assert clone.language_model.head_dim == 6


def test_explicit_root_tied_head_drops_redundant_checkpoint_weights():
    a = adapter()
    raw = tiny_config()
    raw["tie_word_embeddings"] = True
    cfg = a.ModelConfig.from_dict(raw)
    model = a.Model(cfg)
    tensors = dict(tree_flatten(model.parameters()))
    raw_weights = {
        "lm_head.weight": tensors["language_model.model.embed_tokens.weight"]
    }
    for key, value in tensors.items():
        key = key.replace("language_model.model.", "model.", 1)
        key = key.replace("vision_tower.", "visual.", 1)
        raw_weights[key] = value
    sanitized = a.Model.sanitize(SimpleNamespace(config=cfg), raw_weights)
    model.load_weights(list(sanitized.items()), strict=True)
    assert model(mx.array([[1, 2]])).logits.shape == (1, 2, 20)


@pytest.mark.parametrize("head_location", ["nested", "root"])
def test_mlx_vlm_nested_config_update_keeps_independent_head(head_location):
    from mlx_vlm.utils import update_module_configs

    a = adapter()
    raw = tiny_config()
    raw["tie_word_embeddings"] = False
    if head_location == "root":
        raw["head_dim"] = raw["text_config"].pop("head_dim")
    cfg = update_module_configs(
        a.ModelConfig.from_dict(raw), a, raw, ["text", "vision"]
    )
    model = a.Model(cfg)
    assert model.language_model.lm_head.weight.shape == (20, 8)
    assert model.layers[0].self_attn.q_proj.weight.shape == (12, 8)


def test_image_prefill_and_cached_decode_match_full_multimodal_logits():
    """Image positions and their decode offset survive the vision/cache boundary."""
    a = adapter()
    raw = tiny_config()
    raw.update(image_token_id=17, vision_start_token_id=18, vision_end_token_id=19)
    raw["vision_config"]["spatial_merge_size"] = 2
    with mx.stream(mx.cpu):
        cfg = a.ModelConfig.from_dict(raw)
        model = a.Model(cfg)
        rng = np.random.default_rng(731)
        # Deterministic nonzero parameters make ignoring pixels or their spatial
        # positions visible, while keeping all computations in FP32.
        weights = []
        for key, value in tree_flatten(model.parameters()):
            if "norm" in key and key.endswith("weight"):
                array = rng.uniform(0.8, 1.2, value.shape)
            else:
                array = rng.normal(0, 0.2, value.shape)
            weights.append((key, mx.array(array.astype(np.float32))))
        model.load_weights(weights, strict=True)
        cached_model = a.Model(a.ModelConfig.from_dict(raw))
        cached_model.load_weights(weights, strict=True)
        # Sixteen 2x2 temporal patches merge into four image placeholders.
        pixels = mx.array(rng.normal(size=(16, 24)).astype(np.float32))
        grid = mx.array([[1, 4, 4]])
        ids = mx.array([[1, 18, 17, 17, 17, 17, 19, 2, 3, 4]])
        positions, delta = model.language_model.get_rope_index(ids, grid)
        np.testing.assert_array_equal(
            np.array(positions[:, 0, 2:6]),
            [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]],
        )
        np.testing.assert_array_equal(np.array(delta), [[-2]])
        full = model(ids, pixel_values=pixels, image_grid_thw=grid).logits
        cache = [KVCache() for _ in cached_model.layers]
        prefill = cached_model(
            ids[:, :8], pixel_values=pixels, image_grid_thw=grid, cache=cache
        ).logits
        decode = [cached_model(ids[:, i : i + 1], cache=cache).logits for i in (8, 9)]
        actual = mx.concatenate([prefill, *decode], axis=1)
        np.testing.assert_allclose(
            np.array(actual), np.array(full), rtol=2e-5, atol=2e-5
        )
        # A text-only path can also be internally cache-consistent; verify the
        # image embedding merge actually contributes to the observed logits.
        text_only = model(ids).logits
        assert np.max(np.abs(np.array(full - text_only))) > 1e-3
