"""TeleOCR conversion entry points must install the custom model themselves."""

import json
import logging
import os
import subprocess
import sys

import pytest


def teleocr_config():
    text = {
        "model_type": "qwen2_5_vl_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 1024,
        "rope_theta": 1000000.0,
        "rope_scaling": {"mrope_section": [8, 12, 12], "type": "default"},
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    return {
        **text,
        "model_type": "qwen2_5_vl",
        "text_config": text,
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "auto_map": {
            "AutoModel": "modeling_naviocr.Qwen2_5_VLForConditionalGeneration"
        },
        "vision_config": {
            "model_type": "qwen2_5_vl",
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "out_hidden_size": 64,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": [0],
            "window_size": 112,
        },
        "image_token_id": 120,
        "video_token_id": 121,
        "vision_start_token_id": 122,
        "vision_end_token_id": 123,
    }


def save_attention_weights(path, config, *, missing_norm=False, biased=False):
    import numpy as np
    from safetensors.numpy import save_file

    text = config["text_config"]
    hidden, width = text["hidden_size"], text["head_dim"]
    query = width * text["num_attention_heads"]
    kv = width * text["num_key_value_heads"]
    weights = {}
    for index in range(text["num_hidden_layers"]):
        prefix = f"model.layers.{index}.self_attn."
        for name, shape in {
            "q_proj.weight": (query, hidden),
            "k_proj.weight": (kv, hidden),
            "v_proj.weight": (kv, hidden),
            "o_proj.weight": (hidden, query),
            "q_norm.weight": (width,),
            "k_norm.weight": (width,),
        }.items():
            weights[prefix + name] = np.zeros(shape, dtype=np.float32)
    if missing_norm:
        weights.pop("model.layers.0.self_attn.q_norm.weight")
    if biased:
        weights["model.layers.0.self_attn.q_proj.bias"] = np.zeros(
            query, dtype=np.float32
        )
    save_file(weights, str(path / "model.safetensors"))


def test_unknown_geometry_never_stamps_teleocr_marker():
    from omlx.oq import _stamp_adapter_marker

    config = teleocr_config()
    config.pop("auto_map")
    _stamp_adapter_marker(config)
    assert "omlx_adapter" not in config


@pytest.mark.parametrize("entrypoint", ["validator", "streaming", "proxy"])
@pytest.mark.parametrize(
    "escape", ["absolute", "parent", "symlink", "unindexed_symlink"]
)
def test_checkpoint_shards_cannot_escape_model_directory(
    tmp_path, monkeypatch, entrypoint, escape
):
    import safetensors

    from omlx.oq import _build_streaming_proxy_for_sensitivity, quantize_oq_streaming
    from omlx.patches.mlx_vlm_teleocr_compat.checkpoint import (
        validate_teleocr_checkpoint,
    )

    config = teleocr_config()
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    save_attention_weights(source, config)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes((source / "model.safetensors").read_bytes())
    name = str(outside) if escape == "absolute" else "../outside.safetensors"
    if escape in ("symlink", "unindexed_symlink"):
        name = "escape.safetensors"
        (source / name).symlink_to(outside)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "model.safetensors"
                    if escape == "unindexed_symlink"
                    else name
                }
            }
        )
    )
    opened = []
    original = safetensors.safe_open

    def record_open(path, *args, **kwargs):
        from pathlib import Path

        opened.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", record_open)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="escapes model directory"):
        if entrypoint == "validator":
            validate_teleocr_checkpoint(source, config)
        elif entrypoint == "streaming":
            quantize_oq_streaming(str(source), str(output), oq_level=8)
        else:
            _build_streaming_proxy_for_sensitivity(
                str(source), str(output), dtype="bfloat16"
            )
    assert outside.resolve() not in opened
    assert not output.exists()


@pytest.mark.parametrize("indexed", [True, False])
def test_checkpoint_accepts_huggingface_snapshot_blob_links(tmp_path, indexed):
    from omlx.patches.mlx_vlm_teleocr_compat.checkpoint import (
        validate_teleocr_checkpoint,
    )

    config = teleocr_config()
    repo = tmp_path / "models--vendor--model"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir()
    save_attention_weights(blobs, config)
    (snapshot / "model.safetensors").symlink_to("../../blobs/model.safetensors")
    if indexed:
        (blobs / "index").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}})
        )
        (snapshot / "model.safetensors.index.json").symlink_to("../../blobs/index")
    validate_teleocr_checkpoint(snapshot, config)


def test_checkpoint_rejects_snapshot_blob_directory_escape(tmp_path):
    from omlx.patches.mlx_vlm_teleocr_compat.checkpoint import (
        validate_teleocr_checkpoint,
    )

    config = teleocr_config()
    repo = tmp_path / "models--vendor--model"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    save_attention_weights(outside, config)
    (repo / "blobs").symlink_to(outside, target_is_directory=True)
    (snapshot / "model.safetensors").symlink_to("../../blobs/model.safetensors")
    with pytest.raises(ValueError, match="escapes model directory"):
        validate_teleocr_checkpoint(snapshot, config)


@pytest.mark.parametrize("enhanced", [False, True])
def test_text_only_teleocr_conversion_refused_before_weight_reads(tmp_path, enhanced):
    from omlx.oq import quantize_oq_streaming

    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(teleocr_config()))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="TeleOCR.*text-only conversion"):
        quantize_oq_streaming(
            str(source), str(output), oq_level=8, enhanced=enhanced, text_only=True
        )
    assert not output.exists()


@pytest.mark.parametrize("defect", ["missing_norm", "biased"])
@pytest.mark.parametrize("entrypoint", ["serving", "sanitizer"])
def test_claimed_adapter_requires_attention_weight_evidence(
    tmp_path, defect, entrypoint
):
    from omlx.oq import _build_model_sanitizer
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    config = teleocr_config()
    config.pop("auto_map")
    config["omlx_adapter"] = "teleocr"
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_attention_weights(tmp_path, config, **{defect: True})
    with pytest.raises(ValueError, match="TeleOCR.*(q_norm|bias)"):
        if entrypoint == "serving":
            maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
        else:
            _build_model_sanitizer(config, model_path=tmp_path)


def test_invalid_teleocr_config_cannot_fall_back_to_llm_sanitizer(tmp_path):
    from omlx.oq import _build_model_sanitizer

    config = teleocr_config()
    config["text_config"]["rope_scaling"] = {"type": "unsupported"}
    save_attention_weights(tmp_path, config)
    with pytest.raises(ValueError, match="TeleOCR.*sanitize"):
        _build_model_sanitizer(config, model_path=tmp_path)


@pytest.mark.asyncio
async def test_vlm_start_propagates_checkpoint_validation_error(tmp_path, monkeypatch):
    import mlx_vlm.utils as utils

    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.patches.mlx_vlm_teleocr_compat import TeleOCRCompatibilityError

    config = teleocr_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_attention_weights(tmp_path, config, missing_norm=True)

    def unexpected_load(*args, **kwargs):
        pytest.fail("native loader reached despite failed checkpoint validation")

    monkeypatch.setattr(utils, "load", unexpected_load)
    engine = VLMBatchedEngine(str(tmp_path))
    with pytest.raises(TeleOCRCompatibilityError, match="q_norm"):
        await engine.start()
    assert not engine._loaded
    assert engine._vlm_model is None


@pytest.mark.parametrize("entrypoint", ["serving", "sanitizer"])
@pytest.mark.parametrize("metadata", [True, False])
def test_fresh_process_selects_teleocr(entrypoint, metadata, tmp_path):
    # A prior serving load must not be required before standalone conversion.
    config = teleocr_config()
    if not metadata:
        config.pop("auto_map")
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_attention_weights(tmp_path, config)
    script = """
import json, sys
from pathlib import Path
import mlx.core as mx
path, entrypoint = sys.argv[1:]
config = json.loads((Path(path) / 'config.json').read_text())
if entrypoint == 'serving':
    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    maybe_apply_pre_load_patches(path, for_vlm=True)
else:
    from omlx.oq import _build_model_sanitizer
    sanitize = _build_model_sanitizer(config, model_path=path)
    result = sanitize({'model.language_model.layers.0.self_attn.q_norm.weight': mx.ones((64,))})
    assert set(result) == {'language_model.model.layers.0.self_attn.q_norm.weight'}, set(result)
from mlx_vlm.utils import get_model_and_args
module, _ = get_model_and_args(config)
assert 'teleocr' in module.__name__, module.__name__
ordinary, _ = get_model_and_args({'model_type': 'qwen2_5_vl'})
assert ordinary.__name__ == 'mlx_vlm.models.qwen2_5_vl', ordinary.__name__
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), entrypoint],
        text=True,
        capture_output=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    if metadata:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "Unrecognized Qwen2.5-VL attention geometry" in result.stderr


def test_preload_logs_teleocr_selection(tmp_path, caplog):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    config = teleocr_config()
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_attention_weights(tmp_path, config)
    with caplog.at_level(logging.INFO, logger="omlx.utils.model_loading"):
        maybe_apply_pre_load_patches(str(tmp_path), for_vlm=True)
    assert any(
        "TeleOCR" in record.message and str(tmp_path) in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("export", ["oq8e", "proxy"])
def test_conversion_exports_adapter_and_reloads(tmp_path, tied, export):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from omlx.oq import _build_streaming_proxy_for_sensitivity, quantize_oq_streaming
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    source = tmp_path / "source"
    source.mkdir()
    config = teleocr_config()
    if tied:
        # The reference Transformers 4.57 loader hoists the nested true flag.
        config.pop("tie_word_embeddings")
        config["text_config"]["tie_word_embeddings"] = True
    config["eos_token_id"] = 1
    (source / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[EOS]": 1, "the": 2, "a": 3}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    ).save_pretrained(source)

    from mlx_vlm.utils import load_model

    from omlx.patches.mlx_vlm_teleocr_compat import model as module

    model = module.Model(module.ModelConfig.from_dict(config))
    model.set_dtype(mx.bfloat16)
    # Feed raw HF names through the actual conversion sanitizer. The vision
    # Conv3D layout is PyTorch's [out, in, time, height, width] on disk.
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        name = name.replace("language_model.model.", "model.")
        name = name.replace("language_model.lm_head.", "lm_head.")
        name = name.replace("vision_tower.", "visual.")
        if name == "visual.patch_embed.proj.weight":
            value = value.transpose(0, 4, 1, 2, 3)
        weights[name] = value
    if tied:
        # The public checkpoint carries both identical BF16 copies.
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
    mx.save_safetensors(str(source / "model.safetensors"), weights)
    maybe_apply_pre_load_patches(str(source), for_vlm=True)

    output = tmp_path / "converted"
    if export == "oq8e":
        quantize_oq_streaming(
            str(source),
            str(output),
            oq_level=8,
            enhanced=True,
            imatrix_strict=True,
            imatrix_num_samples=1,
            imatrix_seq_length=8,
        )
    else:
        _build_streaming_proxy_for_sensitivity(str(source), output, dtype="bfloat16")
    saved_config = json.loads((output / "config.json").read_text())
    assert saved_config["omlx_adapter"] == "teleocr"
    assert saved_config["tie_word_embeddings"] is tied
    assert saved_config["auto_map"] == config["auto_map"]
    if export == "oq8e":
        report = json.loads((output / "oq_imatrix_report.json").read_text())
        assert not report["missing"]
        for index in range(2):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert any(
                    f"layers.{index}.self_attn.{name}" in key
                    for key in report["applied"]
                )
        assert any("lm_head" in key for key in report["applied"]) is (not tied)
    # The explicit export marker must survive loss of vendor remote-code metadata.
    saved_config.pop("auto_map")
    (output / "config.json").write_text(json.dumps(saved_config))
    restored = load_model(output, lazy=False, strict=True)
    assert isinstance(
        restored.language_model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear
    )
    assert isinstance(restored.vision_tower.blocks[0].attn.qkv, nn.Linear)
    logits = restored(mx.array([[2, 3, 2]])).logits
    assert logits.shape == (1, 3, 128)
    assert bool(mx.all(mx.isfinite(logits)))
