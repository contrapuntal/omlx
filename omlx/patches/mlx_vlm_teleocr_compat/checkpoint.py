"""Header-only TeleOCR evidence checks; no model tensors or GPU allocation."""

import json
from pathlib import Path

from . import TeleOCRCompatibilityError, text_config_dict


def _confined_checkpoint_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Checkpoint path escapes model directory: {name}")
    resolved = (root / relative).resolve()
    if resolved.is_relative_to(root):
        return resolved
    # HF snapshots use ../../blobs/<hash> symlinks. Allow only this repository's
    # real blob directory, never an arbitrary symlink out of the model cache.
    repo = root.parent.parent
    blobs = repo / "blobs"
    if (
        root.parent.name == "snapshots"
        and repo.name.startswith("models--")
        and blobs.resolve() == blobs
        and resolved.parent == blobs
    ):
        return resolved
    raise ValueError(f"Checkpoint path escapes model directory: {name}")


def canonical_weight_key(key: str) -> str:
    # Specific aliases precede the generic model prefix. Match each key once.
    for source, target in (
        ("model.lm_head.", "language_model.lm_head."),
        ("lm_head.", "language_model.lm_head."),
        ("model.language_model.", "language_model.model."),
        ("model.visual.", "vision_tower."),
        ("visual.", "vision_tower."),
        ("model.", "language_model.model."),
    ):
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def validate_teleocr_checkpoint(model_path: str | Path, config: dict) -> None:
    """Verify every decoder's norms, bias absence and raw/affine tensor shapes.

    This corroborates a metadata claim before serving or conversion selects the
    decoder. Model.load_weights additionally enforces the complete strict binding
    contract, including vision, even for mlx-vlm's non-strict calibration callers.
    """
    from safetensors import safe_open

    path = Path(model_path).resolve()
    text = text_config_dict(config)

    def positive(name):
        value = text.get(name)
        if type(value) is not int or value <= 0:
            raise TeleOCRCompatibilityError(f"TeleOCR requires a positive {name}")
        return value

    hidden = positive("hidden_size")
    heads = positive("num_attention_heads")
    layers = positive("num_hidden_layers")
    width = text.get("head_dim")
    width = hidden // heads if width is None else positive("head_dim")
    kv_heads = text.get("num_key_value_heads")
    kv_heads = heads if kv_heads is None else positive("num_key_value_heads")
    if width <= 0 or heads % kv_heads:
        raise TeleOCRCompatibilityError("TeleOCR has invalid attention geometry")

    shapes = {}
    try:
        index = path / "model.safetensors.index.json"
        # Conversion also scans unindexed shards. Check that entire set before
        # any safe_open call, so an extra symlink cannot bypass index checks.
        names = {
            p.name
            for p in path.glob("*.safetensors")
            if p.name != "consolidated.safetensors"
        }
        if index.exists():
            index = _confined_checkpoint_file(path, index.name)
            weight_map = json.loads(index.read_text())["weight_map"]
            names.update(weight_map.values())
        files = [_confined_checkpoint_file(path, name) for name in sorted(names)]
        if not files:
            raise ValueError("no safetensors files")
        for file in files:
            with safe_open(str(file), framework="np") as tensors:
                for key in tensors.keys():  # noqa: SIM118 - safe_open is not iterable
                    canonical = canonical_weight_key(key)
                    if canonical in shapes:
                        raise ValueError(f"duplicate tensor alias {canonical}")
                    shapes[canonical] = tuple(tensors.get_slice(key).get_shape())
    except Exception as exc:
        raise TeleOCRCompatibilityError(
            f"TeleOCR checkpoint headers unavailable: {exc}"
        ) from exc

    def require(key, expected):
        if shapes.get(key) != expected:
            raise TeleOCRCompatibilityError(
                f"TeleOCR tensor {key}: expected {expected}, found {shapes.get(key)}"
            )

    quantization = config.get("quantization") or config.get("quantization_config") or {}
    if not isinstance(quantization, dict):
        raise TeleOCRCompatibilityError(
            "TeleOCR quantization metadata must be an object"
        )
    for layer in range(layers):
        prefix = f"language_model.model.layers.{layer}.self_attn."
        for norm in ("q_norm", "k_norm"):
            require(prefix + norm + ".weight", (width,))
        for projection, (out_dims, in_dims) in {
            "q_proj": (heads * width, hidden),
            "k_proj": (kv_heads * width, hidden),
            "v_proj": (kv_heads * width, hidden),
            "o_proj": (hidden, heads * width),
        }.items():
            key = prefix + projection
            if key + ".bias" in shapes:
                raise TeleOCRCompatibilityError(
                    f"TeleOCR attention must be bias-free: {key}.bias"
                )
            if key + ".scales" in shapes:
                params = quantization.get(key, quantization)
                if not isinstance(params, dict):
                    params = {}
                bits, group = params.get("bits"), params.get("group_size")
                if (
                    params.get("mode", "affine") != "affine"
                    or bits not in (2, 3, 4, 6, 8)
                    or type(group) is not int
                    or group <= 0
                    or in_dims % group
                    or (in_dims * bits) % 32
                ):
                    raise TeleOCRCompatibilityError(
                        f"TeleOCR has unsupported quantization for {key}"
                    )
                require(key + ".weight", (out_dims, in_dims * bits // 32))
                require(key + ".scales", (out_dims, in_dims // group))
                require(key + ".biases", (out_dims, in_dims // group))
            else:
                require(key + ".weight", (out_dims, in_dims))
