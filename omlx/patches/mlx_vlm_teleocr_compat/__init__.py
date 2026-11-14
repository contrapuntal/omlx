"""Conditional MLX support for TeleOCR's custom Qwen2.5-VL decoder."""

from functools import wraps
from threading import Lock

_INSTALL_LOCK = Lock()
_DISPATCHER_INSTALLED = False


class TeleOCRCompatibilityError(ValueError):
    """An OCR checkpoint cannot safely use the selected implementation."""


def text_config_dict(config: dict) -> dict:
    text = dict(config)
    if isinstance(config.get("text_config"), dict):
        text.update(config["text_config"])
    return text


def is_teleocr_config(config: dict) -> bool:
    """Recognize an explicit architecture claim, subject to weight validation."""
    if config.get("model_type") != "qwen2_5_vl":
        return False
    if config.get("omlx_adapter") == "teleocr":
        return True
    auto_map = config.get("auto_map") or {}
    return isinstance(auto_map, dict) and any(
        value == "modeling_naviocr.Qwen2_5_VLForConditionalGeneration"
        for value in auto_map.values()
    )


def has_nonstandard_qwen_geometry(config: dict) -> bool:
    """Identify unsupported Qwen geometry without asserting a model identity."""
    if config.get("model_type") != "qwen2_5_vl":
        return False
    text = text_config_dict(config)
    hidden, heads, width = (
        text.get("hidden_size"),
        text.get("num_attention_heads"),
        text.get("head_dim"),
    )
    if width is None:
        return False
    if not all(type(value) is int and value > 0 for value in (hidden, heads, width)):
        return True
    return width != hidden // heads


def reject_unrecognized_geometry(config: dict) -> None:
    if has_nonstandard_qwen_geometry(config) and not is_teleocr_config(config):
        raise TeleOCRCompatibilityError(
            "Unrecognized Qwen2.5-VL attention geometry; explicit architecture "
            "metadata and matching weights are required. Refusing to assume TeleOCR."
        )


def vlm_fallback_refusal_reason(
    model_path: str, previous_reason: str | None = None
) -> str | None:
    """Refresh positive evidence; an unreadable config cannot erase prior evidence."""
    import json
    from pathlib import Path

    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except (OSError, ValueError):
        return previous_reason
    if not isinstance(config, dict):
        return previous_reason
    if is_teleocr_config(config):
        return "TeleOCR requires image conditioning"
    if has_nonstandard_qwen_geometry(config):
        return "unrecognized Qwen2.5-VL attention geometry"
    return None


def apply_mlx_vlm_teleocr_compat_patch() -> bool:
    """Install a locked process-wide dispatcher with per-checkpoint selection."""
    import mlx_vlm.utils as utils

    global _DISPATCHER_INSTALLED
    with _INSTALL_LOCK:
        # Later wrappers need not copy our function attributes. Install once
        # for this process; do not wrap their chain again on every model load.
        if _DISPATCHER_INSTALLED:
            return True
        original = utils.get_model_and_args
        if getattr(original, "_omlx_teleocr_compat", False):
            return True

        @wraps(original)
        def get_model_and_args(config, model_path=None):
            if is_teleocr_config(config):
                from . import model

                return model, "qwen2_5_vl"
            return original(config, model_path)

        get_model_and_args._omlx_teleocr_compat = True
        utils.get_model_and_args = get_model_and_args
        _DISPATCHER_INSTALLED = True
    return True


# Delay importing MLX until the model exports are requested. Detection is also
# used by conversion/configuration tools that do not initialize the GPU.
def __getattr__(name):
    if name in {
        "Attention",
        "ModelConfig",
        "TextConfig",
        "VisionConfig",
        "Model",
        "LanguageModel",
        "VisionModel",
    }:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
