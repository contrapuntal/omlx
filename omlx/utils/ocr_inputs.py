# SPDX-License-Identifier: Apache-2.0
"""Model-scoped OCR image-mode input resolution and cache identity."""

import hashlib

from ..exceptions import InvalidRequestError


def resolve_ocr_image_kwargs(
    model_type: str | None,
    images_config: dict | None,
    num_images: int = 0,
) -> dict:
    """Validate explicit image-mode controls and resolve Unlimited-OCR's native modes.

    Args:
        model_type: Architecture type of the model (e.g., 'unlimited-ocr').
        images_config: Dictionary containing explicit image mode parameters.
        num_images: Number of images in the request.

    Returns:
        dict of kwargs for mlx_vlm.prepare_inputs (cropping, base_size, image_size).

    Raises:
        InvalidRequestError: If images_config is supplied on an unsupported model,
            lacks image inputs, or contains invalid mode fields/values.
    """
    if images_config is not None:
        if model_type != "unlimited-ocr":
            raise InvalidRequestError(
                "images_config is supported only for Unlimited-OCR.",
                field="images_config",
            )
        if (
            not isinstance(images_config, dict)
            or set(images_config) != {"image_mode"}
            or images_config["image_mode"] not in ("base", "gundam")
        ):
            raise InvalidRequestError(
                "images_config must contain image_mode='base' or 'gundam'.",
                field="images_config",
            )
        if not num_images:
            raise InvalidRequestError(
                "images_config requires image input.",
                field="images_config",
            )

    if model_type != "unlimited-ocr":
        return {}

    mode = images_config["image_mode"] if images_config is not None else "gundam"
    return {
        "cropping": mode == "gundam",
        "base_size": 1024,
        "image_size": 640 if mode == "gundam" else 1024,
    }


def image_mode_cache_key(image_hash: str, image_kwargs: dict) -> str:
    """Partition image cache identity by resolved preprocessing mode.

    Independent of KV cache ring migration tags. Non-Unlimited-OCR models
    (empty image_kwargs) preserve the raw image hash unchanged.
    For Unlimited-OCR, both explicit and default (omitted images_config)
    configurations resolve to mode-specific kwargs (gundam vs base),
    ensuring cross-mode cache isolation while allowing unconfigured requests
    to share cache with explicit gundam requests.
    """
    if not image_kwargs:
        return image_hash
    mode_tag = "gundam" if image_kwargs.get("cropping", True) else "base"
    identity = f"imgmode:{mode_tag}:{image_kwargs.get('base_size')}:{image_kwargs.get('image_size')}:{image_hash}"
    return hashlib.sha256(identity.encode()).hexdigest()
