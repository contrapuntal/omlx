# SPDX-License-Identifier: Apache-2.0
"""Keep a single <image> token in Unlimited-OCR prompts that already carry one."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Config model_type carried by baidu/Unlimited-OCR checkpoints.
_MODEL_TYPE = "unlimited-ocr"

_APPLIED = False

_LEADING_PAGE_MARKERS_RE = re.compile(r"^\s*(?:<PAGE>\s*)+")
_INVERTED_IMAGE_PAGE_RE = re.compile(r"^\s*(?:<image>\s*)+(?:<PAGE>\s*)+")


_UNLIMITED_OCR_DEFAULT_TEMPLATE = (
    "{% for message in messages %}"
    "{{message['content']}}"
    "{% if not loop.last or message['role'] != 'user' %} {% endif %}"
    "{% endfor %}"
)


class _UnlimitedOCRDefaultChatTemplateProperty(property):
    """Property descriptor carrying the oMLX PR #2164 compatibility marker."""

    _omlx_compat_2164: bool = True


def apply_mlx_vlm_unlimited_ocr_compat_patch() -> bool:
    """Wrap mlx-vlm's get_message_json and UnlimitedOCRProcessor helpers."""
    global _APPLIED
    if _APPLIED:
        _patch_processor_chat_template()
        _patch_processor_page_prompt()
        return False

    try:
        import mlx_vlm.prompt_utils as prompt_utils

        _patch_prompt_utils(prompt_utils)
        _patch_processor_chat_template()
        _patch_processor_page_prompt()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unlimited-OCR mlx-vlm compat patch failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("Unlimited-OCR mlx-vlm compatibility patch applied")
    return True


def _patch_processor_chat_template() -> bool:
    """Apply the mlx-vlm PR #2164 default_chat_template fix to UnlimitedOCRProcessor.

    - Idempotent: safe to call repeatedly without re-patching or stacking descriptors.
    - No-op when upstream fix is present: if UnlimitedOCRProcessor already defines its own
      default_chat_template (without our marker), it leaves it untouched.
    - Preserves explicit template overrides: instance attributes and explicit chat_template
      overrides take precedence.
    """
    try:
        from mlx_vlm.models.unlimited_ocr.processing_unlimitedocr import (
            UnlimitedOCRProcessor,
        )
    except ImportError:
        return False

    existing = UnlimitedOCRProcessor.__dict__.get("default_chat_template")
    if existing is not None:
        if getattr(existing, "_omlx_compat_2164", False):
            # Already patched by oMLX - idempotent no-op
            return False
        # Upstream mlx-vlm defines its own default_chat_template - no-op
        return False

    def _native_template(self):
        return _UNLIMITED_OCR_DEFAULT_TEMPLATE

    UnlimitedOCRProcessor.default_chat_template = (
        _UnlimitedOCRDefaultChatTemplateProperty(_native_template)
    )
    return True


def _patch_processor_page_prompt() -> bool:
    """Repair inverted ``<image><PAGE>`` prompts on Unlimited-OCR-2 processors."""
    try:
        from mlx_vlm.models.unlimited_ocr.processing_unlimitedocr import (
            UnlimitedOCRProcessor,
        )
    except ImportError:
        return False

    orig_page_prompt = getattr(UnlimitedOCRProcessor, "page_prompt", None)
    if orig_page_prompt is None or getattr(
        orig_page_prompt, "_omlx_page_prompt_compat", False
    ):
        return False

    def patched_page_prompt(self, prompt: str, num_images: int) -> str:
        if (
            getattr(self, "page_layout", False)
            and num_images
            and isinstance(prompt, str)
            and "<PAGE>" in prompt
        ):
            prompt = _repair_unlimited_ocr_page_prompt(
                prompt, self.image_token, num_images
            )
        return orig_page_prompt(self, prompt, num_images)

    patched_page_prompt._omlx_page_prompt_compat = True
    patched_page_prompt._omlx_original = orig_page_prompt
    UnlimitedOCRProcessor.page_prompt = patched_page_prompt
    return True


def _repair_unlimited_ocr_page_prompt(
    text: str, image_token: str, num_images: int
) -> str:
    """Keep ``<PAGE>`` directly before ``<image>`` in Unlimited-OCR-2 prompts."""
    page_image = f"<PAGE>{image_token}"
    if image_token not in text:
        page_match = _LEADING_PAGE_MARKERS_RE.match(text)
        if page_match is not None:
            return page_image * num_images + text[page_match.end() :]
        return f"{image_token}{text}"
    if "<PAGE>" in text:
        inv_pattern = (
            _INVERTED_IMAGE_PAGE_RE
            if image_token == "<image>"
            else re.compile(rf"^\s*(?:{re.escape(image_token)}\s*)+(?:<PAGE>\s*)+")
        )
        inv_match = inv_pattern.match(text)
        if inv_match is not None:
            return page_image * num_images + text[inv_match.end() :]
        return re.sub(rf"<PAGE>\s+{re.escape(image_token)}", page_image, text)
    return text


def is_applied() -> bool:
    return _APPLIED


def _patch_prompt_utils(prompt_utils: Any) -> None:
    original = getattr(prompt_utils, "get_message_json", None)
    if original is None or getattr(original, "_omlx_unlimited_ocr_compat", False):
        return

    def patched_get_message_json(
        model_name: str,
        prompt: str,
        role: str = "user",
        skip_image_token: bool = False,
        skip_audio_token: bool = False,
        num_images: int = 0,
        num_audios: int = 0,
        **kwargs,
    ):
        if _is_unlimited_ocr(model_name):
            content = prompt_utils.extract_text_from_content(prompt)
            # Unlimited-OCR uses a single literal <image> token for one or many
            # pages, matching upstream infer_multi (MessageFormat.SINGLE_IMAGE_TOKEN).
            # When the caller supplies leading <PAGE> markers (Unlimited-OCR-2),
            # preserve <PAGE><image> order rather than inverting to <image><PAGE>.
            if role == "user" and not skip_image_token and num_images > 0:
                content = _repair_unlimited_ocr_page_prompt(
                    content, "<image>", num_images
                )
            return {"role": role, "content": content}
        return original(
            model_name,
            prompt,
            role=role,
            skip_image_token=skip_image_token,
            skip_audio_token=skip_audio_token,
            num_images=num_images,
            num_audios=num_audios,
            **kwargs,
        )

    patched_get_message_json._omlx_unlimited_ocr_compat = True
    patched_get_message_json._omlx_original = original
    prompt_utils.get_message_json = patched_get_message_json


def _is_unlimited_ocr(model_name: Any) -> bool:
    return (
        isinstance(model_name, str)
        and model_name.lower().replace("_", "-") == _MODEL_TYPE
    )


__all__ = [
    "apply_mlx_vlm_unlimited_ocr_compat_patch",
    "is_applied",
    "_patch_processor_chat_template",
    "_patch_processor_page_prompt",
]
