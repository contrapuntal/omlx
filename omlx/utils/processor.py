# SPDX-License-Identifier: Apache-2.0
"""Processor-level utility helpers shared across model engines."""

from typing import Any


def repair_processor_multimodal_token_ids(processor: Any) -> None:
    """Restore ProcessorMixin token-id lists on synthetic multimodal processors.

    `mlx-embeddings` constructs its Qwen3-VL processor via
    `object.__new__` and injects an `_UnsupportedVideoProcessor`
    placeholder for `processor.video_processor`, which bypasses
    `ProcessorMixin.__init__`. Newer transformers expect the `__init__`
    side effects — specifically the `image_ids` / `video_ids` /
    `audio_ids` lists — to exist when `apply_chat_template` later calls
    `create_mm_token_type_ids`. Without them the chat-template render
    raises `AttributeError`.

    Repair both the outer processor and an inner `processor` attribute (some
    wrappers expose the multimodal processor under `processor.processor`)
    by populating any missing `*_ids` list from the corresponding
    single-token id attribute. If neither is set, fall back to `[None]` so
    downstream iteration doesn't crash on `AttributeError`; the caller is
    responsible for the semantic correctness of `None` for that modality.
    """
    candidates = [processor]
    inner = getattr(processor, "processor", None)
    if inner is not None and inner is not processor:
        candidates.append(inner)

    for candidate in candidates:
        if not hasattr(candidate, "image_ids"):
            candidate.image_ids = [getattr(candidate, "image_token_id", None)]
        if not hasattr(candidate, "video_ids"):
            candidate.video_ids = [getattr(candidate, "video_token_id", None)]
        if not hasattr(candidate, "audio_ids"):
            candidate.audio_ids = [getattr(candidate, "audio_token_id", None)]
