# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.processor module."""

from types import SimpleNamespace

from omlx.utils.processor import repair_processor_multimodal_token_ids


class TestRepairProcessorMultimodalTokenIds:
    """Test cases for repair_processor_multimodal_token_ids.

    The helper is intended for synthetic Qwen3-VL processors that
    `mlx-embeddings` builds via `object.__new__`, skipping
    `ProcessorMixin.__init__`. Without the repair, transformers'
    `apply_chat_template` -> `create_mm_token_type_ids` raises
    `AttributeError` on the missing `image_ids` / `video_ids` /
    `audio_ids` attributes.
    """

    def test_populates_missing_lists_from_token_id_attrs(self):
        """When `*_token_id` attrs exist, repair installs them as 1-element lists."""
        processor = SimpleNamespace(
            image_token_id=151655,
            video_token_id=151656,
            audio_token_id=151657,
        )
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [151655]
        assert processor.video_ids == [151656]
        assert processor.audio_ids == [151657]

    def test_falls_back_to_none_when_no_token_id_attr(self):
        """If neither `*_ids` nor `*_token_id` exist, install `[None]` so
        downstream iteration doesn't crash on AttributeError."""
        processor = SimpleNamespace()
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [None]
        assert processor.video_ids == [None]
        assert processor.audio_ids == [None]

    def test_preserves_existing_lists(self):
        """If `*_ids` are already populated, the helper does not overwrite them."""
        processor = SimpleNamespace(
            image_ids=[1, 2, 3],
            video_ids=[],
            audio_ids=[42],
            image_token_id=999,  # would be ignored — image_ids already exists
        )
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [1, 2, 3]
        assert processor.video_ids == []
        assert processor.audio_ids == [42]

    def test_repairs_inner_processor(self):
        """Wrapped processors expose the multimodal processor under
        `processor.processor`. The helper repairs both layers."""
        inner = SimpleNamespace(
            image_token_id=151655,
            video_token_id=151656,
        )
        outer = SimpleNamespace(processor=inner)
        repair_processor_multimodal_token_ids(outer)
        # Outer gets the fallback path
        assert outer.image_ids == [None]
        assert outer.video_ids == [None]
        assert outer.audio_ids == [None]
        # Inner gets the values from its own *_token_id
        assert inner.image_ids == [151655]
        assert inner.video_ids == [151656]
        assert inner.audio_ids == [None]

    def test_inner_pointing_to_self_is_not_double_processed(self):
        """A processor whose `processor` attribute is itself (self-loop) must
        not be repaired twice — that would no-op on the second pass but
        adds no value."""
        processor = SimpleNamespace(image_token_id=42)
        processor.processor = processor  # self-loop
        repair_processor_multimodal_token_ids(processor)
        # Behavior: image_ids set once. Repeated calls on the same object
        # leave the first-pass values in place (no overwrite).
        assert processor.image_ids == [42]

    def test_handles_token_id_zero(self):
        """Token id `0` is falsy but a valid token id; the helper still
        wraps it in a 1-element list (not coerced to `[None]`)."""
        processor = SimpleNamespace(
            image_token_id=0, video_token_id=0, audio_token_id=0
        )
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [0]
        assert processor.video_ids == [0]
        assert processor.audio_ids == [0]

    def test_no_inner_processor_attr(self):
        """When `processor` attribute is absent (most non-VL processors),
        only the outer processor is repaired."""
        processor = SimpleNamespace(image_token_id=151655)
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [151655]
        # video and audio fall back to [None] since their *_token_id absent
        assert processor.video_ids == [None]
        assert processor.audio_ids == [None]

    def test_inner_processor_is_none(self):
        """When `processor.processor is None`, only the outer is repaired
        (the candidates list excludes None)."""
        processor = SimpleNamespace(processor=None, image_token_id=1)
        repair_processor_multimodal_token_ids(processor)
        assert processor.image_ids == [1]
