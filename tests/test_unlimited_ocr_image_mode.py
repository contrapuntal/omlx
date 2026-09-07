# SPDX-License-Identifier: Apache-2.0
"""
Unit and regression tests for Unlimited-OCR explicit image preprocessing mode.

Tests:
1. Pydantic schema validation for ImagesConfig and ChatCompletionRequest.
2. Parameter resolution and model-type validation in resolve_ocr_image_kwargs.
3. Cache key identity partitioning via image_mode_cache_key.
4. Engine preflight, diffusion, and MarkItDown validation guards.
5. HTTP endpoint streaming validation (400/422 JSON before SSE headers, image aliases, forwarding).
6. VLM engine input preparation and behavioral cache isolation (whole-request, prefix-range, vision-cache).
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from omlx.api.openai_models import ChatCompletionRequest, ImagesConfig
from omlx.engine.base import GenerationOutput
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.server import app, create_chat_completion
from omlx.utils.image import compute_image_hash, compute_per_image_hashes
from omlx.utils.ocr_inputs import image_mode_cache_key, resolve_ocr_image_kwargs

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class TestImagesConfigSchema:
    """Pydantic schema validation for ImagesConfig and ChatCompletionRequest."""

    def test_valid_image_modes(self):
        """ImagesConfig accepts 'base' and 'gundam'."""
        cfg_base = ImagesConfig(image_mode="base")
        assert cfg_base.image_mode == "base"

        cfg_gundam = ImagesConfig(image_mode="gundam")
        assert cfg_gundam.image_mode == "gundam"

    def test_invalid_image_mode_raises(self):
        """ImagesConfig rejects unknown image modes."""
        with pytest.raises(ValidationError):
            ImagesConfig(image_mode="standard")

        with pytest.raises(ValidationError):
            ImagesConfig(image_mode="")

    def test_extra_fields_forbidden(self):
        """ImagesConfig forbids arbitrary extra fields."""
        with pytest.raises(ValidationError):
            ImagesConfig(image_mode="base", extra_param=123)

    def test_chat_completion_request_with_images_config(self):
        """ChatCompletionRequest parses images_config when supplied."""
        req = ChatCompletionRequest(
            model="Unlimited-OCR-oQ8",
            messages=[{"role": "user", "content": "hello"}],
            images_config={"image_mode": "base"},
        )
        assert req.images_config is not None
        assert req.images_config.image_mode == "base"

    def test_chat_completion_request_omitted_images_config(self):
        """ChatCompletionRequest defaults images_config to None."""
        req = ChatCompletionRequest(
            model="Unlimited-OCR-oQ8",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.images_config is None

    def test_chat_completion_request_invalid_images_config_raises(self):
        """ChatCompletionRequest rejects invalid images_config payload."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="Unlimited-OCR-oQ8",
                messages=[{"role": "user", "content": "hello"}],
                images_config={"image_mode": "invalid_mode"},
            )


class TestResolveOCRImageKwargs:
    """Validation and resolution in resolve_ocr_image_kwargs."""

    def test_unlimited_ocr_base_mode(self):
        """Unlimited-OCR with base mode yields cropping=False, image_size=1024."""
        kwargs = resolve_ocr_image_kwargs(
            model_type="unlimited-ocr",
            images_config={"image_mode": "base"},
            num_images=1,
        )
        assert kwargs == {
            "cropping": False,
            "base_size": 1024,
            "image_size": 1024,
        }

    def test_unlimited_ocr_gundam_mode(self):
        """Unlimited-OCR with gundam mode yields cropping=True, image_size=640."""
        kwargs = resolve_ocr_image_kwargs(
            model_type="unlimited-ocr",
            images_config={"image_mode": "gundam"},
            num_images=1,
        )
        assert kwargs == {
            "cropping": True,
            "base_size": 1024,
            "image_size": 640,
        }

    def test_unlimited_ocr_default_mode(self):
        """Unlimited-OCR without images_config defaults to gundam parameters."""
        kwargs = resolve_ocr_image_kwargs(
            model_type="unlimited-ocr",
            images_config=None,
            num_images=1,
        )
        assert kwargs == {
            "cropping": True,
            "base_size": 1024,
            "image_size": 640,
        }

    def test_non_unlimited_ocr_without_config_returns_empty(self):
        """Non-Unlimited-OCR models return empty kwargs when images_config is None."""
        assert resolve_ocr_image_kwargs("qwen2_vl", None, 1) == {}
        assert resolve_ocr_image_kwargs(None, None, 1) == {}
        assert resolve_ocr_image_kwargs("deepseekocr", None, 1) == {}

    def test_non_unlimited_ocr_with_config_raises(self):
        """Non-Unlimited-OCR models raise InvalidRequestError when images_config is set."""
        with pytest.raises(InvalidRequestError) as exc_info:
            resolve_ocr_image_kwargs(
                model_type="qwen2_vl",
                images_config={"image_mode": "base"},
                num_images=1,
            )
        assert exc_info.value.field == "images_config"
        assert "supported only for Unlimited-OCR" in exc_info.value.message

        with pytest.raises(InvalidRequestError) as exc_info2:
            resolve_ocr_image_kwargs(
                model_type=None,
                images_config={"image_mode": "base"},
                num_images=1,
            )
        assert exc_info2.value.field == "images_config"

    def test_images_config_without_images_raises(self):
        """Supplying images_config with 0 images raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            resolve_ocr_image_kwargs(
                model_type="unlimited-ocr",
                images_config={"image_mode": "base"},
                num_images=0,
            )
        assert exc_info.value.field == "images_config"
        assert "requires image input" in exc_info.value.message

    def test_malformed_images_config_dict_raises(self):
        """Malformed dictionary inputs raise InvalidRequestError."""
        # Extra keys
        with pytest.raises(InvalidRequestError):
            resolve_ocr_image_kwargs(
                "unlimited-ocr",
                {"image_mode": "base", "extra": "val"},
                num_images=1,
            )

        # Invalid mode string
        with pytest.raises(InvalidRequestError):
            resolve_ocr_image_kwargs(
                "unlimited-ocr",
                {"image_mode": "other"},
                num_images=1,
            )

        # Non-dict object
        with pytest.raises(InvalidRequestError):
            resolve_ocr_image_kwargs(
                "unlimited-ocr",
                "not_a_dict",  # type: ignore
                num_images=1,
            )


class TestImageModeCacheKey:
    """Cache identity partitioning via image_mode_cache_key."""

    def test_unconfigured_kwargs_returns_raw_hash(self):
        """Empty or falsy image_kwargs preserves the raw image hash unchanged (non-OCR models)."""
        raw_hash = "abc1234567890def"
        assert image_mode_cache_key(raw_hash, {}) == raw_hash
        assert image_mode_cache_key(raw_hash, None) == raw_hash  # type: ignore

    def test_unlimited_ocr_unconfigured_partitions_cache(self):
        """Unconfigured Unlimited-OCR resolves to gundam kwargs and partitions the cache identity."""
        raw_hash = "abc1234567890def"
        default_kwargs = resolve_ocr_image_kwargs("unlimited-ocr", None, num_images=1)
        key = image_mode_cache_key(raw_hash, default_kwargs)
        assert key != raw_hash
        expected = hashlib.sha256(f"imgmode:gundam:1024:640:{raw_hash}".encode()).hexdigest()
        assert key == expected

    def test_base_and_gundam_produce_distinct_cache_keys(self):
        """Identical raw hash produces distinct cache keys under base vs gundam."""
        raw_hash = "5f2d48a1c90e"
        base_kwargs = {"cropping": False, "base_size": 1024, "image_size": 1024}
        gundam_kwargs = {"cropping": True, "base_size": 1024, "image_size": 640}

        base_key = image_mode_cache_key(raw_hash, base_kwargs)
        gundam_key = image_mode_cache_key(raw_hash, gundam_kwargs)

        assert base_key != raw_hash
        assert gundam_key != raw_hash
        assert base_key != gundam_key

    def test_cache_key_is_deterministic(self):
        """image_mode_cache_key returns the exact same hash for repeated calls."""
        raw_hash = "deadbeef"
        kwargs = {"cropping": False, "base_size": 1024, "image_size": 1024}
        expected = hashlib.sha256(
            f"imgmode:base:1024:1024:{raw_hash}".encode()
        ).hexdigest()

        assert image_mode_cache_key(raw_hash, kwargs) == expected
        assert image_mode_cache_key(raw_hash, kwargs) == image_mode_cache_key(
            raw_hash, kwargs
        )


class TestPreflightAndDiffusionValidation:
    """Preflight and diffusion guards rejecting invalid images_config."""

    def test_diffusion_model_rejects_images_config(self):
        """_validate_diffusion_request raises InvalidRequestError if images_config is given."""
        engine = MagicMock(spec=VLMBatchedEngine)
        engine.is_diffusion_model = True
        engine._validate_diffusion_request = (
            VLMBatchedEngine._validate_diffusion_request.__get__(engine)
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            engine._validate_diffusion_request(
                kwargs={"images_config": {"image_mode": "base"}}
            )
        assert exc_info.value.field == "images_config"
        assert "not supported for diffusion models" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_preflight_chat_validates_images_config(self):
        """preflight_chat validates images_config before prefill memory check."""
        engine = MagicMock(spec=VLMBatchedEngine)
        engine._loaded = True
        engine.is_diffusion_model = False
        engine.model_type = "qwen2_vl"
        engine.preflight_chat = VLMBatchedEngine.preflight_chat.__get__(engine)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": TINY_PNG_B64},
                    },
                ],
            }
        ]

        # Non-Unlimited-OCR model should raise InvalidRequestError
        with pytest.raises(InvalidRequestError) as exc_info:
            await engine.preflight_chat(
                messages=messages,
                images_config={"image_mode": "base"},
            )
        assert exc_info.value.field == "images_config"


class TestHTTPStreamingAndEndpointValidation:
    """FastAPI TestClient HTTP tests for streaming validation and parameter forwarding."""

    @pytest.fixture
    def client(self):
        """TestClient with initialized server state and engine pool."""
        from omlx.server import _server_state
        orig_pool = _server_state.engine_pool
        mock_pool = MagicMock()
        mock_entry = MagicMock(
            config_model_type=None,
            preserve_thinking_default=None,
            model_context_length=32768,
        )
        mock_pool.get_entry.return_value = mock_entry
        _server_state.engine_pool = mock_pool
        try:
            yield TestClient(app, raise_server_exceptions=False)
        finally:
            _server_state.engine_pool = orig_pool

    def test_streaming_invalid_mode_returns_422_json_before_sse(self, client):
        """Invalid image_mode in streaming request returns 422 JSON before SSE headers."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Unlimited-OCR-oQ8",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "document parsing."},
                            {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                        ],
                    }
                ],
                "images_config": {"image_mode": "unsupported_mode"},
            },
        )
        assert resp.status_code == 422
        assert resp.headers["content-type"].startswith("application/json")
        assert "text/event-stream" not in resp.headers["content-type"]
        body = resp.json()
        assert "error" in body or "detail" in body

    def test_streaming_extra_fields_returns_422_json_before_sse(self, client):
        """Extra fields in images_config return 422 JSON before SSE headers."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Unlimited-OCR-oQ8",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "document parsing."},
                            {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                        ],
                    }
                ],
                "images_config": {"image_mode": "base", "extra_param": True},
            },
        )
        assert resp.status_code == 422
        assert resp.headers["content-type"].startswith("application/json")
        assert "text/event-stream" not in resp.headers["content-type"]

    def test_streaming_unsupported_model_returns_400_json_before_sse(self, client):
        """Setting images_config on non-Unlimited model returns 400 JSON before SSE headers."""
        mock_engine = MagicMock(spec=VLMBatchedEngine)
        mock_engine.model_type = "qwen2_vl"
        mock_engine.tokenizer = MagicMock()

        with patch("omlx.server.get_engine_for_model", new_callable=AsyncMock) as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2-vl-7b",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "transcribe"},
                                {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                            ],
                        }
                    ],
                    "images_config": {"image_mode": "base"},
                },
            )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert "text/event-stream" not in resp.headers["content-type"]
        data = resp.json()
        assert "images_config" in str(data)

    def test_streaming_no_images_returns_400_json_before_sse(self, client):
        """Supplying images_config without images returns 400 JSON before SSE headers."""
        mock_engine = MagicMock(spec=VLMBatchedEngine)
        mock_engine.model_type = "unlimited-ocr"
        mock_engine.tokenizer = MagicMock()

        with patch("omlx.server.get_engine_for_model", new_callable=AsyncMock) as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Unlimited-OCR-oQ8",
                    "stream": True,
                    "messages": [{"role": "user", "content": "text only"}],
                    "images_config": {"image_mode": "base"},
                },
            )
        assert resp.status_code == 400
        assert resp.headers["content-type"].startswith("application/json")
        assert "text/event-stream" not in resp.headers["content-type"]
        assert "requires image input" in str(resp.json())

    @pytest.mark.parametrize(
        "image_part",
        [
            {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
            {"type": "image_url", "image_url": {"url": TINY_PNG_B64, "detail": "high"}},
            {"type": "input_image", "image_url": {"url": TINY_PNG_B64}},
            {"type": "input_image", "image_url": {"url": TINY_PNG_B64, "detail": "auto"}},
        ],
    )
    def test_accepted_image_aliases_pass_validation(self, client, image_part):
        """Accepted image input aliases (image_url, input_image) pass early validation."""
        mock_engine = MagicMock(spec=VLMBatchedEngine)
        mock_engine.model_type = "unlimited-ocr"
        mock_engine.message_extractor = None
        mock_engine.tokenizer = MagicMock()
        mock_engine.tokenizer.encode.return_value = [1, 2, 3]
        mock_engine.tokenizer.apply_chat_template.return_value = "document parsing."
        mock_engine.count_chat_tokens.return_value = 10
        mock_engine.preflight_chat = AsyncMock()

        async def fake_stream_chat(*args, **kwargs):
            yield GenerationOutput(
                text="ok",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=2,
            )

        mock_engine.stream_chat = fake_stream_chat

        with patch("omlx.server.get_engine_for_model", new_callable=AsyncMock) as mock_get_engine:
            mock_get_engine.return_value = mock_engine
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Unlimited-OCR-oQ8",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "document parsing."}, image_part],
                        }
                    ],
                    "images_config": {"image_mode": "base"},
                },
            )
        # Should succeed past early validation and establish SSE stream
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_valid_base_mode_forwarding(self, client):
        """Valid base mode forwards chat_kwargs['images_config'] to engine in non-streaming and streaming."""
        mock_engine = MagicMock(spec=VLMBatchedEngine)
        mock_engine.model_type = "unlimited-ocr"
        mock_engine.message_extractor = None
        mock_engine.tokenizer = MagicMock(spec=["encode", "apply_chat_template", "decode"])
        mock_engine.tokenizer.encode.return_value = [1, 2, 3]
        mock_engine.tokenizer.apply_chat_template.return_value = "document parsing."
        mock_engine.count_chat_tokens.return_value = 10
        mock_engine.preflight_chat = AsyncMock()

        mock_output = MagicMock()
        mock_output.completion_tokens = 5
        mock_output.prompt_tokens = 10
        mock_output.cached_tokens = 0
        mock_output.text = "Parsed markdown"
        mock_output.finish_reason = "stop"
        mock_output.first_token_at = None
        mock_output.tool_calls = None
        mock_output.thinking = None
        mock_output.reasoning_content = None
        mock_engine.chat = AsyncMock(return_value=mock_output)

        async def fake_stream_chat(*args, **kwargs):
            yield GenerationOutput(
                text="Parsed markdown",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=5,
            )

        mock_engine.stream_chat = fake_stream_chat

        with patch("omlx.server.get_engine_for_model", new_callable=AsyncMock) as mock_get_engine:
            mock_get_engine.return_value = mock_engine

            # 1. Non-streaming forward
            resp_non_stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Unlimited-OCR-oQ8",
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "document parsing."},
                                {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                            ],
                        }
                    ],
                    "images_config": {"image_mode": "base"},
                },
            )
            assert resp_non_stream.status_code == 200
            mock_engine.chat.assert_called_once()
            _, non_stream_kwargs = mock_engine.chat.call_args
            assert non_stream_kwargs.get("images_config") == {"image_mode": "base"}

            # 2. Streaming forward
            resp_stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Unlimited-OCR-oQ8",
                    "stream": True,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "document parsing."},
                                {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                            ],
                        }
                    ],
                    "images_config": {"image_mode": "base"},
                },
            )
            assert resp_stream.status_code == 200
            # Preflight must have received images_config
            _, preflight_kwargs = mock_engine.preflight_chat.call_args
            assert preflight_kwargs.get("images_config") == {"image_mode": "base"}


class TestVLMEngineInputPreparationAndCacheIsolation:
    """Input preparation and behavioral cache isolation in VLMBatchedEngine."""

    @pytest.fixture
    def mock_vlm_engine(self):
        """Factory for a mocked VLMBatchedEngine with vision pipeline hooks."""
        engine = MagicMock(spec=VLMBatchedEngine)
        engine.model_type = "unlimited-ocr"
        engine._model_name = "Unlimited-OCR-oQ8"
        engine._enable_thinking = None
        engine._processor = MagicMock()
        engine._vlm_model = MagicMock()
        engine._vlm_model.config = MagicMock()
        engine._format_messages_for_vlm_template = MagicMock(
            return_value=([{"role": "user", "content": "<image>"}], [(0, 1)])
        )
        engine._image_token_count = MagicMock(return_value=100)
        engine._vision_cache = None
        engine._vision_cache_enabled = False
        engine._language_prompt_kwargs = MagicMock(return_value={})

        dummy_embeds = MagicMock()
        dummy_embeds.inputs_embeds = MagicMock()
        dummy_embeds.to_dict = MagicMock(return_value={})
        engine._vlm_model.get_input_embeddings = MagicMock(return_value=dummy_embeds)
        engine._prepare_vision_inputs = (
            VLMBatchedEngine._prepare_vision_inputs.__get__(engine)
        )
        return engine

    def test_process_chat_messages_pops_and_forwards_images_config(self):
        """_process_chat_messages pops images_config from kwargs and passes to _prepare_vision_inputs."""
        engine = MagicMock(spec=VLMBatchedEngine)
        engine._apply_ocr_prompt = MagicMock(side_effect=lambda msgs: msgs)
        engine._prepare_vision_inputs = MagicMock(
            return_value=([1, 2, 3], None, {}, "imghash", 0, [])
        )
        engine._process_chat_messages = (
            VLMBatchedEngine._process_chat_messages.__get__(engine)
        )

        messages = [{"role": "user", "content": "text only"}]
        kwargs = {"images_config": {"image_mode": "base"}, "temperature": 0.2}

        engine._process_chat_messages(messages, tools=None, kwargs=kwargs)

        assert "images_config" not in kwargs
        engine._prepare_vision_inputs.assert_called_once()
        _, call_kwargs = engine._prepare_vision_inputs.call_args
        assert call_kwargs["images_config"] == {"image_mode": "base"}

    def test_prepare_vision_inputs_passes_image_kwargs_to_prepare_inputs(self, mock_vlm_engine):
        """_prepare_vision_inputs forwards resolved kwargs and asserts partitioned key."""
        dummy_img = Image.new("RGB", (32, 32))
        raw_hash = compute_image_hash([dummy_img])
        messages = [{"role": "user", "content": "document parsing."}]

        mock_inputs = {
            "input_ids": MagicMock(ndim=2, tolist=lambda: [[101, 102]]),
            "pixel_values": MagicMock(),
        }

        with patch(
            "omlx.engine.vlm.apply_chat_template_with_reasoning_effort_fallback",
            return_value="document parsing.",
        ), patch("mlx_vlm.utils.prepare_inputs", return_value=mock_inputs) as mock_prep:
            (
                token_ids,
                vlm_embeds,
                vlm_kwargs,
                image_hash,
                image_cache_key_start,
                image_cache_key_ranges,
            ) = mock_vlm_engine._prepare_vision_inputs(
                messages=messages,
                images=[dummy_img],
                images_config={"image_mode": "base"},
            )

            # Assert prepare_inputs received base mode parameters
            assert mock_prep.called
            call_kwargs = mock_prep.call_args.kwargs
            assert call_kwargs.get("cropping") is False
            assert call_kwargs.get("base_size") == 1024
            assert call_kwargs.get("image_size") == 1024

            # Assert exact partitioned cache key behavior
            expected_base_key = image_mode_cache_key(
                raw_hash, {"cropping": False, "base_size": 1024, "image_size": 1024}
            )
            assert image_hash == expected_base_key
            assert image_hash != raw_hash

    def test_whole_request_cache_isolation_across_modes(self, mock_vlm_engine):
        """Behavioral test: base, gundam, and omitted modes produce partitioned whole-request keys."""
        dummy_img = Image.new("RGB", (32, 32), color="blue")
        raw_hash = compute_image_hash([dummy_img])
        messages = [{"role": "user", "content": "document parsing."}]

        mock_inputs = {
            "input_ids": MagicMock(ndim=2, tolist=lambda: [[101, 102]]),
            "pixel_values": MagicMock(),
        }

        with patch(
            "omlx.engine.vlm.apply_chat_template_with_reasoning_effort_fallback",
            return_value="document parsing.",
        ), patch("mlx_vlm.utils.prepare_inputs", return_value=mock_inputs):
            # 1. Base mode
            _, _, _, hash_base, _, _ = mock_vlm_engine._prepare_vision_inputs(
                messages=messages, images=[dummy_img], images_config={"image_mode": "base"}
            )
            # 2. Gundam mode
            _, _, _, hash_gundam, _, _ = mock_vlm_engine._prepare_vision_inputs(
                messages=messages, images=[dummy_img], images_config={"image_mode": "gundam"}
            )
            # 3. Unconfigured / omitted
            _, _, _, hash_default, _, _ = mock_vlm_engine._prepare_vision_inputs(
                messages=messages, images=[dummy_img], images_config=None
            )

        # Cross-mode behavioral assertions
        assert hash_base != hash_gundam
        assert hash_base != raw_hash
        assert hash_gundam != raw_hash
        # Omitted configuration inherits gundam defaults, sharing identity with explicit gundam
        assert hash_default == hash_gundam

        expected_base = hashlib.sha256(f"imgmode:base:1024:1024:{raw_hash}".encode()).hexdigest()
        expected_gundam = hashlib.sha256(f"imgmode:gundam:1024:640:{raw_hash}".encode()).hexdigest()
        assert hash_base == expected_base
        assert hash_gundam == expected_gundam

    def test_prefix_range_cache_isolation_across_modes(self, mock_vlm_engine):
        """Behavioral test: prefix cache ranges partition cumulative hashes across modes."""
        img1 = Image.new("RGB", (32, 32), color="red")
        img2 = Image.new("RGB", (32, 32), color="green")
        raw_prefix_hash = compute_image_hash([img1])
        raw_full_hash = compute_image_hash([img1, img2])

        messages = [
            {"role": "user", "content": "page 1"},
            {"role": "assistant", "content": "page 1 text"},
            {"role": "user", "content": "page 2"},
        ]
        # Multi-turn range setup
        mock_vlm_engine._format_messages_for_vlm_template = MagicMock(
            return_value=(messages, [(0, 1), (2, 1)])
        )

        mock_inputs = {
            "input_ids": MagicMock(ndim=2, tolist=lambda: [[101, 102]]),
            "pixel_values": MagicMock(),
        }

        with patch(
            "omlx.engine.vlm.apply_chat_template_with_reasoning_effort_fallback",
            return_value="rendered",
        ), patch("mlx_vlm.utils.prepare_inputs", return_value=mock_inputs):
            _, _, _, _, _, ranges_base = mock_vlm_engine._prepare_vision_inputs(
                messages=messages,
                images=[img1, img2],
                images_config={"image_mode": "base"},
            )
            _, _, _, _, _, ranges_gundam = mock_vlm_engine._prepare_vision_inputs(
                messages=messages,
                images=[img1, img2],
                images_config={"image_mode": "gundam"},
            )

        assert len(ranges_base) == 2
        assert len(ranges_gundam) == 2

        # Step 1 cumulative hash (after first image)
        cum_base_1 = ranges_base[0][1]
        cum_gundam_1 = ranges_gundam[0][1]
        assert cum_base_1 != cum_gundam_1
        assert cum_base_1 == image_mode_cache_key(
            raw_prefix_hash, {"cropping": False, "base_size": 1024, "image_size": 1024}
        )
        assert cum_gundam_1 == image_mode_cache_key(
            raw_prefix_hash, {"cropping": True, "base_size": 1024, "image_size": 640}
        )

        # Step 2 cumulative hash (after second image)
        cum_base_2 = ranges_base[1][1]
        cum_gundam_2 = ranges_gundam[1][1]
        assert cum_base_2 != cum_gundam_2
        assert cum_base_2 == image_mode_cache_key(
            raw_full_hash, {"cropping": False, "base_size": 1024, "image_size": 1024}
        )
        assert cum_gundam_2 == image_mode_cache_key(
            raw_full_hash, {"cropping": True, "base_size": 1024, "image_size": 640}
        )

    def test_vision_feature_cache_isolation_across_modes(self, mock_vlm_engine):
        """Behavioral test: per-image vision feature cache keys are partitioned across modes."""
        import mlx.core as mx

        mock_vlm_engine._vision_cache_enabled = True
        mock_cache = {}

        class DummyVisionCache:
            def get(self, key, model_name):
                return mock_cache.get(key)

            def put(self, key, model_name, features):
                mock_cache[key] = features

        mock_vlm_engine._vision_cache = DummyVisionCache()
        mock_vlm_engine._compute_vision_features = MagicMock(
            return_value=mx.zeros((1, 100, 64))
        )
        mock_vlm_engine._vision_features_match_image_tokens = MagicMock(return_value=True)
        mock_vlm_engine._split_vision_features = MagicMock(
            return_value=[mx.zeros((1, 100, 64))]
        )

        img = Image.new("RGB", (32, 32), color="yellow")
        raw_per_hashes = compute_per_image_hashes([img])
        assert len(raw_per_hashes) == 1
        raw_h = raw_per_hashes[0]

        mock_inputs = {
            "input_ids": MagicMock(ndim=2, tolist=lambda: [[101, 102]]),
            "pixel_values": MagicMock(),
        }

        with patch(
            "omlx.engine.vlm.apply_chat_template_with_reasoning_effort_fallback",
            return_value="rendered",
        ), patch("mlx_vlm.utils.prepare_inputs", return_value=mock_inputs):
            # 1. Base mode miss -> populate cache
            mock_vlm_engine._prepare_vision_inputs(
                messages=[{"role": "user", "content": "ocr"}],
                images=[img],
                images_config={"image_mode": "base"},
            )
            expected_base_key = image_mode_cache_key(
                raw_h, {"cropping": False, "base_size": 1024, "image_size": 1024}
            )
            assert expected_base_key in mock_cache

            # 2. Gundam mode: must be a cache miss because keys are partitioned!
            expected_gundam_key = image_mode_cache_key(
                raw_h, {"cropping": True, "base_size": 1024, "image_size": 640}
            )
            assert expected_gundam_key not in mock_cache

            # Populate gundam mode
            mock_vlm_engine._prepare_vision_inputs(
                messages=[{"role": "user", "content": "ocr"}],
                images=[img],
                images_config={"image_mode": "gundam"},
            )
            assert expected_gundam_key in mock_cache

            # Verify that both entries coexist without collision
            assert expected_base_key != expected_gundam_key
            assert len(mock_cache) == 2
