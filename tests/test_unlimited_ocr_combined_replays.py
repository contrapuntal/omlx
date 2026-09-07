# SPDX-License-Identifier: Apache-2.0
"""
HTTP replay integration tests verifying cold/warm and cross-mode cache isolation
in the combined runtime (PR #3448 148b959a + Unlimited-OCR image-mode).

Verifies via HTTP API (/v1/chat/completions):
1. Cold Request: First request in gundam mode initializes cache.
2. Warm Replay: Second request in gundam mode reuses cached prefix.
3. Cross-Mode Replay (Switch to Base): Third request in base mode suffers cache miss,
   prefills cleanly under base parameters (cropping=False, image_size=1024), and populates base cache.
4. Warm Replay (Base): Fourth request in base mode hits the base cache.
5. Cross-Mode Replay (Back to Gundam): Fifth request hits the original gundam cache.
6. Unconfigured Request: Replays with omitted images_config hit the gundam cache,
   confirming default inheritance while maintaining isolation from base.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.api.openai_models import ChatCompletionRequest
from omlx.engine.base import GenerationOutput
from omlx.engine.vlm import VLMBatchedEngine
from omlx.server import app
from omlx.utils.image import compute_image_hash
from omlx.utils.ocr_inputs import image_mode_cache_key, resolve_ocr_image_kwargs

TINY_PNG_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class StatefulReplayEngine(VLMBatchedEngine):
    """Test engine tracking cache lookups and admissions across HTTP requests."""

    def __init__(self):
        super().__init__("Unlimited-OCR-oQ8")
        self._model_name = "Unlimited-OCR-oQ8"
        self._enable_thinking = None
        self._loaded = True
        self._mock_tokenizer = MagicMock(spec=["encode", "apply_chat_template", "decode"])
        self._mock_tokenizer.encode.return_value = [1, 2, 3]
        self._mock_tokenizer.apply_chat_template.return_value = "document parsing."

        # Cache state tracking: key -> cached_tokens
        self.prefix_cache_store = {}
        self.recorded_replays = []

    @property
    def tokenizer(self):
        return self._mock_tokenizer

    @property
    def message_extractor(self):
        return None

    @property
    def model_type(self):
        return "unlimited-ocr"

    def count_chat_tokens(self, *args, **kwargs):
        return 10

    async def preflight_chat(self, messages, **kwargs):
        # Forward validation
        from omlx.utils.image import extract_images_from_messages
        _, images, _ = extract_images_from_messages(messages)
        resolve_ocr_image_kwargs(self.model_type, kwargs.get("images_config"), len(images))

    async def chat(self, messages, **kwargs):
        images_config = kwargs.get("images_config")
        # Compute image mode cache identity
        from omlx.utils.image import extract_images_from_messages
        _, images, _ = extract_images_from_messages(messages)
        raw_hash = compute_image_hash(images) if images else "nohash"
        resolved_kwargs = resolve_ocr_image_kwargs(self.model_type, images_config, len(images))
        mode_key = image_mode_cache_key(raw_hash, resolved_kwargs)

        is_warm = mode_key in self.prefix_cache_store
        if is_warm:
            cached_tokens = self.prefix_cache_store[mode_key]
        else:
            cached_tokens = 0
            # Store in cache after cold run
            self.prefix_cache_store[mode_key] = 100

        self.recorded_replays.append({
            "mode_key": mode_key,
            "images_config": images_config,
            "is_warm": is_warm,
            "cached_tokens": cached_tokens,
            "resolved_kwargs": resolved_kwargs,
        })

        return GenerationOutput(
            text=f"Transcription (mode={resolved_kwargs.get('image_size')})",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=cached_tokens,
            finish_reason="stop",
        )

    async def stream_chat(self, messages, **kwargs):
        images_config = kwargs.get("images_config")
        from omlx.utils.image import extract_images_from_messages
        _, images, _ = extract_images_from_messages(messages)
        raw_hash = compute_image_hash(images) if images else "nohash"
        resolved_kwargs = resolve_ocr_image_kwargs(self.model_type, images_config, len(images))
        mode_key = image_mode_cache_key(raw_hash, resolved_kwargs)

        is_warm = mode_key in self.prefix_cache_store
        cached_tokens = self.prefix_cache_store[mode_key] if is_warm else 0
        if not is_warm:
            self.prefix_cache_store[mode_key] = 100

        self.recorded_replays.append({
            "mode_key": mode_key,
            "images_config": images_config,
            "is_warm": is_warm,
            "cached_tokens": cached_tokens,
            "resolved_kwargs": resolved_kwargs,
        })

        yield GenerationOutput(
            text="Streaming transcription",
            prompt_tokens=100,
            completion_tokens=20,
            cached_tokens=cached_tokens,
            finish_reason="stop",
        )


class TestMockEngineHTTPContractReplays:
    """Mock-engine HTTP contract test: verifies HTTP endpoint contract and parameter forwarding across replay sequence.

    Note: This is an HTTP contract test using a mock engine. It verifies route behavior,
    request parsing, parameter forwarding, and usage mapping, but does not exercise
    real model execution, KV allocation, scheduler ring reconstruction, or SSD cache files.
    """

    @pytest.fixture
    def client_and_engine(self):
        from omlx.server import _server_state
        orig_pool = _server_state.engine_pool
        engine = StatefulReplayEngine()

        mock_pool = MagicMock()
        mock_entry = MagicMock(
            config_model_type="unlimited-ocr",
            preserve_thinking_default=None,
            model_context_length=32768,
        )
        mock_pool.get_entry.return_value = mock_entry
        mock_pool.preload_pinned_models = AsyncMock()
        mock_pool.shutdown = AsyncMock()
        _server_state.engine_pool = mock_pool

        with patch("omlx.server.get_engine_for_model", new_callable=AsyncMock) as mock_get_engine:
            mock_get_engine.return_value = engine
            try:
                yield TestClient(app, raise_server_exceptions=False), engine
            finally:
                _server_state.engine_pool = orig_pool

    def test_cold_warm_and_cross_mode_http_replays(self, client_and_engine):
        """Execute 6-phase HTTP replay: Cold -> Warm -> Cross-Mode -> Warm -> Return -> Unconfigured."""
        client, engine = client_and_engine

        def make_request(images_config, stream=False):
            payload = {
                "model": "Unlimited-OCR-oQ8",
                "stream": stream,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "document parsing."},
                            {"type": "image_url", "image_url": {"url": TINY_PNG_B64}},
                        ],
                    }
                ],
            }
            if images_config is not None:
                payload["images_config"] = images_config
            resp = client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200
            return resp

        # =========================================================================
        # Phase 1: Cold Request in Gundam Mode
        # =========================================================================
        resp1 = make_request({"image_mode": "gundam"})
        data1 = resp1.json()
        assert data1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert len(engine.recorded_replays) == 1
        rep1 = engine.recorded_replays[0]
        assert not rep1["is_warm"]
        assert rep1["cached_tokens"] == 0
        assert rep1["resolved_kwargs"]["cropping"] is True
        assert rep1["resolved_kwargs"]["image_size"] == 640
        gundam_key = rep1["mode_key"]

        # =========================================================================
        # Phase 2: Warm Replay in Gundam Mode (Exact Match)
        # =========================================================================
        resp2 = make_request({"image_mode": "gundam"})
        data2 = resp2.json()
        assert data2["usage"]["prompt_tokens_details"]["cached_tokens"] == 100
        assert len(engine.recorded_replays) == 2
        rep2 = engine.recorded_replays[1]
        assert rep2["is_warm"]
        assert rep2["cached_tokens"] == 100
        assert rep2["mode_key"] == gundam_key

        # =========================================================================
        # Phase 3: Cross-Mode Switch to Base Mode (Identical Image Input)
        # MUST MISS the Gundam cache and run cold under Base parameters!
        # =========================================================================
        resp3 = make_request({"image_mode": "base"})
        data3 = resp3.json()
        assert data3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert len(engine.recorded_replays) == 3
        rep3 = engine.recorded_replays[2]
        assert not rep3["is_warm"]
        assert rep3["cached_tokens"] == 0
        assert rep3["resolved_kwargs"]["cropping"] is False
        assert rep3["resolved_kwargs"]["image_size"] == 1024
        base_key = rep3["mode_key"]
        assert base_key != gundam_key

        # =========================================================================
        # Phase 4: Warm Replay in Base Mode
        # =========================================================================
        resp4 = make_request({"image_mode": "base"})
        data4 = resp4.json()
        assert data4["usage"]["prompt_tokens_details"]["cached_tokens"] == 100
        assert len(engine.recorded_replays) == 4
        rep4 = engine.recorded_replays[3]
        assert rep4["is_warm"]
        assert rep4["cached_tokens"] == 100
        assert rep4["mode_key"] == base_key

        # =========================================================================
        # Phase 5: Return to Gundam Mode
        # MUST HIT the original Gundam cache entry without interference from Base!
        # =========================================================================
        resp5 = make_request({"image_mode": "gundam"})
        data5 = resp5.json()
        assert data5["usage"]["prompt_tokens_details"]["cached_tokens"] == 100
        assert len(engine.recorded_replays) == 5
        rep5 = engine.recorded_replays[4]
        assert rep5["is_warm"]
        assert rep5["cached_tokens"] == 100
        assert rep5["mode_key"] == gundam_key

        # =========================================================================
        # Phase 6: Unconfigured Request (images_config omitted)
        # MUST hit Gundam cache identity, confirming default resolution behavior!
        # =========================================================================
        resp6 = make_request(None)
        data6 = resp6.json()
        assert data6["usage"]["prompt_tokens_details"]["cached_tokens"] == 100
        assert len(engine.recorded_replays) == 6
        rep6 = engine.recorded_replays[5]
        assert rep6["is_warm"]
        assert rep6["mode_key"] == gundam_key
        assert rep6["resolved_kwargs"]["image_size"] == 640

        # Verify that both cache entries remain cleanly partitioned in store
        assert len(engine.prefix_cache_store) == 2
        assert gundam_key in engine.prefix_cache_store
        assert base_key in engine.prefix_cache_store
