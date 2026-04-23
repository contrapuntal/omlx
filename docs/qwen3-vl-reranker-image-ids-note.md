# Qwen3-VL Reranker `image_ids` Note

Date: 2026-04-23

## Summary

Converted `Qwen/Qwen3-VL-Reranker-8B` to MLX 8-bit and verified that `oMLX`
discovers it as:

- model: `Qwen3-VL-Reranker-8B-8bit-mlx`
- type: `reranker`
- engine: `reranker`

The model loaded successfully, but the first live `/v1/rerank` request failed
with:

```text
'Qwen3VLProcessor' object has no attribute 'image_ids'
```

This was not a conversion problem. It was a runtime compatibility bug in the
VL reranker path.

## Upstream Search

Checked upstream `jundot/omlx` issues and PRs for:

- `image_ids`
- `Qwen3VLProcessor`
- `reranker image_ids`
- `Qwen3-VL-Reranker`
- `apply_chat_template reranker`
- `ProcessorMixin reranker`

Results:

- No exact upstream issue or PR found for the `image_ids` reranker failure.
- Related issue exists:
  - `#877` "Qwen3-VL-Reranker-2B is rejected on oMLX 0.3.6 with \"Unsupported reranker architecture: Qwen3VLForConditionalGeneration\""
- In issue `#877`, upstream said Qwen3-VL reranker support landed on `main`
  and `/v1/rerank` image inputs were supported.

Conclusion: this `image_ids` error appears to be a follow-up bug not already
tracked upstream.

## Root Cause

`omlx/models/embedding.py` already contains a compatibility repair for
synthetic multimodal processors loaded through `mlx-embeddings`.

That repair restores:

- `image_ids`
- `video_ids`
- `audio_ids`

The VL reranker path in `omlx/models/reranker.py` loaded the same style of
processor but did not apply the same repair. When the reranker request path
reached chat-template / multimodal token preparation, Transformers expected
`Qwen3VLProcessor.image_ids` to exist and raised.

## Local Fix

Applied the same processor repair in the VL reranker loader:

- file: `omlx/models/reranker.py`
- behavior: after `mlx_embeddings.load(...)`, restore missing
  `image_ids/video_ids/audio_ids` on the outer processor and inner
  `processor.processor` when present

Added regression coverage:

- file: `tests/test_reranker_vl.py`
- test: `test_load_vl_reranker_repairs_missing_multimodal_token_ids`

## Verification

Local test:

```text
uv run pytest tests/test_reranker_vl.py -q
16 passed
```

Live server verification on `:8100` after restart:

```json
{
  "id": "rerank-a7548380",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.3523804247379303,
      "document": {
        "text": "Paris is the capital of France."
      }
    },
    {
      "index": 1,
      "relevance_score": 0.0017200156580656767,
      "document": {
        "text": "Berlin is the capital of Germany."
      }
    }
  ],
  "model": "Qwen3-VL-Reranker-8B-8bit-mlx",
  "usage": {
    "total_tokens": 0
  }
}
```

Relevant server log after fix:

```text
Reranker model loaded successfully: /Volumes/MacExternalStorage/models/rerank/Qwen3-VL-Reranker-8B-8bit-mlx
Rerank: 3 docs, 0 tokens in 0.588s
```

## Follow-up

If this should be reported upstream, the issue should point out:

1. Qwen3-VL reranker discovery and load succeed.
2. `/v1/rerank` fails at runtime on missing `Qwen3VLProcessor.image_ids`.
3. `embedding.py` already contains the correct compatibility pattern.
4. Applying the same repair in `reranker.py` fixes the problem.
