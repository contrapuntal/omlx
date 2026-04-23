# Handoff: Gemma-4 family OOMs in oMLX engine, not in mlx-vlm

## TL;DR

`gemma-4-26b-a4b-it-MLX-8bit` crashes oMLX's server with a Metal GPU OOM
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`) on a trivial `"hi"` request.
`gemma-4-31b-it-MLX-8bit` (dense, not MoE) also crashes with the same error
at `ctx_size=8192`. Qwen3.6-35B-A3B-MLX-8bit — a larger MoE VLM — works at
full 256K ctx on the same hardware and same process. The bug is **inside
oMLX's engine path**, not in mlx-vlm or the model checkpoints. Your job:
find where, with a reproducer that doesn't depend on the live server.

## Environment

- Machine: Mac Studio, 128 GB unified memory, Metal wired limit
  `iogpu.wired_limit_mb = 114688` (112 GB)
- Project: `/Volumes/MacExternalStorage/proj/omlx` (branch `dev`)
- Venv: `/Volumes/MacExternalStorage/proj/omlx/.venv` (managed by uv)
- Key pin versions (just bumped, NOT the cause — bug predates the bump):
  - mlx-vlm: `1bf77424` (v0.4.4, 2026-04-22)
  - mlx-lm: `ed1fca4` (v0.31.3)
  - transformers: `5.5.4`
- Model path: `/Volumes/MacExternalStorage/LMStudio/models/unsloth/gemma-4-26b-a4b-it-MLX-8bit`
- Gemma-4 26B A4B config essentials: MoE (128 experts, top_k=8),
  `vocab_size=262144`, `hidden_size=2816`, `num_hidden_layers=30`,
  all `layer_types = "sliding_attention"` (window=1024), `final_logit_softcapping=30.0`,
  `max_position_embeddings=262144`, `num_kv_shared_layers=0`,
  `hidden_size_per_layer_input=0`.
- Gemma-4 31B: dense (not MoE), also Gemma-4 family, fails at ctx_size 8192.

## Crash signature

From `~/Library/Logs/omlx/server.err.log` — identical both times loaded:

```
Loaded model: gemma-4-26b-a4b-it-MLX-8bit (estimated: 27.34GB, total: 27.34GB)
Structured output requires xgrammar. Install with: pip install 'omlx[grammar]'
libc++abi: terminating due to uncaught exception of type std::runtime_error:
  [METAL] Command buffer execution failed: Insufficient Memory
  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)
```

The crash is an **unhandled C++ abort** from MLX — SIGABRT, no Python traceback.
launchd restarts the process. Model loads fine (27.34 GB); crash fires ~1 ms
after the "Structured output requires xgrammar" per-request log line. The
`"Structured output requires xgrammar"` line originates from
`omlx/engine/vlm.py:422-432` (`grammar_compiler` property, accessed per-request).

## What's already ruled out (with evidence)

Two repro scripts exist in `scripts/` — both run successfully on the exact
same model, venv, and hardware:

| Script | Setup | Result | Rules out |
|---|---|---|---|
| `scripts/debug_gemma4_oom.py` | Bare `mlx_vlm.utils.load` + `model.language_model(cache=None)` on `"hi"` | peak 26.04 GB, full logits produced | mlx-vlm code, checkpoint, transformers 5.5.4, hardware |
| `scripts/debug_gemma4_oom_v2.py` | Same, + `mlx_lm.models.cache.make_prompt_cache(model.language_model)` (30× `RotatingKVCache(max_size=1024)`) | peak 26.04 GB | Standard sliding-window KV cache |

Both print staged memory via `mx.synchronize(); mx.get_peak_memory()`.
Run them with `uv run python scripts/debug_gemma4_oom.py` from the repo root.

Additional exclusions from log/code inspection:

- Not vision/audio path — `Gemma4Model.get_input_embeddings` short-circuits
  when `pixel_values=None` and `audio_features=None`, confirmed by the trace
  in `mlx_vlm/models/gemma4/gemma4.py:70-163`.
- Not `specprefill` — `omlx/patches/specprefill.py:sparse_prefill` only runs
  on long prompts with explicit sparse indices; a 14-token `"hi"` doesn't
  trigger it.
- Not the mlx-lm decode model — built at load time with "VLM decode model
  ready (weight sharing, zero-copy)" (`omlx/engine/vlm.py:545-580`), but
  crash is during first prefill through the VLM model, before decode runs.
- Not today's mlx-vlm bump — user explicitly confirmed the bug predates
  pinning to `1bf77424`.
- Not Metal wired-memory limit of weights — 27 GB under 112 GB ceiling.

## Strongest current hypothesis

The bug is in **oMLX's engine path between request receipt and model forward**,
specifically in how it allocates the per-layer KV cache backing for Gemma-4's
sliding-attention layers. Evidence:

- Gemma-4-31B OOMs at `ctx_size=8192` but not below → allocation scales with
  a token/ctx-size parameter, pointing at KV cache (not forward transients).
- oMLX uses `PagedCacheManager` (vLLM-style paged KV) instead of mlx-lm's
  `RotatingKVCache`. Log: `block_size=1024, initial_blocks=256,
  max_blocks=100000, max_tokens=102400000`. The Python `CacheBlock`
  bookkeeping in `omlx/cache/paged_cache.py` is lightweight, but the actual
  GPU tensor backing (per layer, per block) has not been located yet —
  this is the next thing to read.
- The log line `Aligning paged cache block_size=256 to 1024 (RotatingKVCache
  window_size=1024, multiplier=1x)` comes from `omlx/scheduler.py`; grep for
  `Aligning paged cache` to find the alignment code. This only triggers for
  models with sliding-window attention (Gemma-4, maybe a few others). Qwen3.6
  probably uses full attention and takes a different path.
- Per-layer 1024-token block: `2 (K,V) × 8 kv_heads × 256 head_dim × 1024
  tokens × 2 bytes (bf16) = 8 MB`. 30 layers × 8 MB = 240 MB per block-slot.
  If `initial_blocks=256` is per-layer, that's 60 GB allocated eagerly.
  If it's global, far less. Need to verify.

## Your task: option B — drive `VLMBatchedEngine` directly

Write `scripts/debug_gemma4_oom_v3.py` that imports and drives oMLX's
`VLMBatchedEngine` programmatically without the HTTP server. Goals:

1. **Reproduce the OOM outside the live server.** If v3 OOMs, you've
   localized the bug to the engine and have a fast iteration loop for
   bisection. If v3 doesn't OOM, the bug is in a higher layer
   (scheduler/server) and v3 still serves as a control.
2. Stage memory checkpoints (same `mx.synchronize(); mx.get_peak_memory()`
   pattern as v1/v2) around: engine construction, first request prep,
   cache allocation, forward pass. Identify which stage allocates the
   failing tensor.

### Starting points in the codebase

- `omlx/engine/vlm.py` — `VLMBatchedEngine` class. Its `__init__` (around
  line 440–600) sets up the adapter, builds the mlx-lm decode model, wires
  up scheduler config. The `__call__` / generate paths are further down.
- `omlx/engine_pool.py` — how the server instantiates engines. Probably has
  a `load_model(model_name)` helper you can call directly.
- `omlx/scheduler.py` — `SchedulerConfig` construction and the
  paged-cache alignment logic (`Aligning paged cache block_size=...` log).
- `omlx/cache/paged_cache.py:498-568` — `PagedCacheManager.__init__`.
  Note: `initial_blocks=256` default. `CacheBlock` is Python-side state.
  GPU tensor backing is elsewhere — likely in a `PagedKVCache` or similar
  class that wraps actual `mx.array` buffers. **Find that class first.**
  Search: `grep -rn "mx.zeros\|mx.array" omlx/cache/` — look for shapes
  involving `num_layers`, `block_size`, `num_heads`, `head_dim`.
- `omlx/cache/hybrid_cache.py` — interacts with sliding-window layers.
  `HybridCache.has_rotating_layers` / `get_max_window_size` suggest it's
  where Gemma-4's cache shape gets computed.

### Specific things to instrument

- Log the shape and bytes of every `mx.zeros` / `mx.array` allocation in
  `omlx/cache/*.py` during a single-request run.
- Log `num_layers`, `window_size`, `block_size`, `initial_blocks`,
  `num_kv_heads`, `head_dim` as seen by the paged cache for this model.
- Count total bytes allocated per layer × num_layers before the forward
  runs. Compare to the 26 GB weight footprint and the 112 GB ceiling.

### Control comparison

Run the same instrumented path on `Qwen3.6-35B-A3B-MLX-8bit` (known-good,
256K ctx works). The delta in allocation sizes between Qwen3.6 and
Gemma-4-26B will point at the Gemma-specific path or per-layer factor.

### What NOT to do

- **Do not "fix" anything yet.** Per systematic-debugging Phase 1: no
  fixes without a confirmed root cause. This handoff has eliminated
  suspects but hasn't named the failing allocation. Name it first.
- Do not revert the mlx-vlm pin. The user confirmed the bug predates it.
- Do not assume all Gemma-4 variants fail identically — cross-check the
  26B A4B (MoE) result against the 31B (dense) result. The common factor
  is Gemma-4, not MoE.

## Related context files

- `docs/known-issues.md` — existing pattern for documenting root-caused but
  deferred bugs. The Molmo2 entry is a good template for the final writeup
  once root cause is confirmed. Note: Molmo2's 40-line repro *reproduced*
  the bug (proving "not oMLX"); yours *doesn't* reproduce (proving
  "yes oMLX") — the symmetric conclusion.
- `omlx/engine/vlm.py:519-520` — explicit comment: *"mlx-vlm language
  models may produce degenerated output in batched decode (e.g. gemma4
  missing KV sharing between layers)."* Suggests the team has seen
  gemma4-specific engine issues before; this may be another.

## Success criteria for your handoff back

1. A v3 repro script in `scripts/` that either reproduces the OOM or
   doesn't — either outcome is useful.
2. The specific tensor allocation (file:line, shape, bytes) responsible
   for the memory blowup.
3. Confirmation of why Gemma-4 triggers it and Qwen3.6 doesn't
   (specific code branch or config-derived number).
4. A one-paragraph proposed fix with tradeoffs — NOT implemented yet,
   for user review.
