# Known Issues (deferred)

Issues we've root-caused or partially investigated but chosen not to fix yet.
When an issue here ages past "acceptable workaround", promote to a GitHub
issue and cite the relevant `git log` entries that updated this file.

A short "Resolved workarounds" section at the bottom lists bugs we patched
locally in oMLX that are still open upstream — useful when evaluating
whether a given oMLX patch can be dropped after a dependency rev.

---

## `mlx-community/Molmo2-8B-8bit` — image prompts degenerate to `!!!!`

- **First observed:** 2026-04-21 (VLM sweep on Molmo2-8B-8bit)
- **Status:** deferred; **not an oMLX bug** — broken checkpoint artifact on HF.
  No upstream code fix is pending; mitigation is model selection.
- **Severity:** any image request produces `!!!!` repeated to `max_tokens`.
  Text-only prompts work correctly.

### Symptom

- Text-only prompt → coherent response ("Hello! How can I help you today?").
- Prompt with any image (before OR after text) → output is `!!!!...` until
  `max_tokens`; `finish_reason: length`.

### Root cause (confirmed via 40-line pure mlx-vlm repro, no oMLX code in loop)

Molmo2 was trained in float32 and is numerically sensitive — Blaizzy's own
guidance (see upstream reference below) is that Molmo2 quants must be built
from a float16 base, not from the allenai/Molmo2-8B float32 weights.

The `mlx-community/Molmo2-8B-8bit` HF checkpoint was built without that
conversion — its `config.json` still declares `"dtype": "float32"` alongside
`{"bits": 8, "group_size": 64, "mode": "affine"}`. At inference time the
vision tower's dequantized activations overflow the fp16 range used for the
merged embeddings:

1. `get_input_embeddings()` in `mlx_vlm/models/molmo2/molmo2.py` computes
   `image_features = self.vision_tower(pixel_values, token_pooling)` — which
   contains `+inf` at a handful of entries (~2 per single-image request).
2. The additive merge `flat_x[positions] = flat_x[positions] + image_features`
   injects those `inf` values into the prefill embeddings at `<im_patch>`
   positions.
3. The language-model's first attention softmax sees `inf` — output becomes
   all-`nan` from that position onward.
4. `argmax` over a nan-filled logit vector in MLX returns index 0.
   Qwen2Tokenizer maps id 0 to `'!'` → the model emits `!` every step.
5. No stop condition trips because every logit is `nan`, so generation only
   halts at `max_tokens`.

Text-only prompts never run the vision tower, so nothing overflows.

### Not the cause

- Not a chat-template issue. `chat_template.jinja` is auto-loaded by
  transformers ≥ 4.45 into `tokenizer.chat_template`, reaches
  `Molmo2Processor.apply_chat_template`, and renders `<|image|>` correctly.
  The processor expands it to 392 `<im_patch>` tokens (id 151938) with a
  consistent `image_token_pooling (392,4)` / `image_grids (1,4)` /
  `image_num_crops [2]` set.
- Not oMLX's `_prepare_vision_inputs` forwarding — `image_token_pooling`,
  `image_grids`, `image_num_crops` are correctly surfaced via
  `extra_model_inputs` and reach `get_input_embeddings`.
- Not `_detect_mrope` — Molmo2's text_config has `rope_scaling: None`, so
  mRoPE is correctly False; decode routes through the standard path.
- Not the vocab_size / layers / prefix-cache classes of prior oMLX VLM bugs
  (all fixed earlier in this doc): model loads cleanly, first request fails
  identically on a cold cache.

### Upstream reference

- [Blaizzy/mlx-vlm#655](https://github.com/Blaizzy/mlx-vlm/issues/655)
  (CLOSED 2026-01-30) — "Molmo2 MLX Implementation Produces Broken Output
  (Repeats Prompts)". Same degenerate-output signature. Closed by the
  maintainer with the explanation above and a reference fp16 upload
  (`mlx-community/Molmo2-8B-5bit`). No code fix in mlx-vlm is pending; the
  resolution is to use fp16-based quants.
- [Blaizzy/mlx-vlm#639](https://github.com/Blaizzy/mlx-vlm/pull/639)
  (MERGED) — where the Molmo2 code path was introduced.

### Minimal repro (no oMLX)

```python
# .venv/bin/python
from PIL import Image
import numpy as np
from mlx_vlm.utils import load, prepare_inputs

model, processor = load("mlx-community/Molmo2-8B-8bit")
rendered = processor.apply_chat_template(
    [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "What is this?"}]}],
    add_generation_prompt=True, tokenize=False)
img = Image.new("RGB", (378, 378), (127, 0, 0))
inp = prepare_inputs(processor, images=[img], prompts=[rendered])
extra = {k: v for k, v in inp.items()
         if k not in ("input_ids", "attention_mask", "pixel_values") and v is not None}
out = model.get_input_embeddings(inp["input_ids"], inp["pixel_values"], **extra)
arr = np.asarray(out.inputs_embeds)  # asarray forces MLX materialization
print("inf count:", int(np.isinf(arr).sum()))   # > 0 → overflow confirmed
```

### Workaround options

*To be decided — all have trade-offs; none is strictly in scope for the
current sweep.*

- **Pin-away in `model_discovery.py`:** add `mlx-community/Molmo2-8B-8bit`
  to a blocklist so oMLX never advertises it. Clean for this repo+quant but
  doesn't help future bad uploads with the same pattern.
- **Substitute a known-good quant:** switch sweep/default to
  `mlx-community/Molmo2-8B-5bit` (or `-4bit` / `-bf16` built from fp16 —
  verify by checking `config.json` shows `dtype: float16`). Confirms
  Blaizzy's fp16 prerequisite. Requires download + re-benchmark.
- **Local oMLX patch:** cast `pixel_values`/`image_features` to float32 in
  a `molmo2`-specific override of `get_input_embeddings`, à la the
  LFM2.5-VL projector monkey-patch in `engine/vlm.py`. Defensible if other
  `dtype: float32` Molmo2 uploads show up, but buys complexity to cover an
  artifact bug.
- **File an HF community discussion** on `mlx-community/Molmo2-8B-8bit`
  requesting a rebuild from the fp16 weights (citing #655). Cheapest long-term
  fix; depends on the uploader responding.

---

## SmolVLM2 + schema-less `response_format: json_object` — 4000-token runaway, invalid JSON

- **First observed:** 2026-04-20
- **Status:** deferred; investigate further before filing upstream
- **Severity:** client-visible failure for JSON mode on SmolVLM2 (and likely other small/older VLMs). Server doesn't crash; returns 4000 tokens of prose and a `JSON validation failed` warning.

### Symptom

Periodic client requests (e.g. ~56s cadence — likely Open WebUI title generation)
send `response_format: {"type": "json_object"}`. Every completion:
- Hits `max_tokens` exactly (never stops naturally).
- Fails `extract_json_from_text`, logs `omlx.server - WARNING - JSON validation failed: Failed to extract valid JSON from output`.
- Does **not** produce any `GrammarMatcher rejected token` warnings, unlike LFM2.5-VL under identical scenarios.

### Suspected root cause (not yet confirmed with captured output)

Two-layer bug:

1. **Grammar terminates, generation continues unchecked.**
   `response_format: json_object` compiles to `compile_builtin_json_grammar()`
   (schema-less "any JSON"). The grammar terminates as soon as the model
   emits any complete JSON value (e.g. `{}`, `null`, a short object). After
   termination, `GrammarConstraintProcessor.__call__` short-circuits at
   `omlx/api/grammar.py:89` (`if self._terminated: return logits`). Nothing
   forces EOS or signals the scheduler to finish; the model freely generates
   prose until `max_tokens`.

2. **Greedy JSON extractor spans the prose.**
   `extract_json_from_text` at `omlx/api/tool_calling.py:1130-1140`
   falls through to Strategy 3's greedy regex `(\{[\s\S]*\})`, which
   matches from the first `{` to the **last** `}` anywhere in the 4000-token
   output — typically capturing narrative between braces. `json.loads`
   rejects the resulting string, extractor returns `None`, and the server
   logs the failure even though the model did produce valid JSON near the
   start.

### Not the cause

- Not the vocab_size bug fixed in `resolve_vocab_size` — SmolVLM2's
  `vocab_size` is 49280 at both the top level and `text_config`; the
  resolver returns it correctly.
- Not the layer-access bug fixed in `VLMModelAdapter.layers` — the model
  loads cleanly post-fix; the failure is at inference-time grammar/extraction.

### To confirm before fixing

Capture one failing completion's raw output text (curl reproduction or
temporary debug log). We expect to see a short valid JSON prefix followed
by narrative — confirming both layers of the hypothesis.

### Fix candidates (both worth applying)

1. `omlx/api/grammar.py` — when `matcher.is_terminated()` becomes True,
   force EOS (or signal the scheduler to finish the request) instead of
   silently letting the model ramble to `max_tokens`.
2. `omlx/api/tool_calling.py` `extract_json_from_text` — replace the
   greedy regex with a brace-balanced scan (or non-greedy + incremental
   parse) so a short valid JSON followed by prose extracts cleanly.

Either fix alone would likely clear the observed symptom; both together
is defense-in-depth.

---

## `Model type jvlm not supported` — decode-fallback warning, jina-vlm only

- **First observed:** 2026-04-20
- **Status:** deferred; unfixable at the oMLX layer alone.
- **Severity:** warning only; costs some decode performance on jina-vlm.

Previously affected smolvlm / glm4v / llava-qwen2 too. Fixed 2026-04-20 in
`omlx/engine/vlm.py` via `_vlm_aware_get_classes` resolver (redirects
`text_config.model_type` + an alias table) and mRoPE-skip guard. Remaining
affected model: **jina-vlm** (model_type `jvlm`), whose text backbone is
not published as an mlx-lm module and has no compatible alias.

Resolution requires either:
- Upstream `mlx_lm.models.jvlm` module, or
- A local shim model module and an alias `"jvlm": "<local-name>"` in
  `_VLM_TEXT_BACKBONE_ALIASES`.

Until then, jina-vlm logs `mlx-lm decode model failed, using vlm fallback:
Model type jvlm not supported.` at load and decodes via the VLM's own
`language_model` with `_IntOffsetCacheProxy` overhead.

---

# Resolved workarounds (local patches, upstream still open)

## LFM2.5-VL-* silent fallback to text-only LLM

- **First observed:** 2026-04-20
- **Status:** **fixed locally** via monkey-patch; upstream bugs still open.
- **Upstream:** `Blaizzy/mlx-vlm` **#1000, #1001, #1002** (all open,
  no maintainer response).

### Symptom

oMLX warns `VLM loading failed for LFM2.5-VL-*: Received 2 parameters not
in model: multi_modal_projector.layer_norm.{bias,weight}` and silently
falls back to loading the model as a text-only LLM.  Users lose vision
capability without an obvious signal.

### Root cause

LiquidAI's MLX export of LFM2.5-VL ships a contradictory checkpoint:
- `config.json`: `projector_use_layernorm: False`
- `model.safetensors`: `multi_modal_projector.layer_norm.{weight,bias}` present

mlx-vlm's `Lfm2VlMultiModalProjector.__init__` respects the flag and
creates `nn.Identity` for the slot (no params), so `model.load_weights`
with default `strict=True` rejects the checkpoint's two layer_norm
weights.

The weights are real: the predecessor `mlx-community/LFM2-VL-450M-bf16`
has no `projector_use_layernorm` key (mlx-vlm default is `True`) and
ships the same `layer_norm.*` weights — they load cleanly.  The 2.5
export just flipped the flag incorrectly.

### Local workaround

`omlx/engine/vlm.py` — `_patch_lfm2_vl_projector_layernorm()` replaces
`Lfm2VlMultiModalProjector.__init__` with a wrapper that, when
`config.projector_use_layernorm=False`, still instantiates a real
`nn.LayerNorm` so the checkpoint weights load.  Delegates to the
original `__init__` unchanged when the flag is `True`.

### When to remove this patch

Drop the patch once any of the following is true:
- `Blaizzy/mlx-vlm` fixes #1000/#1001/#1002 upstream (check release
  notes when bumping the `mlx-vlm` pin).
- LiquidAI republishes LFM2.5-VL checkpoints with consistent
  config/weights.
- A future lfm2-vl variant genuinely ships with no LayerNorm — then
  the current patch will raise a loud "Missing 2 parameters" error at
  load, signalling it's time to refine the override.

---

## Non-mRoPE VLM cache-hit AttributeError — `_rope_deltas`

- **First observed:** 2026-04-21 (dolphin-vision-72b-4bit, reported as
  "OMLX /v1/models healthy, but chat/completions drops mid-stream")
- **Status:** **fixed in-tree** (scheduler.py); candidate for upstream
  bug report.
- **Severity:** hard 500 on any non-mRoPE VLM once its prefix cache
  accumulates a hit (e.g. after a reload). Presents as a streaming
  connection dropping mid-response, because the error fires inside the
  engine loop after the 200-OK header has already been sent.

### Root cause

`scheduler.py:_do_external_prefill` unconditionally reads
`self.model._language_model._rope_deltas` when `start_offset > 0`
(prefix-cache hit). The attribute is only written by mRoPE-aware VLMs
(Qwen3-VL, GLM-4.6V, Gemma3 family, ...) during their
`get_input_embeddings()` path.

Non-mRoPE VLMs — **dolphin-vision / llava-qwen2, pixtral, llava-next,
llava-bunny, paligemma, SmolVLM2, jina-vlm, ...** — never set the
attribute, so any cache-hit request raises
`AttributeError: 'LanguageModel' object has no attribute '_rope_deltas'`.

The pre-existing `hasattr(self.model, "_language_model")` guard was too
broad: it checks whether the adapter is a VLM, not whether the VLM's
language_model actually has `_rope_deltas`.

### Why it only surfaced now

The failure requires `start_offset > 0`, i.e. a prefix-cache hit. The
SSD prefix cache at `~/.omlx/cache/` persists across restarts, so a
cold dolphin session never triggers it — the bug only manifests on the
**second session after the cache has been populated**. The user's
earlier "53/53/53" clean sweep happened before the cache had any
reusable dolphin prefixes. The reload required to deploy the
decode-model fix re-exposed the latent bug.

### Local fix

`omlx/scheduler.py:_do_external_prefill` — the save/zero block now gates
on `getattr(self.model, "_uses_mrope", False)` (adapter-level boolean
set by `VLMModelAdapter._detect_mrope` at init) AND uses
`getattr(self.model._language_model, "_rope_deltas", None)` to tolerate
mRoPE models before their first `get_input_embeddings()` call.

### Upstream candidacy

This fix is a clean candidate for filing against `jundot/omlx` as a
bug fix (no existing issue; see grep of recent issues). The problematic
lines were introduced by commit `512c21b` (2026-04-11, "per-request
mRoPE position tracking for batched VLM decode") — they were written
assuming only mRoPE VLMs would reach this branch.

Bundle candidates for a single upstream PR (all open at the oMLX layer,
all novel fixes, all smoke-tested):
- This scheduler guard.
- `resolve_vocab_size` → prefer lm_head / text_config over top-level.
- `_prepare_vision_inputs` → forward `image_token_index` to `prepare_inputs`.
- `VLMModelAdapter.layers` → flat+nested pattern handling.
- LFM2.5-VL projector LayerNorm workaround (above).
- Decode-model resolver + mRoPE-skip guard.

---

## dolphin-vision-72b Metal OOM from decode-model alias

- **First observed:** 2026-04-21 (crash loop; every dolphin load
  OOMed Metal within ~7 s of `GrammarCompiler initialized`, before any
  request arrived)
- **Status:** **fixed** by reverting the alias; separate follow-up
  needed to restore the optimization without the memory cost.
- **Severity:** hard SIGABRT crash loop; server auto-restarts but never
  serves a dolphin request. Presents to clients as HTTP 200 + mid-stream
  TCP drop with a ~14 s wallclock (= launchd throttle + reload time).

### Root cause

The alias `"llava-qwen2": "qwen2"` in `_VLM_TEXT_BACKBONE_ALIASES`
caused dolphin-vision-72b-4bit (38 GB on disk) to take the successful
`_build_decode_model` path.  That path calls
`mlx_lm.utils.load_model(lazy=True, strict=False)`, whose interior
unconditionally does:

```python
weights = {}
for wf in weight_files:
    weights.update(mx.load(wf))
...
model.load_weights(list(weights.items()), strict=False)
```

For dolphin this materialises **~38 GB** of weight arrays in unified
memory before the subsequent oMLX overlay
(`lm_model.load_weights(lm_params, strict=False)`) replaces them by
reference with the VLM's existing tensors.  Until GC + MLX allocator
reclaim the orphaned copies, the process holds ~2× dolphin's weights
simultaneously — roughly 76 GB just in model tensors, on top of the
vision encoder, memory-enforcer scratch, and Metal command-buffer
overhead.  On the 128 GB Mac Studio (120 GB process memory ceiling,
similar Metal wired budget), the next Metal command buffer submitted
after load — emitted by the engine's post-load warmup path, before any
request is handled — failed with
`[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`, a `libc++abi`
hard crash (SIGABRT) that no Python `except` can catch.

### A/B evidence

| Session | `llava-qwen2 → qwen2` alias | `_build_decode_model` result | Dolphin sweeps |
|---|---|---|---|
| 21:22 on 04-20 | dormant (resolver had kwarg bug) | fallback; `self._lm_model = None` | **53/53/53 clean** |
| 23:49 on 04-20 | active | `VLM decode model ready` | AttributeError then OOM |
| 01:00–01:26 on 04-21 | active | `VLM decode model ready` | 5 successive SIGABRT OOMs |

### Local fix

Removed the `llava-qwen2: qwen2` entry from `_VLM_TEXT_BACKBONE_ALIASES`
in `omlx/engine/vlm.py`.  `_vlm_aware_get_classes` now falls through to
`_get_classes(cfg)`, which raises `ValueError("Model type llava-qwen2
not supported")`, caught by the pre-existing
`except Exception as e: logger.warning(...)` in `_build_decode_model`.
Result: `self._lm_model = None`, `VLMModelAdapter` uses the VLM's own
`language_model` for decode — the same working path as the 53/53/53
sweep.  Small VLMs (SmolVLM2, LFM2.5-VL) keep their fix via the
`text_config.model_type` flattening path, which does not depend on the
removed alias.

### Follow-up (not done)

To restore the 2× decode speedup for dolphin without the memory cost,
`_build_decode_model` would need to construct the text backbone directly
(read config → resolve classes → instantiate → quantize from config)
without the intermediate `mx.load` step, then overlay VLM weights onto
the zero-weight model.  Estimated ~20 LoC; requires reimplementing the
relevant portion of `mlx_lm.utils.load_model` in oMLX (or upstreaming a
`skip_weight_load=True` kwarg to mlx-lm).  Defer until after the current
sweep stabilises.

### Debugging note

Classic "two bugs stacked" incident: today I fixed the
`_rope_deltas` AttributeError ([see above](#non-mrope-vlm-cache-hit-attributeerror--_rope_deltas))
believing it was the dolphin regression, only to find the user's next
probe still failed with the *same client-visible symptom* (HTTP 200 +
mid-stream drop).  The `_rope_deltas` bug was inside `if vlm_embeds is
not None:` — it only fires on VLM image requests, never on text-only
requests to a VLM, which is what the sweep was using.  Two independent
bugs, same outward symptom — the lesson is to verify a fix against the
specific failure path the client is exercising, not infer coverage
from "the server works now."
