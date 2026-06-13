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
- **Status:** **no longer reproducing as of 2026-05-16.** Checkpoint on HF
  appears to have been re-quantized from a float16 base since this entry was
  filed — safetensor weights are now float16 (norm) + uint32 (packed
  quantized linear), even though `config.json` still declares
  `"dtype": "float32"` (the field is stale). A re-run produces coherent
  image responses ("The image is a uniform dark blue color…"). The original
  failure mode requires the float32-base build that mlx-community no longer
  serves. Entry kept for historical context and as a reminder that the
  `dtype` config field can lag behind actual weight precision.
- **Severity (historical):** any image request produced `!!!!` repeated to
  `max_tokens`. Text-only prompts worked correctly.

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
  dev-legacy LFM2.5-VL projector monkey-patch (no longer on `fork-main`;
  see the LFM2.5-VL entry below). Defensible if other
  `dtype: float32` Molmo2 uploads show up, but buys complexity to cover an
  artifact bug.
- **File an HF community discussion** on `mlx-community/Molmo2-8B-8bit`
  requesting a rebuild from the fp16 weights (citing #655). Cheapest long-term
  fix; depends on the uploader responding.

---

## `falcon_ocr` / `falcon_perception` — quantized models will produce uint32 logits

- **First observed:** 2026-04-30 (latent-vulnerability scan during mlx-vlm
  issue #1091 review)
- **Status:** deferred; **not an oMLX bug** — upstream mlx-vlm code path,
  no published quantized variant of either model exists yet. File upstream
  PR if/when someone uploads a quantized falcon_ocr or falcon_perception.
- **Severity:** any sampling pass on a quantized falcon_ocr / falcon_perception
  language model will fail or silently produce garbage — `argmax` on uint32
  logits returns valid token IDs but they're noise.

### Symptom (predicted, no live repro)

`falcon_ocr/language.py:434` and `falcon_perception/language.py:508` both
do:

```python
logits = out.astype(self.model.embed_tokens.weight.dtype)
```

When `embed_tokens` is quantized, `weight.dtype` is `mx.uint32` (the packed
storage container). The cast turns the float logits into uint32, which
either crashes the next softmax/sampler call or — if the sampler tolerates
integer input — produces token IDs sampled from quantized integer noise.

### Root cause

Same bug class as #1091 (and as PR #398 / gemma3n before it). The dtype
oracle `embed_tokens.weight.dtype` happens to equal the model's compute
dtype on unquantized models but is `uint32` on quantized ones. In the VLM
issue (#1091 / our PR) the cast was applied to pixel values; here it's
applied to logits, with the same underlying mistake.

### Why it's not in the current PR (#1091 fix)

R2 of our pre-submission review caught it; both reviewers explicitly
recommended keeping it out of the MiniCPM-o/FastVLM PR to keep that scope
focused on the reported bug. No published quantized falcon_ocr or
falcon_perception MLX variant exists on `mlx-community`, so the bug is
purely latent.

### Fix shape (when ready)

Source the dtype from a non-quantized layer — e.g., the model's compute
dtype tracked elsewhere, or `out.dtype` itself (if `out` is already in the
right float). Both files have only one occurrence each, so a tiny
focused PR mirrors the structure of our current MiniCPM-o/FastVLM fix.

### Upstream reference

- `mlx_vlm/models/falcon_ocr/language.py:434`
- `mlx_vlm/models/falcon_perception/language.py:508`
- Sibling fix landed in our PR for #1091 (MiniCPM-o + FastVLM, pixel cast variant)

---

## `apple/FastVLM-7B` — load fails on stock mlx-vlm; tied-only forward pass would silently corrupt outputs even if load were patched

- **First observed:** 2026-04-30 (during MLX FastVLM tier build for oMLX)
- **Status:** **fixed locally** on `contrapuntal/mlx-vlm@fix/fastvlm-untied-lm-head` (commit `10f139e`); HOLDING — push + upstream PR after `Blaizzy/mlx-vlm#1098` merges. oMLX runtime pin bumped to `19e563d` on 2026-05-07 (mlx-vlm sync); both FastVLM bugs (sanitize short-circuit + unconditional tied projection) and `#1098` are still unfixed upstream at that commit.
- **Severity:** any attempt to convert or load `apple/FastVLM-7B` (or any FastVLM variant with `tie_word_embeddings: false`) through stock mlx-vlm fails at load. If only the load were patched, outputs would be silently wrong because the forward pass always uses the tied path even when an untied `lm_head` exists.

### Symptom

`mlx_vlm convert --hf-path apple/FastVLM-7B --mlx-path X --dtype bfloat16 --trust-remote-code` raises during the load phase:

```
File ".../mlx_vlm/utils.py", line 336, in load_model
    model.load_weights(list(weights.items()))
File ".../mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(...)
ValueError: Received 1 parameters not in model:
lm_head.weight.
```

`apple/FastVLM-0.5B` and `apple/FastVLM-1.5B` are unaffected — both ship with `tie_word_embeddings: true` and have no separate `lm_head.weight` in the safetensors.

### Root cause (two interlocking bugs)

1. **Load-time prefix bug** — `mlx_vlm/models/fastvlm/fastvlm.py:215-216` short-circuits on any `lm_head` key in `Model.sanitize`:

   ```python
   if "lm_head" in key:
       return key
   ```

   The fall-through line below would otherwise add the `language_model.` prefix that the FastVLM `Model` hierarchy expects. Without the prefix, `model.load_weights` sees an unknown key and raises.

2. **Tied-only forward pass** — `mlx_vlm/models/fastvlm/language.py:30` unconditionally projects via the embedding matrix:

   ```python
   out = self.model.embed_tokens.as_linear(out)
   ```

   No `if self.config.tie_word_embeddings: ... else: out = self.lm_head(out)` branch (canonical pattern in `qwen2_vl/language.py:531-534` and `qwen2_5_vl/language.py:539-542`). Even if bug #1 were patched in isolation, the loaded `lm_head.weight` would remain dead and the model would compute the output projection from a weight matrix it was not trained against.

The combination of #1 + #2 means **no one has ever successfully run `apple/FastVLM-7B` through stock mlx-vlm.** Existing community uploads of MLX-quantized 7B FastVLM (e.g. `InsightKeeper/FastVLM-7B-MLX-{4,6,8}bit`) either patched their conversion locally and the resulting MLX checkpoints inherit a similar load-time fragility, or they "load" only because the conversion silently dropped `lm_head.weight` and they produce wrong outputs via the tied path.

### Live repro / fix verification

| | Stock `mlx-vlm@main` (or pinned `19e563d`) | `fix/fastvlm-untied-lm-head` (commit `10f139e`) |
|---|---|---|
| Convert `apple/FastVLM-7B` → bf16 | ❌ raises at load | ✅ writes 15 GB to `/Volumes/MacExternalStorage/models/vlm/FastVLM-7B-bf16/` |
| Vision generation on the resulting bf16 model | n/a | ✅ `"A white airplane with blue stripes and a blue tail fin..."` (~1.2 s, M-series) |
| Convert + run `apple/FastVLM-0.5B` and `apple/FastVLM-1.5B` (tied) | ✅ works | ✅ unchanged (non-regression verified) |

### Fix shape

Two-file diff (`mlx_vlm/models/fastvlm/fastvlm.py` + `mlx_vlm/models/fastvlm/language.py`), 8 +/4 −:

- Drop the `lm_head` short-circuit in `Model.sanitize` so `language_model.` is prepended to `lm_head.weight` like every other top-level weight.
- Gate the forward-pass projection on `self.config.tie_word_embeddings`, mirroring `qwen2_vl` / `qwen2_5_vl`.
- Update `LanguageModel.sanitize` to pop the prefixed key (`language_model.lm_head.weight`) since `Model.sanitize` runs first in `utils.py:252-269`.

### When to update oMLX

After **both** `Blaizzy/mlx-vlm#1098` (pixel cast on quantized LMs) and the held FastVLM-untied PR (one-line bump from `10f139e`) merge upstream, bump the `mlx-vlm` pin in `pyproject.toml` past both commits. Until then:

- `/Volumes/MacExternalStorage/models/vlm/FastVLM-7B-bf16/` is loadable only via the `fix/fastvlm-untied-lm-head` worktree at `/Volumes/MacExternalStorage/proj/mlx-vlm-fastvlm-untied/`.
- `/Volumes/MacExternalStorage/models/vlm/FastVLM-1.5B-8bit/` is loadable only via the `fix/quantized-vlm-pixel-dtype-cast` worktree (`/Volumes/MacExternalStorage/proj/mlx-vlm-fix-pixel-dtype/`) since it depends on #1098's pixel-cast fix.
- `/Volumes/MacExternalStorage/models/vlm/FastVLM-0.5B-bf16/` works via the runtime pin today (tied + bf16, hits neither bug).

### Upstream reference

- `mlx_vlm/models/fastvlm/fastvlm.py:215-216` and `mlx_vlm/models/fastvlm/language.py:30,33-38` — bug sites
- `mlx_vlm/models/qwen2_vl/language.py:265-266,531-534` — canonical untied pattern this fix mirrors
- `Blaizzy/mlx-vlm#639` — original FastVLM addition; untied path was never exercised by the test plan
- `Blaizzy/mlx-vlm#1098` (open, our PR) — sibling bug class (pixel cast on quantized LMs) that 1.5B-8bit FastVLM separately depends on

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

## `Model type yasa2 not supported` — `RekaAI/reka-edge-2603` and Yasa2 family

- **First observed:** 2026-05-17 (regression sweep flagged `reka-edge-2603` as 500)
- **Status:** deferred; unfixable at the oMLX layer alone.
- **Severity:** hard 500 on any request. VLM load fails, LLM fallback also fails — `engine_pool` surfaces the chained error correctly (post-Fix-B), so users see *why* both legs failed instead of a generic 500.

### Symptom

```
omlx.engine_pool - WARNING - VLM loading failed for reka-edge-2603, falling back to LLM:
  Model type yasa2 not supported. Error: No module named 'mlx_vlm.speculative.drafters.yasa2'
omlx.server - ERROR - POST /v1/chat/completions → 500 (unhandled):
  VLM load failed: Model type yasa2 not supported. ... ;
  LLM fallback also failed: Model type yasa2 not supported.
```

### Root cause

`reka-edge-2603`'s `config.json` declares `model_type: yasa2`, `architectures: ['Yasa2ForConditionalGeneration']`, `text_config.model_type: yasa_model`. Neither `mlx_vlm.models.yasa2` nor `mlx_lm.models.yasa2` (nor `yasa_model`) exists. mlx-vlm's load path also probes for a speculative-decoding drafter at `mlx_vlm.speculative.drafters.yasa2`, which is what produces the `No module named` portion of the error — but the real gap is the model module itself.

### Resolution requires

- Upstream `mlx_vlm.models.yasa2` (full model + processor + image processor) plus `mlx_lm.models.yasa_model` (text backbone), or
- A local model module + alias entry — similar shape to the jina-vlm hand-off above. Substantial work; not a one-line shim.

### Workaround

None at the model level. Use a different VLM until upstream support lands. Reka's checkpoints predating Yasa2 (e.g. the earlier `reka-flash` / `reka-edge-2024` builds, if available as mlx-vlm-compatible) may work depending on their `model_type`.

---

## `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mxfp8` — published checkpoint has outlier mxfp8 scales, dequant overflows fp16

- **First observed:** 2026-05-16 (image-input regression sweep)
- **Status:** **not an oMLX, mlx-vlm, or mlx-lm bug** — the published mxfp8
  checkpoint itself has pathological dequantization scales. Reported as an
  HF discussion against the model card on 2026-05-17. Local copy of the
  broken artifact deleted; 4-bit sibling kept and verified working.
- **Severity:** every generated token argmaxes to `<unk>` (id 0); output is
  unusable on any input including text-only.

### Symptom

Any forward pass produces sequences of `<unk>` tokens to `max_tokens`,
`finish_reason: length`. Text-only and image+text identical.

### Root cause

The published mxfp8 weights dequantize past fp16 range in **30 of 162
language-layer scale tensors**. E8M0 dequant factor is `2^(scale − 127)`;
the worst observed scale value is 255 (= `2^128 ≈ 3.4e38`, fp32 infinity
boundary). Any encoded scale ≥ 144 produces a dequantized value past
fp16's 65 504 limit, so the matmul overflows to `inf` and propagates NaN
through the residual stream.

Layer-by-layer bisection localizes the first NaN to layer 1's MoE block —
`shared_experts.up_proj` and `switch_mlp.fc1`. Both return `inf` even on
tiny inputs (`ones * 0.001`), so the dequantized weight matrices are the
overflow source, not the activations. The pathology is concentrated in
every MoE layer's `switch_mlp.fc1` (23 tensors), six `shared_experts.up_proj`
in early MoE layers, and one `mixer.in_proj` at layer 46.

The conversion path used (mlx-vlm 0.4.5 → `mlx_vlm/convert.py` →
`nn.quantize`) does not clamp outlier scales when assigning E8M0; a single
large-magnitude weight in a 32-element group sets the scale to 255.

### Not the cause

- Not a download corruption issue (local SHA-256 of shard 1
  `ed59e28f...0b306` matches HF exactly; all 7 shard sizes match).
- Not an mxfp8 dequant kernel bug (MLX correctly decodes scale 255 as
  `2^128`; the issue is that the encoded scale should never have been 255).
- Not specific to NemotronH / Mamba (the affected modules are generic
  `QuantizedLinear` and `QuantizedSwitchLinear`; the same overflow would
  happen for any mxfp8 model whose conversion produced such scales).
- Not in mlx-vlm's `nemotron_h_nano_omni` wrapper — that delegates entirely
  to `mlx_lm.models.nemotron_h`; the bug is one layer further down, in
  the checkpoint artifact's scale values.

### Upstream reference

- HF discussion against the model card (filed 2026-05-17):
  https://huggingface.co/mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mxfp8/discussions
- [Blaizzy/mlx-vlm#1088](https://github.com/Blaizzy/mlx-vlm/pull/1088)
  (MERGED 2026-04-29) — introduced the `nemotron_h_nano_omni` module.
  Not the source of this bug; cited for context.

### Workaround

`mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit` (affine,
group_size 64) runs cleanly on the same reproducer. Verified end-to-end
2026-05-17. The 5-bit and 6-bit affine variants are likely fine but
untested.

### Secondary fix candidate (not yet filed)

`mlx-lm`'s `nn.quantize` for the mxfp8 path should clamp encoded scales
at ≤ 143 (≈ `2^16`, just below fp16 max) so future conversions don't
reproduce this for other models. Worth filing against `ml-explore/mlx-lm`
separately if time permits.

### When to remove this entry

- The model uploader re-uploads `Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mxfp8`
  with proper scales (re-quantization from `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`).

---

## `apple/FastVLM-1.5B` — benchmark-VQA output format, not a bug

- **First observed:** 2026-05-16 (image-input regression sweep)
- **Status:** **not a bug** — model training artifact, by Apple's design.
- **Severity:** outputs are coherent but heavily biased toward
  `Answer: <letter>` / `Answer: <very short option>` shape on free-form
  prompts. Some prompts yield prose; many resolve to a single number or
  letter. Downstream chat clients can reasonably classify the output as
  "unanswerable."

### Symptom

- Text-only `"Reply with the word HELLO in capitals."` → `"HELLO"` (clean).
- Text-only `"What color is the sky?"` → `"Answer: Blue"`.
- Image + `"Describe the dominant color in one word."` → `"Answer: 1"`.
- Image + `"What is the color of this image?"` → `"Answer: 1"`.
- Image + `"What color is dominant?"` → `"Answer: c) Ditoryl"`.

The model is *seeing* the image (text-only color question answers
correctly, distinct images produce different outputs). It tends to
collapse to a benchmark-VQA-shaped answer when image input is present.

### Not the cause

- Not the chat-template handling of multimodal `content` lists. Both
  FastVLM-0.5B and FastVLM-1.5B ship the same 5-line `chat_template.jinja`;
  0.5B produces prose like "The image contains various shades of blue,
  green, and a touch of yellow" on identical inputs.
- Not an image-token-integration bug — text-only produces the same
  multiple-choice format with no image present.

### Upstream evidence (it's the training, not the code)

The [`apple/FastVLM-1.5B` model card](https://huggingface.co/apple/FastVLM-1.5B)
evaluation table covers exclusively benchmark-VQA targets — Ai2D,
ScienceQA, MMMU, VQAv2, ChartQA, TextVQA, InfoVQA, DocVQA, OCRBench,
RealWorldQA, SeedBench-Img — every one of which expects a number,
letter, or short answer. No general-chat evaluation appears anywhere
in the card. The model is performing exactly as evaluated.

[`apple/ml-fastvlm#64`](https://github.com/apple/ml-fastvlm/issues/64)
(open since 2025-10-02, no maintainer response) reports a related issue
on FastVLM-0.5B — model doesn't emit EOS after answering a multi-choice
benchmark prompt. Different mechanism (missing stop token after the
benchmark answer) but reinforces the picture that Apple ships FastVLM
as benchmark-tuned, not chat-tuned.

No issue in `apple/ml-fastvlm` or discussions on the HF model card
addresses free-form chat behavior for the 1.5B variant. There is no
upstream PR to wait on.

### Workaround

Use FastVLM-0.5B (small, prose-leaning but inaccurate) or FastVLM-7B
(after PR [Blaizzy/mlx-vlm#1193](https://github.com/Blaizzy/mlx-vlm/pull/1193)
or its eventual upstream equivalent — required for the untied `lm_head`
to be wired). On the same slate-blue test image, FastVLM-7B answered
"Blue" correctly to "Describe the dominant color in one word."; 1.5B
returned "Answer: 1" on the same prompt.

If a downstream client reports `"HUNG"` on FastVLM-1.5B, that is most
likely a streaming-timeout artifact, not a model hang: a `"Answer: 1"`
response with `finish_reason: stop` after 2 tokens may be too short for
clients expecting longer streams.

---

# Resolved workarounds (local patches, upstream still open)

## LFM2.5-VL-* silent fallback to text-only LLM

- **First observed:** 2026-04-20 (re-diagnosed 2026-06-13)
- **Status:** **fixed upstream** in the `contrapuntal/mlx-vlm` fork
  (branch `fix/lfm2-vl-projector-honor-layernorm-flag`, pushed; PR to
  `Blaizzy/mlx-vlm` not yet opened). **No active carry on `fork-main`** —
  the older `_patch_lfm2_vl_projector_layernorm()` monkey-patch (commit
  `b07f15a`) lives only on `dev-legacy` and was never carried forward.
- **Upstream:** `Blaizzy/mlx-vlm` **#1000, #1001, #1002** (related, open).

### Symptom

Two distinct load failures depending on the checkpoint, both ending in a
silent fall back to text-only LLM (vision capability lost without an
obvious signal):
- `Missing 2 parameters: multi_modal_projector.layer_norm.{bias,weight}`
  — checkpoint sets `projector_use_layernorm: False` and ships **no**
  layer_norm weights (e.g. `LFM2.5-VL-1.6B-bf16`).
- `Received 2 parameters not in model:
  multi_modal_projector.layer_norm.{bias,weight}` — checkpoint sets the
  flag `False` but ships the weights anyway (e.g.
  `LFM2.5-VL-450M-MLX-bf16`).

### Root cause

The `projector_use_layernorm` flag is unreliable across LiquidAI's
LFM2.5-VL exports — config and weights disagree, and the disagreement
goes in both directions:

| Checkpoint | `projector_use_layernorm` | `layer_norm` weights |
| --- | --- | --- |
| LFM2.5-VL-1.6B-bf16 | `False` | absent |
| LFM2.5-VL-450M-MLX-bf16 | `False` | present |
| LFM2-VL-{450M,1.6B,3B}-bf16 | absent (→ default `True`) | present |

mlx-vlm 0.6.3's `Lfm2VlMultiModalProjector.__init__` allocated
`nn.LayerNorm` **unconditionally**, ignoring the flag, so the
weights-absent 1.6B failed `load_weights(strict=True)` with "Missing 2".
(This is the opposite of the original 2026-04-20 diagnosis, which
described mlx-vlm creating `nn.Identity` and rejecting *extra* weights —
that no longer matches the pinned mlx-vlm, which regressed to
unconditional allocation.)

### Fix

- **Upstream (mlx-vlm):** `Lfm2VlMultiModalProjector` now builds
  `layer_norm` only when `projector_use_layernorm` is set (matching the
  HF `transformers` reference), and `sanitize` drops orphan
  `layer_norm.*` weights when the flag is `False`. Fixes the 1.6B and any
  HF-format inconsistent export. The omlx `.venv` copy of `lfm2_vl.py`
  is surgically patched to match, pending an `mlx-vlm` pin bump.
- **450M-MLX checkpoint:** `load_model` skips `sanitize` for
  `format=mlx` checkpoints, so the orphan-drop cannot reach this export.
  Its two orphan `layer_norm.*` tensors were stripped in place (lossless
  — never applied at `flag=False`); originals saved as
  `model.safetensors.orig` + `.index.json.orig`.

### When this is fully resolved

- `Blaizzy/mlx-vlm` merges the projector fix and the `mlx-vlm` pin is
  bumped past it — then the `.venv` surgical patch is no longer needed.
- LiquidAI republishes LFM2.5-VL checkpoints with consistent
  config/weights.

---

## Molmo2 family — classifier gap routes bf16 variant through LLM fallback

- **First observed:** 2026-05-16 (Molmo2-8B-bf16 image+text both 500)
- **Status:** **fixed in-tree** (`omlx/model_discovery.py`); upstream PR candidate.
- **Severity:** any Molmo2 variant whose `config.json` omits a top-level
  `vision_config` placeholder (the `bf16` checkpoint on HF) classified as
  `llm`, routed through `BatchedEngine` (mlx-lm), and 500'd with
  `Model type molmo2 not supported.` in ~10 ms. Quantized siblings (8bit,
  mxfp8) happened to ship an empty `vision_config: {}` placeholder so the
  classifier's vision-config heuristic catch-all rescued them.

### Symptom

```
omlx.engine_pool - INFO - Loading model: Molmo2-8B-bf16
omlx.server - ERROR - POST /v1/chat/completions → 500 (unhandled):
  Model type molmo2 not supported.
```

No engine start, no VLM warning. Failure fires before `engine_pool.py`'s
fallback try/except, so the combined-error surfacing (engine_pool VLM-fallback)
doesn't apply — the engine was never typed as VLM in the first place.

### Root cause

`omlx/model_discovery.detect_model_type` had two gaps for Molmo2:

1. **`VLM_ARCHITECTURES`** listed `MolmoForCausalLM` (original Molmo) but
   not `Molmo2ForConditionalGeneration` (Molmo2's actual class). With the
   architecture missing, the architecture-based VLM check at line 471 fell
   through.
2. **`VLM_MODEL_TYPES`** listed `molmo` but not `molmo2`. With the
   model_type missing, the model_type-based VLM check at line 478 also
   fell through.

Detection landed on the fallback heuristic `if "vision_config" in config:
return "vlm"`. Molmo2-8B-8bit and mxfp8 have `vision_config: {}` (empty
placeholder) so this fires; bf16 has no `vision_config` key at all, so it
fell all the way through to `return "llm"`.

A third gap: even after adding Molmo2 to the two sets, the "text-only
quant" guard at line 473 broke bf16 again — it treats `VLM_MODEL_TYPES`
match + no `vision_config` as "text-only quant" and returns LLM. Molmo
family stores its vision sub-config under `vit_config`, not `vision_config`.

### Local fix

`omlx/model_discovery.py`:

- Add `"Molmo2ForConditionalGeneration"` to `VLM_ARCHITECTURES`.
- Add `"molmo2"` to `VLM_MODEL_TYPES`.
- Extend the "text-only quant" guard to accept either `vision_config` or
  `vit_config` as evidence the model retains its vision tower.

### Verified

All three Molmo2 variants (`-bf16`, `-8bit`, `-mxfp8`) now classify as
`vlm` and produce coherent text + image responses end-to-end. Tested on
a 224×224 solid-blue PNG: bf16 → `"The image is completely blue…"`;
8bit → `"The image is a uniform dark blue color…"`; mxfp8 →
`"The image is predominantly navy blue…"`.

### Upstream candidacy

Bundle with the other `model_discovery` / `VLMModelAdapter` fixes for a
single upstream PR against `jundot/omlx`.

---

## VLMModelAdapter.layers — `.blocks` vs `.layers` AttributeError on Molmo-family + Moondream3

- **First observed:** 2026-05-16 (Molmo2-8B-mxfp8 image request hangs)
- **Status:** **fixed in-tree** (`omlx/models/vlm.py`); upstream PR candidate (already on the bundle list below).
- **Severity:** hangs / mid-stream 500 on any request to a VLM whose inner language transformer names its decoder stack `.blocks` instead of `.layers`. Affects `molmo`, `molmo2`, `molmo_point`, `moondream3` — all currently-published quants. Error fires inside the engine loop after the 200-OK header is sent, so clients see a stalled stream rather than a clean error.

### Symptom

Any chat-completion request against an affected model hangs until `--max-time`. Server log shows:

```
omlx.engine_core - ERROR - Engine loop error: 'Molmo2Transformer' object has no attribute 'layers'
  File "omlx/scheduler.py:1676" in _do_external_prefill
  File "mlx_lm/models/cache.py:32" in make_prompt_cache
  File "omlx/models/vlm.py:107" in make_cache
  File "omlx/models/vlm.py:75" in layers
AttributeError: 'Molmo2Transformer' object has no attribute 'layers'
```

(class name varies — `MolmoTransformer`, `Molmo2Transformer`, `MolmoPointTransformer`, `Moondream3Transformer`.)

### Root cause

`VLMModelAdapter.layers` in `omlx/models/vlm.py` was hardcoded to `self._language_model.model.layers`. This assumes every VLM's inner transformer names its decoder-layer stack `self.layers`. A survey of mlx-vlm's 49 `language.py` modules shows 45 use `self.layers` and **4 use `self.blocks`** — the Molmo family and Moondream3.

When `scheduler._do_external_prefill` → `make_prompt_cache(self.model)` → `model.make_cache()` → `[KVCache() for _ in range(len(self.layers))]`, the property raises `AttributeError`. The engine loop catches it but the request never receives a response body.

### Local fix

`omlx/models/vlm.py:72-89` — the property now falls through a four-quadrant attribute landscape: `model.layers` → `model.blocks` → flat `layers` → flat `blocks`. First hit wins. Existing `.layers`-based models keep their fast path; `.blocks`-based models get covered without per-model special-casing.

### Verified

`mlx-community/Molmo2-8B-mxfp8` after the fix: text-only `"reply with one word"` → `"Hello!"`; image input on a solid-blue test PNG → `"The image is a solid dark blue color. There are no other colors or variations…"`. Other affected families (`molmo`, `molmo_point`, `moondream3`) have not been live-tested but go through the same code path.

### When to remove this patch

- Upstream mlx-vlm normalises the decoder-stack name across all 4 families (e.g. renames `Molmo2Transformer.blocks` → `.layers` in a backwards-compatible pass), or
- Upstream `jundot/omlx` accepts a PR that ships this multi-path layers property.

### Upstream candidacy

Clean candidate for the `jundot/omlx` bundle PR listed below (already cited there as "VLMModelAdapter.layers → flat+nested pattern handling").

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

---

## transformers `is_torch_available()` stale-cache + DummyObject persistence

- **First observed:** 2026-04 (LFM2-VL-3B-bf16 HTTP 500 on image requests)
- **Status:** **fixed locally** in `omlx/engine/vlm.py`; upstream issue deliberately not filed (see "Why not upstream" below).
- **Severity:** any long-running oMLX process started before torch was installed in its venv permanently sees `is_torch_available()=False` for the lifetime of the process. Image-bearing requests to models whose Transformers processor path expects torch (LFM2-VL, etc.) return HTTP 500.

### Symptom

```
omlx.server - ERROR - POST /v1/chat/completions → 500: Internal server error
```

for any image-bearing request to LFM2-VL-3B-bf16 (and similar Transformers-fast-processor VLMs), even when `torch` is installed in the venv. Text-only requests to the same model work. Restarting oMLX resolves it — until the next time torch is added mid-session.

### Root cause

Two interlocking pieces of Transformers' lazy-import machinery:

1. `transformers.utils.is_torch_available()` is decorated with `functools.lru_cache`. The first caller in a process locks in the answer until `cache_clear()` or process restart. If oMLX is started before torch is on the path and any import-time code calls `is_torch_available()`, the result `False` sticks.
2. Transformers' `_LazyModule.__getattr__` resolves `AutoImageProcessor` (and friends) on first access. When `is_torch_available()` is cached `False` at that moment, the class binds to a `DummyObject` subclass and sets it on `transformers.AutoImageProcessor` via `setattr`. Subsequent attribute reads return the cached dummy directly, bypassing `__getattr__` entirely — independent of any later `cache_clear()`.

The combined effect: the cache_clear call is necessary but insufficient. Once the lazy module has bound a dummy attribute, only a process restart re-resolves it.

### Local workaround

`omlx/engine/vlm.py:_patch_torch_free_image_processor()`:

1. Per model load, clear the two relevant `@lru_cache` slots:

   ```python
   from transformers.utils.import_utils import (
       is_torch_available, is_torchvision_available,
   )
   is_torch_available.cache_clear()
   is_torchvision_available.cache_clear()
   ```

   This catches the no-torch → torch transition for any consumer of `is_torch_available()` that re-evaluates per call (`requires_backends`, `_resolve_backend`, etc.).

2. If `transformers.AutoImageProcessor.is_dummy` is still True *and* the cleared `is_torch_available()` now returns True, log a WARNING telling the user to restart oMLX. Fall through to a PIL fallback for OCR processors in the meantime.

The split is honest about the limit: cache-clear handles the recoverable case, the warning handles the unrecoverable one.

### Why not upstream

PR `Blaizzy/mlx-vlm#1189` ("Clear cached torch availability checks in install_auto_processor_patch") was filed 2026-05-19 and closed by the maintainer with "Please open an issue / We don't use torch as a dependency." The issue follow-up was drafted (see `docs/drafts/mlx-vlm-torch-cache-staleness-issue.md`) but **not filed**, for two reasons:

1. Adversarial review confirmed the proposed fix can't recover from already-bound dummy attributes — i.e., the upstream fix would not be complete on its own. The local oMLX implementation acknowledges this with its restart-required warning; upstream has no such acknowledgment surface.
2. The bug only matters in long-lived consumer processes (oMLX is one). Asking mlx-vlm to manage Transformers' cache lifecycle on every model load — without a complete recovery path — is a weak ask versus the maintainer's stated stance.

### When to remove this patch

Drop the patch once any of the following is true:

- Transformers exposes a public API that lets consumers force-re-resolve `_LazyModule` bindings, *and* that API is wired through Transformers' optional-backend machinery (no equivalent exists today).
- mlx-vlm changes its `AutoProcessor.from_pretrained` patching to either (a) not rely on transformers' lazy-bound `AutoImageProcessor` at all, or (b) refresh the binding itself before chaining.
- The oMLX startup path is restructured to guarantee `is_torch_available()` is evaluated only after the venv is fully initialised — at which point the cache_clear becomes provably unnecessary and the patch can be replaced by a one-time check.

### Reference branch

`contrapuntal/mlx-vlm:fix/clear-stale-torch-availability-cache` — the closed PR's diff. Demonstrates the per-load `cache_clear` shape if the project ever decides to upstream a partial fix.

---

## Contextual AI Reranker v2 family (`ContextualAI/reranker_v2_*`, `MistralForCausalLM`) — unsupported by current reranker engine

- **First observed:** 2026-05-19 (ctxl-rerank-v2-6b-bf16 served via `/v1/rerank`)
- **Status:** **unsupported**; no in-tree workaround. Routing-only classifier change explored and discarded.
- **Severity:** the model loads (it's a valid MLX-converted MistralForCausalLM checkpoint) but cannot produce correct reranker scores via any current oMLX path.

### Symptom

`ctxl-rerank-v2-6b-bf16` (directory name carries `rerank`) classifies as `llm`, so `/v1/chat/completions` is the only routed path. That path fails immediately with `Chat template error: Cannot use chat template functions because tokenizer.chat_template is not set` — the checkpoint ships no chat_template.

### Root cause

The model needs a different reranker contract than oMLX's current CausalLM reranker engine supports. Per the model card's documented inference:

1. **Custom prompt template, no chat_template.** Inputs are formatted as a raw string: `Check whether a given document contains information helpful to answer the query.\n<Document> {doc}\n<Query> {query} {instruction} ??`
2. **bf16-token-as-score decoding.** A custom `logits_processor` constrains the first generated token; the token id's bytes are then reinterpreted as a `bf16` float to recover a continuous relevance score (the README shows scores like `-2.2969`, `-4.6875`).

This is fundamentally different from oMLX's `MLXRerankerModel` (omlx/models/reranker.py), which:

- Calls `tokenizer.apply_chat_template(...)` at engine init (line 303) — fails immediately for this checkpoint.
- Resolves `yes` / `no` token ids via `tokenizer.convert_tokens_to_ids("yes")` / `"no"` (lines 286-287) — Mistral's SentencePiece tokenizer encodes word-start tokens as leading-space variants (`▁yes`), so bare `"yes"` returns UNK. Even if the chat_template hurdle were cleared, scoring would compare logits at the UNK position — meaningless.
- Appends `<think>\n\n</think>\n\n` to the suffix (line 314) — Qwen3-specific thinking-then-answering format.
- Computes scores via softmax over `logit[yes]` vs `logit[no]` — incompatible with the bf16-encoded-token-id mechanism the model expects.

### Why the routing-only fix was discarded

Closed PR [jundot/omlx#1313](https://github.com/jundot/omlx/pull/1313) added `MistralForCausalLM` to `CAUSAL_LM_RERANKER_ARCHITECTURES`. The classification works — `detect_model_type` returns `"reranker"` — but routing to the reranker engine just moves the failure from "no chat_template at chat serving" to "no chat_template at reranker engine init," and patching the chat_template hurdle would still leave the three downstream incompatibilities above. Closed after codex + gemini adversarial reviews surfaced the layered failure modes.

### What real support would require

A new engine type (or a `ContextualAIReranker` subclass of the existing reranker model):

- Custom prompt formatter using the documented `<Document>` / `<Query>` template — bypass `apply_chat_template` entirely.
- Custom logits processor + bf16-reinterpretation score decoder mirroring the model card's `infer_w_vllm` reference implementation.
- Tokenizer hygiene: handle SentencePiece word-start variants instead of bare `"yes"` / `"no"` lookups.

Worth opening as a feature request against `jundot/omlx` if there is demand for the family. Reference branch: `contrapuntal/omlx:fix/mistral-causal-lm-reranker-classifier` (the discarded routing-only attempt).

### Workaround for users today

None at the oMLX layer. Users wanting to evaluate `ContextualAI/reranker_v2_6b` should run the reference `infer_w_vllm` / `infer_w_hf` code from the model card directly (vLLM ≥ 0.8.5 BF16 or `transformers` ≥ 4.51.0). oMLX won't see the model as a usable reranker without the engine work above.
