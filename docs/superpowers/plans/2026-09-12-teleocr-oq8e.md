# TeleOCR MLX and oQ8e implementation plan

Goal: load StarDoc-AI/TeleOCR correctly in oMLX and produce a validated oQ8e checkpoint.

The adapter reuses Qwen2.5-VL image processing, vision tower, and position indexing,
while implementing TeleOCR's explicit head dimension, bias-free attention, and Q/K
normalization. Recognition requires custom auto_map metadata or a saved adapter
marker, corroborated by checkpoint attention headers. Unidentified nonstandard
Qwen geometry is rejected. Confirm tensor shapes and embedding/head ownership
against the pinned reference checkpoint.

## Constraints

- Isolated branch feat/teleocr-oq8e, based on fork-main because conversion depends
  on its existing oQe carries. Main checkout remains on fork-main.
- Preserve existing Qwen2.5-VL behavior and current dependency pins.
- Production inference must not need PyTorch or execute remote model Python.
- Full PDF layout/cropping/assembly orchestration is outside this change.
- Tests exercise model math, cached inference, checkpoint round trips, and routing.
- Store downloaded weights and validation outputs outside the tracked tree.

## Disposition and upstreaming

Retain the topic branch for review. If landed on fork-main, this is a temporary
carry needed because the current mlx-vlm pin lacks native TeleOCR support.
Prepare the decoder, checkpoint routing, and model/math/cache tests as an independent
mlx-vlm contribution; oQe calibration and the serving failure policy remain in
oMLX. No upstream submission has been made. Once native support is accepted and
pinned here, remove the local decoder/interception and verify existing marked
oQ8e exports still load. Each intervening pin bump requires adapter, conversion,
engine-pool, and real image-load validation. See docs/teleocr.md for the transition.

## Task 1: Reference inventory

- [x] Pin TeleOCR revision and download checkpoint, processor, and example images.
- [x] Audit attention, rotary positions, vision, embedding/head ownership, tensor names.
- [x] Generate deterministic reference outputs and first-token logits using an isolated
  PyTorch reference process. Record any dependency or hardware constraints explicitly.

## Task 2: Adapter

- [x] Add omlx/patches/mlx_vlm_teleocr_compat/ with detection, installation, and model.
- [x] Write failing tests in tests/test_mlx_vlm_teleocr_compat.py before implementation.
- [x] Preserve head_dim=128, Q/K RMSNorm, bias-free projections, Qwen2.5-VL mRoPE.
- [x] Support original and MLX tensor layouts, strict loads, cached and uncached forwards.
- [x] Review adapter for compatibility and numerical correctness.

## Task 3: Loading and conversion integration

- [x] Install the adapter from omlx/utils/model_loading.py and omlx/oq.py.
- [x] Keep raw config/provenance and adapter detection after export/reload.
- [x] Test standalone sanitization, calibration forwards and complete tiny conversion.
- [x] Confirm all expected decoder projections collect importance statistics.

## Task 4: Real validation

- [x] Validate BF16 against the reference on text, table, formula, code, layout images.
- [x] Produce oQ8 and oQ8e checkpoints and compare fixed prompts/outputs.
- [x] Run image-chat API, streaming, repeated/cached request and reload smoke checks.
- [x] Record size, memory, timing, quality and observed limitations.

Acceptance: no unexplained missing tensors; deterministic FP32 attention parity within
2e-5 absolute/relative tolerance; full versus cached logits within 2e-5 for FP32 tiny
fixtures; all 28 decoder layers calibrated; converted models reload with correct
adapter; no malformed/repetitive regressions on the supplied smoke set. OCR results
are reported against BF16; a small smoke set does not establish benchmark parity.

## Task 5: Delivery

- [x] Run targeted regression tests and verify distributable includes adapter.
- [x] Document conversion, serving, reference revision, validation and limitations.
- [x] Obtain independent review, resolve findings, and leave the topic branch reviewable.

## Review follow-up

- [x] Reject unknown root/nested head geometry without selecting or labeling TeleOCR.
- [x] Surface TeleOCR VLM failures instead of accepting an automatic text-only fallback.
- [x] Log adapter installation after metadata and attention-header validation.
- [x] Preserve upstream constructor state and root head_dim through nested rebuilding.
- [x] Stamp both exported configs explicitly; strictly reload conversion and proxy outputs.
- [x] Record the temporary carry rationale and upstream removal conditions.
- [x] Retain positive fallback-refusal evidence from discovery through load; unreadable
  configs cannot erase it. Preserve ordinary fallback when there is no such evidence.
- [x] Run GC and MLX cache cleanup before refusing fallback.
- [x] Propagate checkpoint/config validation failures through serving and conversion.
- [x] Enforce strict full weight binding even when calibration requests strict=False.
- [x] Lock dispatcher installation and verify concurrent installs preserve stock routing.
- [x] Map model.lm_head aliases, drop only position_ids buffers, reject windowed variants.
- [x] Guard inherited layer state and the replaced attention surface across pin bumps.
- [x] Confine shard/index paths, including unindexed files, while allowing standard
  Hugging Face links to the same model repository's real blob directory.
- [x] Reject TeleOCR text-only exports before weight reads and output creation.
- [x] Keep unknown-geometry rejection in oMLX entry points; preserve subsequent loader
  wrappers without repeated installation of the global dispatcher.
- [x] Verify chunked prefill with cache-generated array masks against full attention;
  retain the existing slice order and strict binding based on these contracts.
