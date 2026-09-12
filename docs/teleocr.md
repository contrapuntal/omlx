# TeleOCR on oMLX

oMLX can load the original `StarDoc-AI/TeleOCR` checkpoint and convert it directly
to oQ8e. This requires the TeleOCR compatibility adapter in this branch. The model
declares `qwen2_5_vl` but has a custom decoder, so stock Qwen2.5-VL support alone
is insufficient.

## Conversion

Download the original BF16 model into an oMLX model directory. In the quantization
panel select **oQ8**, enable **enhanced/oQe**, retain **BF16**, and leave **Text
Only** disabled. The output can be discovered and served normally after conversion.
TeleOCR text-only conversion is rejected before reading weights or creating output.

The equivalent Python conversion, run from an environment containing this code:

```python
from omlx.oq import quantize_oq_streaming

quantize_oq_streaming(
    model_path="/path/to/TeleOCR",
    output_path="/path/to/TeleOCR-oQ8e",
    oq_level=8,
    enhanced=True,
    dtype="bfloat16",
    text_only=False,
    imatrix_strict=True,
)
```

The output directory must not already exist. Default enhanced calibration uses
128 sequences of 512 tokens from oMLX's built-in text corpus. Vision weights and
normalization parameters remain in floating point. The token embedding is
quantized with standard oQ; the decoder projections receive importance weighting.
This text calibration does not establish document-specific accuracy improvements.

## Serving and prompts

Place the converted directory under the server's configured model directory and
use `/v1/chat/completions` with an image and one of the reference task prompts:

- Text: `Please output the text content from the image.`
- Table: `This is the image of a table. Please output the table in OTSL format.`
- Formula: `Please write out the expression of the formula in the image using LaTeX format.`
- Code: `The image contains a code snippet, please output the parsing result.`
- Layout: `Analyze the image layout.`
- Distorted layout: `\nMulti-point Layout Segmentation Analysis.` (leading newline).

Use temperature 0 and allow up to 4096 output tokens for these examples. The
reference layout examples resize to 1036 × 1036 with bicubic interpolation before
inference. The source processor specifies a 12,845,056-pixel maximum; oMLX model
settings may impose additional image limits. Tables emit OTSL and layout requests
emit geometry/labels. PDF rasterization, cropping, table-to-HTML conversion, and
complete document assembly belong to the separate [TeleOCR pipeline](https://github.com/caipeng328/TeleOCR).

## Compatibility details

The adapter implements explicit 128-dimensional attention heads, bias-free Q/K/V/O
projections, and per-head Q/K RMSNorm before Qwen2.5-VL's multimodal rotary position
encoding. It reuses the existing vision tower, torch-free processor, and cache
position handling. Recognition requires the checkpoint's `modeling_naviocr`
auto-map or the saved `omlx_adapter: teleocr` marker. Unidentified `qwen2_5_vl`
checkpoints with nonstandard head geometry are rejected; geometry alone never
selects TeleOCR or stamps an export. Nested text fields take precedence over root
fields. A re-upload missing both identifiers requires an architecture audit before
adding metadata; do not infer compatibility solely from its head dimension.

Before serving or streaming conversion selects the decoder, safetensors headers
must corroborate all layers' Q/K norms, bias-free projections, and raw or affine
quantized attention shapes. No tensor materialization is needed for this check.
Shard names cannot be absolute or traverse parent directories. Resolved shard and
index paths must stay inside the model directory, with an exception for standard
Hugging Face snapshot links into that model repository's real `blobs` directory.
Unindexed shards are checked too, before conversion opens any weight headers.
Full weight binding is always strict, including calibration paths requesting
`strict=False` and MLX-formatted checkpoints that skip sanitization. An adapter
marker is an architecture claim, not permission to bypass validation. Serving
does not execute the remote Python model.

The server logs TeleOCR adapter installation. If a recognized TeleOCR checkpoint
fails VLM loading, the pool stops the failed engine, collects garbage, synchronizes
MLX, clears its cache, and surfaces the load error. The refusal of automatic
text-only fallback is captured during discovery and refreshed before loading, so
losing or corrupting config.json after recognition cannot erase it. Missing,
unreadable, or unresolved configs without prior TeleOCR/unsupported-geometry
evidence retain the existing fallback behavior for ordinary models.
Both streaming quantization and sensitivity-proxy export explicitly stamp the
adapter marker from metadata after checkpoint validation. TeleOCR sanitize setup
errors propagate instead of degrading to the generic LLM sanitizer.

The loader dispatcher remains process-wide and installs once under a lock;
checkpoints without TeleOCR metadata delegate to the previous loader. Unsupported
geometry is rejected at oMLX admission/conversion, not by the global dispatcher.
Later wrappers are preserved even if they omit `functools.wraps`. It stays installed
for reloads; replacing the entire loader chain requires preserving this hook.
Native upstream model registration is the planned replacement.

Only full text attention is supported. The pinned checkpoint declares
`use_sliding_window: false` and 28 `full_attention` layer entries; windowed variants
are rejected. Parent constructors are retained and their stock components are
replaced before checkpoint materialization. MLX builds lazy initialization graphs,
but repeated construction remains a cost. Tests preserve non-attention layer
state and flag changes to the replaced upstream attention surface for review.
Array-valued attention masks must cover cached keys plus new keys, as provided by
the installed cache API. Chunk-local square masks do not satisfy that contract;
moving the mask slice before cache update would truncate valid prefix masks.

The original checkpoint stores equal embedding and output-head tensors. Its
declared Transformers 4.57.1 runtime hoists the nested `tie_word_embeddings: true`
and shares those parameters. The adapter preserves that behavior and removes the
redundant head. An explicit root override is respected; an untied variant also
receives output-head importance statistics during oQe calibration.

## Validation record

Source revision: [`73759a2db254bee30618af0c10f7aebd4e1fa231`](https://huggingface.co/StarDoc-AI/TeleOCR/tree/73759a2db254bee30618af0c10f7aebd4e1fa231).
Validation used Apple Silicon, MLX 0.32.2, the repository's installed mlx-vlm,
and an isolated PyTorch/Transformers 4.57.1 reference process.

- Full BF16 checkpoint and processor load strictly, with 28 decoder layers and
  the expected head width and tied embeddings.
- FP32 synthetic attention matches an independent NumPy reference within 2e-5.
  Actual image prefill plus cached decoding matches full-sequence logits within
  2e-5 on a small FP32 model.
- On identical processor inputs for text, table, formula, code, and layout, BF16
  MLX and the original PyTorch runtime selected the same first token. Logit cosine
  similarity ranged from 0.999628 to 0.999990; mean absolute error ranged from
  0.0139 to 0.0398. Native processors produced identical token IDs and image grids,
  with small pixel-resampling differences. These are numerical comparisons, not
  a claim of bitwise equivalence between runtimes.
- Full reference generations used CPU BF16 for text/table/formula and MPS BF16
  for code/layout; all first-token reference logits used CPU BF16. Text, formula,
  code, and layout outputs matched native MLX after trimming outer whitespace.
- Strict default oQ8e calibration collected 196 decoder projection entries,
  completing all 28 layers with no skipped forwards or missing importance entries.
- Both oQ8 and oQ8e generated complete responses on the six supplied text, table,
  formula, code, layout, and distorted-layout examples at a 4096-token limit.
  Five outputs exactly matched native MLX BF16. The table differed by one
  character out of 433 for both quantizations. These are differences against
  BF16 output, not ground-truth OCR error rates. This smoke set shows no advantage
  of oQ8e over oQ8.
- The original checkpoint is 2,830,223,488 bytes; each quantized safetensors file
  is 1,960,379,838 bytes (1.96 GB / 1.83 GiB). Retaining the large vision tower
  limits the size reduction. Default strict oQ8e conversion took about 22 seconds
  on the validation machine; this is not a general performance guarantee.
- The actual oMLX HTTP server passed image requests, a repeated streaming request,
  concurrent text/formula requests, and unload/reload. Reload successfully restored
  a 256-token paged prefix cache and reproduced the text response.
- 538 affected tests passed. A wheel built successfully and contains both adapter
  modules. No new production dependencies or Metal kernels are required.
- Review follow-up passed 723 tests, including the full engine-pool suite,
  missing/renamed metadata routing, root-level head dimensions, constructor state,
  refusal of text-only fallback, and strict reloads of both export paths. The full
  existing oQ8e artifact reloaded strictly and reproduced the same 66-token image
  text response after these changes.
- The second review hardened unidentified-geometry rejection, attention-header
  validation, strict calibration binding, config-loss fallback, head aliases,
  dispatcher installation, and full-attention assumptions. The expanded suite
  passed 864 tests. One existing `test_text_only_uses_vlm_prepare_path` failure was
  reproduced on unchanged fork-main: its expected call omits `images_config=None`.
  Both original BF16 and oQ8e checkpoints passed the new checks and reproduced
  their previous image-text outputs exactly. The rebuilt wheel contains all three
  adapter modules, including header validation.
- The third review added confined shard/index reads, early text-only export
  refusal, preservation of known OCR evidence through config loss, ordinary-model
  fallback regressions, and opaque-wrapper/cached-array-mask coverage. The expanded
  suite passed 892 tests with the same pre-existing VLM test failure described
  above. Both real BF16 and oQ8e checkpoints again reproduced their previous image
  text exactly; the wheel includes the updated validator. The proposed mask-slice
  move and relaxed weight binding were not adopted because they violate the tested
  cache and completeness contracts. Parent initialization remains a documented
  construction cost until native upstream support replaces this carry.

The local validation scripts, source weights, generated checkpoints, reference
arrays, and logs are stored outside the tracked tree under the checkout's sibling
`teleocr-data` directory. This is a compatibility and smoke validation, not a
reproduction of TeleOCR's published document benchmarks.

## Upstream disposition

This is a temporary compatibility carry. It stays local while the pinned mlx-vlm
does not provide TeleOCR's decoder and checkpoint routing; oMLX needs both to serve
the original and converted artifacts without remote model code. The intended home
for the architecture, native loader selection, and math/cache tests is mlx-vlm.
Prepare that change independently of oMLX's oQe integration, then submit it upstream
for a first-class TeleOCR model or an equivalent scoped extension to Qwen2.5-VL.
No upstream PR has been submitted yet.

After upstream support is accepted and an oMLX dependency bump includes it, remove
the local model and loader interception, preserving recognition of already exported
`omlx_adapter: teleocr` artifacts. Keep oMLX's quantization metadata and serving
failure policy here. Until then, every mlx-vlm pin bump must run the adapter,
conversion, and engine-pool regressions plus a strict real-checkpoint image smoke
test. Parent-constructor delegation reduces drift but does not eliminate the need
to validate inherited internal APIs.
