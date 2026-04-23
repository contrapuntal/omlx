"""Repro v2: same as v1 but with mlx-lm make_prompt_cache.

Counterpart to debug_gemma4_oom.py — v1 runs without cache, v2 runs with a
standard sliding-window RotatingKVCache (what a correctly-wired oMLX path
should produce). Both succeed on the upstream-fixed code path and together
form the regression baseline for the gemma-4 OOM fix (see HANDOFF.md).
"""
import sys, time, traceback
import mlx.core as mx
from mlx.utils import tree_flatten

MODEL_PATH = "/Volumes/MacExternalStorage/LMStudio/models/unsloth/gemma-4-26b-a4b-it-MLX-8bit"

def gb(n): return f"{n/(1024**3):6.2f}GB"

def stage(label):
    mx.synchronize()
    peak = mx.get_peak_memory()
    active = mx.get_active_memory()
    t = time.time() - T0
    print(f"[{t:6.2f}s] peak={gb(peak)} active={gb(active)}  |  {label}", flush=True)

T0 = time.time()
stage("start")

from mlx_vlm.utils import load, prepare_inputs
from mlx_lm.models.cache import make_prompt_cache
stage("imports done")

model, processor = load(MODEL_PATH)
stage("load() returned")
mx.synchronize()

try:
    cache = make_prompt_cache(model.language_model)
    print(f"cache: {len(cache)} entries; sample types: {[type(c).__name__ for c in cache[:5]]}", flush=True)
    mx.synchronize()
    stage(f"make_prompt_cache OK ({len(cache)} entries)")
except Exception as e:
    traceback.print_exc(); stage(f"make_prompt_cache FAILED: {e}"); sys.exit(1)

msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
rendered = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
inputs = prepare_inputs(processor, images=None, prompts=[rendered])
input_ids = inputs["input_ids"]
print(f"input_ids.shape = {input_ids.shape}", flush=True)

feats = model.get_input_embeddings(input_ids=input_ids)
mx.synchronize()
stage(f"get_input_embeddings OK  shape={feats.inputs_embeds.shape}")

print("\n=== language_model prefill WITH cache ===", flush=True)
try:
    out = model.language_model(
        inputs=None,
        inputs_embeds=feats.inputs_embeds,
        per_layer_inputs=feats.per_layer_inputs,
        cache=cache,
    )
    stage("language_model returned (graph built)")
    mx.synchronize()
    stage(f"logits materialized  shape={out.logits.shape}")
except Exception as e:
    traceback.print_exc(); stage(f"language_model FAILED: {e}"); sys.exit(1)

print("\n--- cache state post-prefill ---")
for i, c in enumerate(cache[:3]):
    offset = getattr(c, 'offset', None)
    max_size = getattr(c, 'max_size', None)
    shape = getattr(getattr(c, 'keys', None), 'shape', None)
    print(f"  layer {i}: type={type(c).__name__} offset={offset} max_size={max_size} keys.shape={shape}")

print("\nSUCCESS - prefill with cache did not OOM.")
