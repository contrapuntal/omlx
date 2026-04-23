"""Bare mlx-vlm repro for gemma-4-26b-a4b-it-MLX-8bit OOM on "hi"."""
import sys, time, traceback
import mlx.core as mx
from mlx.utils import tree_flatten

MODEL_PATH = "/Volumes/MacExternalStorage/LMStudio/models/unsloth/gemma-4-26b-a4b-it-MLX-8bit"

def gb(n): return f"{n/(1024**3):6.2f}GB"

def stage(label):
    mx.synchronize()
    peak = mx.metal.get_peak_memory()
    active = mx.metal.get_active_memory()
    t = time.time() - T0
    print(f"[{t:6.2f}s] peak={gb(peak)} active={gb(active)}  |  {label}", flush=True)

T0 = time.time()
stage("start")

from mlx_vlm.utils import load, prepare_inputs
stage("imports done")

model, processor = load(MODEL_PATH)
stage("load() returned")

# Touch and sync all params
leaves = [v for _, v in tree_flatten(model.parameters()) if isinstance(v, mx.array)]
mx.synchronize()
stage(f"weights reachable ({len(leaves)} tensors)")

msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
rendered = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
print(f"\n--- rendered prompt ({len(rendered)} chars) ---")
print(rendered[:500])
print("--- end ---\n", flush=True)
stage("chat template applied")

inputs = prepare_inputs(processor, images=None, prompts=[rendered])
input_ids = inputs["input_ids"]
print(f"input_ids.shape = {input_ids.shape}", flush=True)
stage("prepare_inputs done")

# Stage A: get_input_embeddings
try:
    feats = model.get_input_embeddings(input_ids=input_ids)
    mx.synchronize()
    stage(f"get_input_embeddings OK  inputs_embeds.shape={feats.inputs_embeds.shape}")
except Exception as e:
    traceback.print_exc()
    stage(f"get_input_embeddings FAILED: {e}")
    sys.exit(1)

# Stage B: language_model prefill
print("\n=== calling language_model forward (prefill) ===", flush=True)
try:
    out = model.language_model(
        inputs=None,
        inputs_embeds=feats.inputs_embeds,
        per_layer_inputs=feats.per_layer_inputs,
        cache=None,
    )
    stage("language_model call returned (graph built, not materialized)")
    mx.synchronize()
    stage(f"logits materialized OK  shape={out.logits.shape}  dtype={out.logits.dtype}")
except Exception as e:
    traceback.print_exc()
    stage(f"language_model FAILED: {e}")
    sys.exit(1)

print("\nSUCCESS - no OOM in bare mlx-vlm prefill.")
