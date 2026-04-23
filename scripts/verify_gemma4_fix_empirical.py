"""Empirical verification harness for the VLM decode-model weight-sharing fix.

Runs ONE model per invocation so each model gets a fresh Python process
(guarantees true memory isolation — no cumulative leakage across models).
Prints both human-readable output and a single JSON line suitable for
aggregation (`| jq -s`) across multiple invocations.

Usage:
    uv run python scripts/verify_gemma4_fix_empirical.py LIST
    uv run python scripts/verify_gemma4_fix_empirical.py <label-substring>

Example — run each model in its own process, aggregate the JSON lines:
    for m in "LFM2-VL" "MinerU" "Qwen2.5-VL-7B" "Gemma-4-26B-A4B" \
             "Gemma-4-31B" "Qwen3.6-35B"; do
      uv run python scripts/verify_gemma4_fix_empirical.py "$m" > /tmp/$m.json
    done
    cat /tmp/*.json | jq -s . > /tmp/verify_all.json

See ``scripts/verify_gemma4_fix_all.sh`` for the combined runner.

What each run measures:
  1. VLM language_model key count (what we're sharing from the VLM side)
  2. mlx-lm decode model key count (where weights should land)
  3. Matches under legacy hardcoded 'language_model.' prefix (47df15a)
  4. Matches under _resolve_weight_share_prefix (the fix)
  5. Whether _detect_complex_backbone flags it for MoE bypass
  6. Peak + active GPU memory after VLM load (T1)
"""
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model
from mlx_vlm.utils import load as vlm_load

from omlx.engine.vlm import _resolve_weight_share_prefix, _vlm_aware_get_classes
from omlx.models.vlm import VLMModelAdapter


MODELS = [
    ("LFM2-VL-1.6B",
     "/Volumes/MacExternalStorage/LMStudio/models/mlx-community/LFM2-VL-1.6B-bf16",
     "lfm2", 3),
    ("MinerU2.5-1.2B",
     "/Volumes/MacExternalStorage/LMStudio/models/mlx-community/MinerU2.5-2509-1.2B-bf16",
     "mineru", 2),
    ("Qwen2.5-VL-7B",
     "/Volumes/MacExternalStorage/LMStudio/models/mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
     "qwen2.5-vl", 9),
    ("Gemma-4-26B-A4B (MoE)",
     "/Volumes/MacExternalStorage/LMStudio/models/unsloth/gemma-4-26b-a4b-it-MLX-8bit",
     "gemma-4", 27),
    ("Gemma-4-31B (dense)",
     "/Volumes/MacExternalStorage/LMStudio/models/unsloth/gemma-4-31b-it-MLX-8bit",
     "gemma-4", 33),
    ("Qwen3.6-35B-A3B (MoE, mRoPE)",
     "/Volumes/MacExternalStorage/LMStudio/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit",
     "qwen-moe", 37),
]

LEGACY_PREFIX = "language_model."  # what 47df15a hardcoded


def gb(n):
    return round(n / (1024**3), 2)


def preflight_memory():
    """Return dict of current memory state."""
    out = {"wired_limit_gb": None, "free_gb": None, "server_running": False}
    try:
        r = subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True)
        out["wired_limit_gb"] = round(int(r.strip()) / 1024, 1)
    except Exception:
        pass
    try:
        vm = subprocess.check_output(["vm_stat"], text=True)
        free_pages = int([l for l in vm.splitlines() if "Pages free" in l][0].split()[-1].rstrip("."))
        out["free_gb"] = round(free_pages * 16384 / (1024**3), 1)
    except Exception:
        pass
    try:
        subprocess.check_output(["pgrep", "-f", "omlx serve"], text=True, timeout=2)
        out["server_running"] = True
    except Exception:
        pass
    return out


def analyze_one(label, path, family, approx_gb):
    result = {
        "label": label, "path": path, "family": family, "approx_gb": approx_gb,
        "ok": False, "error": None,
        "vlm_key_count": 0, "lm_key_count": 0, "legacy_matches": None,
        "resolved_prefix": None, "resolved_matches": None, "legacy_correct": None,
        "is_complex_moe": None,
        "peak_gb_after_load": None, "active_gb_after_load": None,
        "vlm_load_s": None, "lm_build_s": None,
    }

    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass

    try:
        t0 = time.time()
        vlm_model, processor = vlm_load(path)
        mx.synchronize()
        result["vlm_load_s"] = round(time.time() - t0, 1)
        result["peak_gb_after_load"] = gb(mx.get_peak_memory())
        result["active_gb_after_load"] = gb(mx.get_active_memory())

        vlm_params = dict(tree_flatten(vlm_model.language_model.parameters()))
        vlm_keys = list(vlm_params.keys())
        result["vlm_key_count"] = len(vlm_keys)

        t1 = time.time()
        lm_model, _ = load_model(
            Path(path), lazy=True, strict=False,
            get_model_classes=_vlm_aware_get_classes,
        )
        result["lm_build_s"] = round(time.time() - t1, 1)
        lm_keys_set = set(dict(tree_flatten(lm_model.parameters())).keys())
        result["lm_key_count"] = len(lm_keys_set)

        result["legacy_matches"] = sum(1 for k in vlm_keys if (LEGACY_PREFIX + k) in lm_keys_set)
        resolved = _resolve_weight_share_prefix(vlm_keys, lm_keys_set)
        result["resolved_prefix"] = resolved
        result["resolved_matches"] = sum(1 for k in vlm_keys if (resolved + k) in lm_keys_set)
        result["legacy_correct"] = result["legacy_matches"] == result["vlm_key_count"]
        result["is_complex_moe"] = VLMModelAdapter._detect_complex_backbone(vlm_model)
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)
    return result


def print_human(result, preflight):
    print(f"Preflight: wired_limit={preflight['wired_limit_gb']}GB  "
          f"free_before_run={preflight['free_gb']}GB  "
          f"server_running={preflight['server_running']}", file=sys.stderr)
    if preflight["server_running"]:
        print("  ⚠️  oMLX server is running — may compete for memory with this harness",
              file=sys.stderr)
    print(file=sys.stderr)

    r = result
    if not r["ok"]:
        print(f"✗ {r['label']}  FAILED: {r['error']}", file=sys.stderr)
        return

    # Critical metric: how many of mlx-lm's parameter slots get bound?
    # (not how many VLM keys match — the VLM may expose extra keys that are
    # fine to drop). An unfilled mlx-lm slot stays lazy/uninitialized.
    lm_count = r["lm_key_count"]
    legacy_cover = f"{r['legacy_matches']}/{lm_count}"
    fixed_cover = f"{r['resolved_matches']}/{lm_count}"
    legacy_ok = r["legacy_matches"] == lm_count
    fixed_ok = r["resolved_matches"] == lm_count

    print(f"{'─' * 78}", file=sys.stderr)
    print(f"  {r['label']}  (~{r['approx_gb']} GB)", file=sys.stderr)
    print(f"{'─' * 78}", file=sys.stderr)
    print(f"  VLM param keys      = {r['vlm_key_count']}", file=sys.stderr)
    print(f"  mlx-lm param keys   = {r['lm_key_count']}  (number of slots that must be filled)", file=sys.stderr)
    print(f"  legacy '{LEGACY_PREFIX}' prefix binds: {legacy_cover} mlx-lm slots "
          f"{'✓' if legacy_ok else '✗ — ALL SLOTS LEFT UNINITIALIZED' if r['legacy_matches'] == 0 else '✗ — partial'}", file=sys.stderr)
    print(f"  resolved prefix     = {r['resolved_prefix']!r}", file=sys.stderr)
    print(f"  fixed prefix binds:  {fixed_cover} mlx-lm slots "
          f"{'✓' if fixed_ok else '✗ — STILL UNFILLED'}", file=sys.stderr)
    print(f"  complex-MoE bypass  = {r['is_complex_moe']}", file=sys.stderr)
    print(f"  peak GPU after load  = {r['peak_gb_after_load']} GB", file=sys.stderr)
    print(f"  active GPU after load = {r['active_gb_after_load']} GB", file=sys.stderr)
    print(f"  timing              = vlm_load {r['vlm_load_s']}s  lm_build {r['lm_build_s']}s", file=sys.stderr)
    print(file=sys.stderr)


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    arg = sys.argv[1]
    if arg.upper() == "LIST":
        for m in MODELS:
            print(f"{m[0]:30s}  ~{m[3]:3d} GB   {m[1]}")
        return

    # Fuzzy match on label
    matches = [m for m in MODELS if arg.lower() in m[0].lower()]
    if not matches:
        print(f"No model label matching {arg!r}. Try LIST to see available models.", file=sys.stderr)
        sys.exit(2)
    if len(matches) > 1:
        print(f"Ambiguous label {arg!r}; matches:", file=sys.stderr)
        for m in matches:
            print(f"  {m[0]}", file=sys.stderr)
        sys.exit(2)

    label, path, family, approx_gb = matches[0]
    pre = preflight_memory()
    result = analyze_one(label, path, family, approx_gb)
    result["preflight"] = pre

    # Human-readable → stderr; machine-readable JSON → stdout
    print_human(result, pre)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
