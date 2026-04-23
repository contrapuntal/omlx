#!/usr/bin/env bash
# Runs the per-model harness once per model (fresh Python process = true
# memory isolation), aggregates JSON lines, and prints a markdown table
# suitable for pasting into the upstream issue / PR body.
#
# Stop the oMLX server first:
#   launchctl unload ~/Library/LaunchAgents/com.omlx.server.plist
# Restart after:
#   launchctl load ~/Library/LaunchAgents/com.omlx.server.plist
#
# Run a subset with arg(s):
#   ./scripts/verify_gemma4_fix_all.sh Gemma-4-26B Gemma-4-31B

set -u
cd "$(dirname "$0")/.."

OUT_DIR="${TMPDIR:-/tmp}/verify_gemma4_$$"
mkdir -p "$OUT_DIR"
trap 'echo "results dir: $OUT_DIR"' EXIT

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("LFM2-VL" "MinerU" "Qwen2.5-VL-7B" "Gemma-4-26B-A4B" "Gemma-4-31B" "Qwen3.6-35B")
fi

for m in "${MODELS[@]}"; do
  safe=$(echo "$m" | tr ' ()/' '_')
  echo "=== running $m (output → $OUT_DIR/$safe.json) ==="
  uv run python scripts/verify_gemma4_fix_empirical.py "$m" \
    > "$OUT_DIR/$safe.json" 2> "$OUT_DIR/$safe.log" || echo "  (exited $?)"
  # Echo the human-readable log to this terminal so progress is visible
  cat "$OUT_DIR/$safe.log"
  echo
done

echo "=== AGGREGATED MARKDOWN TABLE ==="
python3 - "$OUT_DIR" <<'PYEOF2'
import json, sys, glob, os
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    try:
        rows.append(json.loads(open(f).read().strip().splitlines()[-1]))
    except Exception as e:
        print(f"skip {f}: {e}", file=sys.stderr)

if not rows:
    print("(no results)")
    sys.exit()

print()
# Critical metric is matches/lm_key_count, not matches/vlm_key_count:
# mlx-lm's slots must all be bound; extra VLM keys are fine to drop.
print("| Model | VLM keys | mlx-lm keys | Legacy-prefix binds | Fixed prefix | Fix binds | MoE bypass | Peak GB |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    if not r.get("ok"):
        print(f"| {r['label']} | — | — | **LOAD FAILED: {r.get('error','')}** | — | — | — | — |")
        continue
    lm = r['lm_key_count']
    legacy_cell = f"{r['legacy_matches']}/{lm}"
    if r['legacy_matches'] == lm:
        legacy_cell += " ✓"
    elif r['legacy_matches'] == 0:
        legacy_cell = f"**0/{lm}** — all slots uninitialized"
    fix_cell = f"{r['resolved_matches']}/{lm}"
    if r['resolved_matches'] == lm:
        fix_cell += " ✓"
    print(f"| {r['label']} | {r['vlm_key_count']} | {lm} | {legacy_cell} | "
          f"`{r['resolved_prefix']}` | {fix_cell} | "
          f"{'yes' if r['is_complex_moe'] else 'no'} | "
          f"{r['peak_gb_after_load']} |")
PYEOF2
