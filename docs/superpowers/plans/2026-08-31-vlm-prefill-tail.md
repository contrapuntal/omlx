# VLM Prefill Tail Clamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep adaptive prefill chunks within the caller's requested token count so short VLM tails cannot create token/embedding length mismatches.

**Architecture:** Preserve the generic adaptive-throttle algorithm and cap only its effective minimum floor. Cover the scheduler contract directly with the existing test harness rather than adding model-specific handling.

**Tech Stack:** Python 3.12, pytest, Ruff, oMLX scheduler.

## Global Constraints

- `Scheduler._adaptive_chunk_size()` must return `1 <= result <= requested` for every positive request.
- Preserve the configured minimum chunk floor whenever `requested` is at least that floor.
- Do not add Qwen4-Exp- or VLM-specific scheduler behavior.

---

### Task 1: Clamp Sub-Floor Adaptive Prefill Tails

**Files:**
- Modify: `omlx/scheduler.py:4288`
- Test: `tests/test_prefill_oom_graceful.py`

**Interfaces:**
- Consumes: `Scheduler._adaptive_chunk_size(requested, request_id, loop_label, kv_len=0)` and the existing `_throttle_ctx`/`_call` test harness.
- Produces: The existing documented return contract, including positive requests smaller than `prefill_min_chunk_tokens`.

- [ ] **Step 1: Write the failing regression test**

```python
def test_throttle_never_enlarges_a_tail_below_min_chunk():
    hard = 40 * _GB
    ns = _throttle_ctx(
        current=hard + _GB,
        hard=hard,
        samples_bpt=1_000_000,
        min_chunk=32,
    )
    ns._fake_current = hard + _GB

    for requested in range(1, 32):
        assert _call(ns, requested) == requested
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=. /Volumes/MacExternalStorage/proj/omlx/.venv/bin/python -m pytest tests/test_prefill_oom_graceful.py::test_throttle_never_enlarges_a_tail_below_min_chunk -q`

Expected: FAIL because the current code returns `32` for `requested=1`.

- [ ] **Step 3: Implement the minimal invariant fix**

```python
min_chunk = min(requested, max(1, self._prefill_min_chunk_tokens))
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the focused command from Step 2.

Expected: one passing test.

- [ ] **Step 5: Run regression and lint verification**

Run:

```bash
PYTHONPATH=. /Volumes/MacExternalStorage/proj/omlx/.venv/bin/python -m pytest tests/test_prefill_oom_graceful.py tests/test_scheduler_chunked_prefill.py -q
uvx ruff check --isolated --select E4,E7,E9,F omlx/scheduler.py tests/test_prefill_oom_graceful.py
git diff --check
```

Expected: all selected tests pass, Ruff reports no syntax or Pyflakes errors,
and Git reports no whitespace errors. The repository's broader Ruff profile
has existing findings outside this change and is not part of this focused fix.

- [ ] **Step 6: Commit the logical change**

```bash
git add docs/superpowers/specs/2026-08-31-vlm-prefill-tail-design.md docs/superpowers/plans/2026-08-31-vlm-prefill-tail.md omlx/scheduler.py tests/test_prefill_oom_graceful.py
git commit -m "fix(scheduler): preserve short prefill tails"
```
