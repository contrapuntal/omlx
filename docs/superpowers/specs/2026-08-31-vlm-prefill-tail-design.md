# VLM Prefill Tail Clamp Design

## Problem

Under adaptive memory throttling, `Scheduler._adaptive_chunk_size()` applies
the configured minimum chunk floor even when the caller requests a shorter
final tail. With a 32-token floor and a 15-token VLM tail, the method returns
32 despite its documented `<= requested` contract. The external prefill path
then slices 15 token IDs but 16 available embeddings, exposing an off-by-one
reshape failure in Qwen4-Exp PLE.

## Design

Cap the effective minimum chunk at `requested` inside
`_adaptive_chunk_size()`. Normal chunks retain the configured floor; positive
tails shorter than the floor use their actual requested size. No VLM-specific
branch or PLE change is needed because the defect is in the scheduler's
generic return-value invariant.

## Testing

Add a focused regression test using the existing real scheduler method and
memory-probe harness. Under forced pressure with a 32-token floor, every
positive request from 1 through 31 must be returned unchanged. This includes
the observed 15-token failure and protects the full sub-floor boundary.

Run the focused regression first to observe the current failure, then apply
the one-line clamp and run the focused test, the full prefill OOM test module,
and scheduler lint checks.
