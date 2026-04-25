# Fork-main migration plan

**Date:** 2026-04-24
**Status:** Executed 2026-04-24
**Scope:** oMLX, plus the companion fork of `mlx-audio` (an external
upstream that oMLX pins and carries patches against).

## Context

The global Forked Repo Workflow policy (in `~/.claude/CLAUDE.md` and
`~/.codex/AGENTS.md`) was rewritten on 2026-04-24 from a Hybrid
`dev`-as-integration-branch pattern to an Integration-Manager + carry-queue
pattern with `fork-main` as the runtime branch. oMLX still held the old
shape with mixed-destiny commits (carry patches, PR-bound work, and
experiments all on one `dev` branch) and needed migration to align with
the new policy.

## Phase 0 — Companion forks of external upstreams

Some consumer repos (oMLX in particular) hold patches against third-party
packages they pin via git URL — the patches live only inside `.venv` and
disappear on every `uv sync`. Before migrating the consumer to `fork-main`,
fork those upstreams first so the patches can land as carries on a fork
that the consumer pins to.

Procedure for each external upstream:

1. `gh repo fork <upstream-org>/<repo>` under `contrapuntal`. Clone as a
   sibling worktree (`/Volumes/MacExternalStorage/proj/<repo>/`).
2. Configure remotes: `upstream = <upstream-org>/<repo>` (https),
   `origin = contrapuntal/<repo>` (ssh). Verify `main`/`master` is
   fast-forward to `upstream/<default>`.
3. For each in-venv patch:
   - `git checkout -b <descriptive-topic-name> main`
   - Apply the patch as one logical commit with a PR-quality message
     (the message should read like the upstream PR description it could
     become).
   - Push to `origin`.
4. `git checkout -b fork-main main`, cherry-pick each topic branch onto
   it, push to `origin`. Capture the immutable SHA — that is what the
   consumer repo pins.
5. Update the consumer's pin (e.g. `pyproject.toml` git URL) to the
   fork's `fork-main` SHA. Re-sync the consumer's venv; verify the
   in-venv patches are no longer needed.
6. Decide PR disposition per topic branch (open upstream now, hold for
   later, or keep as permanent carry).

### Companion fork: mlx-audio (executed 2026-04-24, for oMLX)

- Forked `Blaizzy/mlx-audio` → `contrapuntal/mlx-audio`.
- Topic branches (off `main`, both at `upstream/main` HEAD `a5867d7`):
  - `voxtral-eos-token-ids-bypass` — fixes Voxtral STT crash on
    `eos_token_ids` initialization (tokenizer base class collapses
    `_id`/`_ids` getattr and rejects non-string setattr; bypass via
    `__dict__`). Re-frames the failed upstream PR #450.
  - `canary-mlx-converted-checkpoint-support` — fixes silent
    half-decoder load on MLX-converted canary checkpoints (alt key
    naming + already-MLX conv layout).
- `fork-main` cherry-picks both. SHA:
  `52115c0c3c19f2d8180acffa3546f355c3d9df1f` (rev. after PR-quality
  revisions across rounds 1-5: canary uses sentinel-key format
  detection rather than a shape heuristic; canary alt-naming test
  coverage tightened (cross-attention linear_k/v, alt layer_norm);
  voxtral extracts a `_ensure_eos_token_ids_list` helper so its tests
  exercise production code; voxtral default-id constant privatized to
  `_VOXTRAL_EOS_TOKEN_IDS`; canary PR body + source comments
  re-framed to credit canary-mlx as the third-party converter
  producing the alt-naming format, replacing earlier fictional
  `mlx-community/canary-1b-v2-mlx-*` URLs).
- oMLX `pyproject.toml` audio-extra `mlx-audio` URL re-pinned to
  `git+https://github.com/contrapuntal/mlx-audio@52115c0…`.
- Voxtral PR filed upstream as
  [Blaizzy/mlx-audio#677](https://github.com/Blaizzy/mlx-audio/pull/677).
  Canary PR draft remains at
  `<mlx-audio-clone>/.git/info/pr-drafts/canary.md`; filing deferred.

## Migration procedure

Estimated 1–2 focused hours, **plus** Phase 0 time per external
upstream that needs forking.

### 1. Pre-flight
- Commit or stash any uncommitted work in the main checkout.
- Confirm runtime baseline: launch the program from the main checkout on
  current `dev`, verify it works.
- `git fetch upstream` and ensure `master`/`main` is fast-forward to
  `upstream/<default>`. If not, fast-forward it first.

### 2. Audit `dev`
- `git log master..dev --oneline` — list all commits since divergence.
- Classify each commit (scratch file, e.g. `.git/info/migration-audit.md`):
  - **Carry** — permanent fork-only customization; must follow upstream forever.
  - **PR-bound** — could or should go upstream eventually.
  - **Experiment** — keep as bookmark; doesn't need to follow upstream.
  - **Drop** — already merged upstream, superseded, or known dead.
- Flag commits that touch mixed concerns and need splitting via
  `git rebase -i` (`edit` action, then `git reset HEAD^ <files>`).

### 3. Build `fork-main`
- `git branch fork-main master` — start from upstream tip.
- Cherry-pick (or interactive-rebase from a temp branch) the carry commits in
  chronological order.
- Resolve any conflicts; these surface real issues in the carry patches.
- Verify carry-patch composition: check out `fork-main`, run the program,
  confirm runtime works as it did on `dev`.

### 4. Spin off topic branches
- For each PR-bound cluster from the audit: `git branch <name> master`,
  cherry-pick the relevant commits onto it. Push to `origin` as backup.
- For each experiment cluster: same pattern, named descriptively.

### 5. Switch the main working tree
- In main checkout: `git checkout fork-main`.
- Re-verify runtime works from main checkout.
- Watch for stale build artifacts, caches, or venv state from the old `dev`
  checkout that could mask regressions.

### 6. Rename and clean up
- `git branch -m dev dev-legacy` (don't delete — safety net).
- `git push origin fork-main` (publish new runtime branch).
- `git push origin dev-legacy` (preserve old shape on remote for ~one
  upstream cycle).
- `git push origin --delete dev` (drop old remote head only after the above
  push succeeds).
- Update default branch on `origin` if any tooling depends on it.
- Grep the repo for `dev`-by-name references in CI configs, scripts, docs,
  per-repo CLAUDE.md; update where appropriate.

### 7. Post-migration sanity
- `git worktree list` shows only the main checkout on `fork-main`.
- Exercise the upstream-sync rhythm once end-to-end:
  ```bash
  git fetch upstream
  git fetch upstream <default>:master
  git rebase master
  git push --force-with-lease origin fork-main
  ```
- Schedule deletion of `dev-legacy` for ~one upstream cycle out.

## oMLX-specific notes

- Uncommitted on `dev` at policy-change time (2026-04-24):
  `packaging/venvstacks.toml`, `pyproject.toml`. Resolved before migrating.
- Phase 0 prerequisite: `mlx-audio` companion fork (see above) — had to
  exist before re-pinning `pyproject.toml`.
- `docs/known-issues.md` exists; was swept for `dev`-branch references
  (none found).
- Runtime launches from project root; checked model caches and venv state
  when switching working tree from `dev` to `fork-main`.

## Rollback

If the migration is found broken later:
- `git checkout dev-legacy` in main checkout — runtime returns to old state.
- Push `dev-legacy` back to `origin` as `dev` if external tooling depends on
  the old name.
- Investigate, retry.

## Convention reference

Source policy: `~/.claude/CLAUDE.md` and `~/.codex/AGENTS.md`,
section "Forked Repo Workflow" (both updated 2026-04-24). Convention basis:
Pro Git Ch. 5.1 "Integration-Manager Workflow" (Chacon/Straub),
`gitworkflows(7)` (Hamano), `git worktree` (git 2.5+), carry-patch
maintenance (Andrew Morton's `-mm` tree, kernel subsystems).
