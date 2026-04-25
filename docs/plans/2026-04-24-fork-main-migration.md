# Fork-main migration plan

**Date:** 2026-04-24
**Status:** In progress (oMLX), Planned (LocalAI)
**Scope:** Two repositories — oMLX, LocalAI — plus companion forks of
external upstreams that those repos pin and carry patches against.

## Context

The global Forked Repo Workflow policy (in `~/.claude/CLAUDE.md` and
`~/.codex/AGENTS.md`) was rewritten on 2026-04-24 from a Hybrid
`dev`-as-integration-branch pattern to an Integration-Manager + carry-queue
pattern with `fork-main` as the runtime branch. Two existing repos still hold
the old shape with mixed-destiny commits (carry patches, PR-bound work, and
experiments all on one `dev` branch). Both need migration to align with the
new policy.

**Affected repos:**
- `/Volumes/MacExternalStorage/proj/omlx`
- `/Volumes/MacExternalStorage/proj/LocalAI`

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
- `fork-main` cherry-picks both. SHA: `362228d8480d319cbaeb9c30aebb97fc628b9365`.
- oMLX `pyproject.toml` audio-extra `mlx-audio` URL re-pinned to
  `git+https://github.com/contrapuntal/mlx-audio@362228d…`.
- Upstream PR filing deferred (canary fix should go upstream; voxtral
  carry can be re-PRd to Blaizzy with the `__dict__` framing).

## Per-repo migration procedure

Run for each repo independently. Estimated 1–2 focused hours per repo,
**plus** Phase 0 time per external upstream that needs forking.

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

## Repo-specific notes

### oMLX
- Uncommitted on `dev` at policy-change time (2026-04-24):
  `packaging/venvstacks.toml`, `pyproject.toml`. Resolve before migrating.
- Phase 0 prerequisite: `mlx-audio` companion fork (see above) — must
  exist before re-pinning `pyproject.toml`.
- `docs/known-issues.md` exists; sweep it for `dev`-branch references.
- Runtime launches from project root; check model caches and venv state when
  switching working tree from `dev` to `fork-main`.

### LocalAI
- Path confirmed: `/Volumes/MacExternalStorage/proj/LocalAI`.
- Audit-pass details TBD on first migration session.

## Rollback

If a repo migration goes wrong:
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
