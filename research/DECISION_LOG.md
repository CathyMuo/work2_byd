# Decision log

## 2026-10-09 — Independent public development repository

The user authorized creation of `work2_byd`, cloning the related repositories,
and publishing to their GitHub account as a **public** repository. Ongoing
workflow: developer edits/checks/pushes here; user pulls and runs on 4×H800 in
the factory, then returns results. No factory-side push or Codex is required.

The shared discussion defines Work 2 as search-feedback drift and recovery in
multi-turn search agents, independent of GMST. Bootstrap pins Search-R1,
Stable-RAG, Harness-G, and SearchAtlas as public Git submodules. Search-R1
already includes its modified `verl` training code; no separate current veRL
checkout is substituted into this historical stack.

Start at A0 environment/bootstrap. No GPU inference or research experiment was
run during setup. The diagnostic configuration is a design snapshot, not an
implemented experiment runner. RL is deferred until the plan's A/B evidence
and GO decision exist. Credentials and machine-local assets stay untracked.

## 2026-10-09 — Reuse the existing factory environment

The user clarified that the factory's Search-R1 environment is already set up.
Execution environment management stays with the user; the developer supplies
only a recommended configuration checklist. Remove the environment-report
requirement from the workflow. Retain inventory/packaging tools as optional
troubleshooting utilities.

A0 repository bootstrap is complete. Advance to A1 clean-baseline implementation
using existing Search-R1 interfaces with configurable local paths. This update
does not claim successful inference, ready data assets, or a research GO result.

## 2026-10-09 — Runnable A1 and detailed factory execution guide

The user requested actual experiment scripts and a detailed research-driven
execution document, and clarified that models/data are a checklist only.
Implemented local-source pilot selection (100+100, frozen before outcomes),
local vLLM/HF clean rollouts, upstream retrieval launch/adapter, finite search
and context budgets, answer/tool metrics, per-observation raw token prefixes,
candidate/visible documents, real-case export, and optional result packaging
without an environment inventory prerequisite. No resource was downloaded or
added as experimental data.

Current code implements A1 E0 only; no intervention/recovery/RL result is
claimed. CPU tests cover the protocol, gold-label isolation, preservation of
original token IDs, budget boundaries, fact-title false positives, HTTP schema,
selection reproducibility, metrics, and completed/failed run outputs. Factory
model inference is pending. Next: the user runs 10-question smoke then the 200
question clean baseline, and returns metrics/cases for audit before A2.

`docs/EXECUTION_GUIDE.md` separates executable commands from future stages.
The harness uses Search-R1-style tags with an explicit finite budget; its
scores are not presented as an exact reproduction of the official evaluation.

Validation before publication: 12 CPU unittest cases passed; Python compilation,
shell syntax, all four module CLI help entries, and diff whitespace checks passed.
No factory inference or end-to-end H800 validation ran on the developer server.
