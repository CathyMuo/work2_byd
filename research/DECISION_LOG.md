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

## 2026-10-09 — Accept reported A1 engineering completion; advance to A2

The user returned a factory A1 completion report at source commit `aed1adc`.
Reported counts are internally consistent: 67/200 EM-correct, 195/200 multi-turn,
586 searches (2.93 mean), 189 answered, 10 budget-exhausted, one generation-limit,
zero infrastructure errors, zero context-limit terminations. Raw feedback and
trajectories are not available on this development server; record the figures
as user-reported, not independently artifact-verified.

The realized cohort is the official **7B** Search-R1 PPO checkpoint at reported
revision `713cbe32`, E5-base-v2, Wiki-18, and **FAISS Flat**, with vLLM 0.6.3/BF16,
temperature 0, seed 42, TP=1. Keep this cohort for A2, without changing model,
index, questions, or budgets. The original 3B/HNSW design remains historical;
it must not be used to label these results.

Accept A1 tool/multi-turn/prefix readiness from the report and implement A2.
Select the first five frozen question IDs per dataset before turn eligibility;
include all outcomes, and report turn-2 conditional denominators and skips.
Run two clean continuations and one permutation from each identical token
prefix, with unchanged later normal retrieval and explicit drift auditing.
Do not interpret the lexical-hit EM association as causal evidence or enter
E2–E4/B/C before the relevant gates. No numeric effect threshold is invented.

Factory adaptations (reported): corrupt checkpoint index JSON repaired from
the official source while preserving weights; Hotpot HF dict-of-arrays converted
to original pairs without changing content; 2Wiki source matched schema. Preserve
the existing frozen pilot rather than rerunning selection/conversion for A2.
No model/data download or factory environment provisioning is requested.
The interrupted 7B GRPO training is separate from this diagnostic cohort and
is not restarted by this update; any recovery remains factory-user managed.

## 2026-10-09 — A2 implementation and factory handoff

Implemented A1 raw-token prefix validation/reconstruction, current model and
resource identity checks, live retriever drift audit without caching, two clean
continuations followed by one Top-3 cyclic permutation, unchanged later
retrieval, and preservation of the spent global budget. Controls compare token
IDs/text, queries, observations, status and answer. Same-answer/different-token
controls fail and stop before E1. Source historical continuation agreement is
reported separately; it is not silently used as the fresh control. Unreachable
turns, identical permutations, partial runs and conditional denominators remain
explicit. Query text equality is never called semantic OffTarget or Recovery.

The new default design labels the realized 7B/Flat cohort. A2 inherits the old
A1 manifest's numerical configuration without modifying that source artifact.
A1/A2 share the same continuation implementation; all A1 regression tests pass.
Validation: 24 CPU tests pass, including true-token restoration, one-observation
change, budget accounting, service drift, resource identity, failed controls
and preservation of planned denominators on partial runs. No A2 factory GPU
inference has run here. Current stage remains A2, awaiting the real handoff
metrics/audit/cases; A3/B/C and the unrelated stopped training are not executed.

## 2026-10-09 — Accept reported A2 engineering evidence; implement reviewed A3 pilot

The user returned factory A1+A2 results from `a17f1f3`, with 20/20 exact
E0/E0 controls, 20/20 historical agreements, 20 valid single-observation
permutations and zero drift in 150 live retrieval calls (133 with an existing
reference). Raw artifacts remain on the factory server; record these as reported
evidence, not independently recalculated developer results. Accept the
engineering prerequisites for E2–E4 implementation; no scientific/RL GO.
The descriptive 0.40→0.25 EM, and the turn-1 difference, do not establish
significance, semantic off-target search or recovery from 10 correlated questions.

Fix the legacy parser interpreting `Flat ...; NOT HNSW` as hnsw, reject ambiguous
positive declarations, and prefer `flat`/`hnsw` enums for new configuration.
Do not edit A1/A2 historical manifests or rewrite local descriptions to bypass
validation. New A3 provenance binds original file hashes and includes an additive
index-label erratum for this known old commit; this normalizes a declaration,
not a binary FAISS inspection. Keep the exact F1 comparison; include both repr
values and types in errors. The transient factory failure's cause is unresolved.

Current stage: reviewed **10-question A3 engineering pilot**, first five A1
questions per dataset, turns 1/2. Offline preparation rechecks real local A2
controls, source hashes, metrics and retrieval audit. Freeze independent review
decisions, candidate ranks/slots and evidence before observing new perturbation
outcomes. Approvals require semantic reviewer attestations plus conservative
supporting-sentence screens; automatic lexical hits never prove bridge roles or
noncriticality. E3 excludes bridge facts already present in prior observations.
Use original rank-4–20 real passages, exactly one replacement and a preregistered
engineering character ratio 0.5–2.0; preserve content and record token/cost deltas.
Do not imply token matching or elimination of length confounding.

Fresh E0/E0 and historical agreement must pass before E1/E2–E4; later retrieval
is live and normal, raw prefix and spent global budget are retained. Freeze the
review packet with its SHA256 and preserve partial/failed cases, planned
denominators and skips. Compare each eligible condition with E1 on the same
subset. Report next-query availability/string changes as descriptive only.
Semantic OffTarget/Recovery remain null, and question-clustered paired intervals
are deferred to the full diagnostic protocol. No automatic 200-question expansion,
training restart, baseline-method selection or RL execution is performed.

Validation before publication: **34 CPU tests passed**, including all original
A1/A2 tests, legacy negated-label parsing, exact F1 debug output, immutable
source/review binding, failed A2-gate rejection, semantic-review requirements,
prior-history bridge exclusion, collateral-fact/length/redundancy checks,
single-slot change, budget preservation, control-failure stop, condition-matched
E1 metrics and full mock CLI output. Python compilation, CLI help, shell syntax,
JSON, documentation links and diff checks passed. Factory H800 A3 execution is
pending; model/data/environment remain user-managed.
