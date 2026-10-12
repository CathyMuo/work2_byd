# Work 2 experiment plan — bootstrap snapshot

Date: 2026-10-09. Source: the user's shared discussion,
[工作二方向收敛](https://chatgpt.com/share/6ac84bc4-3cd4-83ea-8a65-ec7b03c325c5),
specifically its final “正式实验方案 V1.0” and independence constraint.
This is an executable-project summary of that discussion, not a new literature
review or a claim that its novelty has been established.

## Research question and independence

Study whether a single changed retrieval observation induces later search
off-target behavior, and whether a search agent can recover under a finite
budget. Working title: *Search Feedback Drift and Recovery in Multi-Turn Search
Agents*. Work 2 must stand alone: no GMST method, training, experimental output,
or outcome-selected artifacts are imported. Existing stopped projects remain
stopped. This repository is the independent root explicitly requested by the user.

## Stage A0 — repository and environment (completed bootstrap)

1. Initialize the public repository and pin the four discussed upstreams:
   Search-R1, Stable-RAG, Harness-G, SearchAtlas.
2. Provide public-clone/update instructions, local asset configuration, and a
   standard-library environment/feedback workflow for the factory's 4×H800.
3. The user confirmed that the factory already has a configured Search-R1
   environment. Reuse it; environment management belongs to the user.
4. Supply a recommended configuration list. Environment reports and setup are
   not prerequisites for implementing experiment code. Keep asset paths
   configurable and verify their validity when actually executing experiments.

Exit condition: the public repository and dependencies are pinned, and the user
confirms an existing Search-R1 execution environment. This bootstrap condition
is satisfied. Model/corpus/index readiness has not been empirically verified.

## Stage A1 — clean baseline (factory completion reported)

- Original proposal: trained Qwen2.5-3B and HNSW. The realized A1 used the
  official Search-R1 **7B** PPO checkpoint (reported `713cbe32`) and **Flat**.
  Keep this cohort for A2; do not relabel it as 3B/HNSW or mix new-resource runs.
- Freeze E5-base-v2, Wiki-18, the realized FAISS Flat index, Top-3 visible passages, maximum 4
  search turns, 8192 context tokens, 512 generated tokens per turn initially.
- Verify tool-call validity, multi-turn frequency, EM/F1, context truncation,
  search counts, and termination. A lack of usable multi-turn traces must be
  diagnosed before interpreting perturbation effects.
- Diagnose corpus-missing facts, retriever misses, and agent evidence misuse
  separately. Title equality alone cannot establish supporting-fact recall.

Required output: `baseline_metrics.json`, clean trajectories, and immutable run
metadata. User-reported completion and analysis are recorded in
`A1_REPORTED_RESULTS.md`; raw artifacts are not available on the developer server.

Implementation and factory commands are documented in `docs/EXECUTION_GUIDE.md`.
Use `work2.prepare_pilot`, `work2.launch_retriever` (or an existing service),
and `work2.run_baseline`. Run a 10-question smoke subset before all 200 frozen
pilot questions. Dependencies/resources are user-managed; no model or dataset
download is performed. A1 emits E0 only. A2/A3 code and their scientific gates
are not passed by this implementation update.

## Stage A2 — paired replay sanity (factory engineering completion reported)

- Save the exact question, full token/prompt prefix, and emitted search query
  immediately before injecting the target retrieval observation.
- Fork continuations from that identical prefix. Re-running a question twice
  from the beginning is not a causal pair.
- Use 10 questions initially; compare E0/E0 to measure service nondeterminism
  and E0/E1 to verify observation substitution.
- Start with temperature 0 and fixed seeds. Temperature 0 is not a guarantee
  of bitwise reproducibility; retain the clean replay control.
- Modify exactly one specified observation. All later calls use the frozen
  normal retriever. Do not mutate the FAISS index or corpus for interventions.

Required output: `replay_sanity.jsonl` with prefix identity and intervention
audit metadata. Advance only after these invariants are checked.

Run `work2.run_replay` / `scripts/run_a2.sh` as documented in
`docs/A2_EXECUTION_GUIDE.md`. Use the fixed first five A1 IDs per dataset,
then condition on target-turn reachability. Restore raw generated tokens and
spent global budget. Verify source/model/tokenizer/resource identity and live
same-query visible retrieval; do not hide drift with response caching. A
nonidentical E0/E0 stops permutation and requires engineering diagnosis, not
a scientific NO-GO. No significance/effect threshold is inferred from A1.
The user reports 20/20 exact controls and unchanged retrieval; the engineering
decision is recorded in `A2_REPORTED_RESULTS.md` and the decision log. Raw A2
artifacts remain on the factory server and are revalidated there before A3.

## Stage A3 — diagnostic pilot

2026-10-12 C/D update: factory reports full 175-event body-overlap screening,
but 60% lexical overlap is not semantic certification, and mechanism memberships
sum to 141 rather than 134 exclusive states. Current substage is receipt validation/
calibration and small mechanism-protocol E freezing, documented in
`docs/CD_REVIEW_AND_MECHANISM_PROTOCOL.md`. Keep not-GO. Audit letters C/D/E/F
are distinct from research Stage C (RL). Mechanism development E may reuse existing
verified cases; independent confirmation F alone needs unseen questions. Do not
use gold-support overlap or future trajectory evidence as a deployable trigger.
72 evidence-use candidates are not automatically the nonoracle witnessed-recovery
subset; same prefix/remaining-budget links are required.

Earlier recorded substage (2026-10-12): **factory-scale raw and semantic mechanism audit**.
The factory reports A1 1000 questions, A3 2000 planned/1972 reachable states and
B recovery experiments on an unpublished fork. The user authorizes public main
as the code authority; preserve factory commits and raw artifacts locally.
Reported evidence is reviewed in `FACTORY_SCALE_REVIEW.md`; this is not a full
code import, independent confirmation or acceptance of the scientific gates.
Run CPU `audit_pairs` on original traces, independent evidence sufficiency review
on ALL degraded condition events, and compact B transition audit before new GPU
experiments. Query differences are not semantic drift, entity mentions are not
sufficient evidence, and baseline failures are not irrecoverability proofs.
No scale rerun or RL is the next action. The original engineering implementation
and stricter eligibility protocol below remain historical public capabilities.

Previous substage: **reviewed 10-question engineering pilot**, implemented in
`prepare_interventions` and `run_interventions`; see `docs/A3_EXECUTION_GUIDE.md`.
Keep first five A1 IDs per dataset and turns 1/2. Independently review all
condition eligibilities and freeze real rank-4–20 replacements before inference.
Exact controls and all invariants are required again. This initial implementation
uses conservative lexical screens plus semantic reviewer attestations, a fixed
0.5–2.0 replacement-character ratio, condition-matched E1 descriptive metrics,
and explicit skip/partial denominators. It is not the full diagnostic protocol:
semantic OffTarget/Recovery, question-clustered intervals, 200-question expansion
and independent confirmation still require review and implementation. Semantic
truth is not established merely by checking a review field. No RL GO is implied.

The old A2 Flat-to-hnsw label bug is corrected only in new derivative provenance;
historical A1/A2 files remain immutable. The transient source F1 error has an
unknown cause; keep exact equality and show both values/types on future failures.

HotpotQA 100 and 2WikiMultiHopQA 100 development questions. Freeze sample IDs
and selection before observing outcomes. Future confirmation: 300+300, and
optionally MuSiQue 200. Keep pilot, confirmation, training, and final test
questions separate; track overlap at question and dataset ID level.

| Condition | Single-observation change | Eligibility |
| --- | --- | --- |
| E0 clean | Original Top-3 | Every valid target prefix |
| E1 permutation | Reorder the same three passages | Same document multiset |
| E2 distractor | Replace one noncritical passage with a real relevant distractor | Supporting-fact validation |
| E3 bridge removal | Replace an identifiable bridge passage with a non-answer passage | Verified bridge present initially |
| E4 redundancy | Replace one noncritical passage with redundant evidence | Critical facts preserved |

Replacement candidates come from the same query's ranks 4–20 where possible;
control length and preserve real factual content. Ineligible cases are recorded
as skipped, not silently relabeled or counted as successful recovery.

Intervene at turn 1 and turn 2 separately. Turn-2 analyses include only agents
that actually reach a second valid search; report conditional denominators.

Report EM/F1, clean-correct to perturbed-wrong degradation and reverse
improvement, OffTarget@1, Recovery@1/2/3, calls, tokens, redundant calls, and
budget exhaustion. Query string differences alone do not establish drift.
Supporting-fact/entity weak labels require manual audit; the studied agent
must not be its own sole target judge. Recovery permits alternate evidence
supporting the same fact, not just the original document ID.

Retain paired confidence intervals and case audits, not just point estimates.
Required outputs: `paired_rollouts.jsonl`, aggregate metrics, and failure audit.

## Stage B — recovery baselines and gate

The factory reports B1/B2/B3 and an E3-restricted B5 oracle on the exploratory
scale cohort; its source implementation is unavailable and not claimed imported.
Audit same-start-state/global-budget/cost/gold isolation and actual B0 condition
assignment. B3 reportedly repairs 44 degraded states while harming 53 stable
states; full-cohort net benefit is unknown without the clean-wrong transitions.
The union 52/134 witnesses includes oracle and defines a subset, not universal
recoverability. Current gate review in STATE records semantic gate 2 and the
condition-matched gate 1 as unestablished; original gate definitions stay intact.

Compare original Search-R1, budget 4→6, dedup/filtering, a missing-subgoal review
prompt, Harness-G where feasible, and an explicitly labeled oracle diagnostic
upper bound. Keep corpus/questions and budgets comparable; graph-tool calls
are not automatically equivalent to ordinary search calls. Report tokens,
retrieval access, and evidence alongside call counts. Oracle information must
never enter a deployable method's test input.

All four conditions from the final shared plan must receive evidence:

1. E2/E3 cause repeatable degradation or recovery failure beyond E1 ordering.
2. Later search behavior differs, beyond mere final-answer fluctuation.
3. Simple B1–B3 baselines do not eliminate the main gap.
4. Useful recovery actions remain possible from the perturbed state.

These are research decisions, not numerical significance thresholds. Decide
using paired uncertainty, audited cases, and separate confirmation evidence.
Record an explicit GO or NO-GO before advancing. Do not invent numeric gates
after seeing results.

## Stage C — deferred RL

RAPO is only an internal candidate name. After GO, investigate paired-trajectory
recovery action credit assignment with a frozen retriever. Compare standard
GRPO, matched perturbation augmentation, recovery reward, paired credit, and
ablations. Match samples, rollout counts, tokens, and tool budgets; evaluate
both clean and perturbed environments. Supporting facts may supervise training
but must not be exposed to test policies. Begin at 3B; 7B is a later scale check.
Check novelty against adjacent work before claiming an original algorithm.

No formal RL training is authorized by passing repository initialization.

## Execution and provenance

Developer checks and pushes on this server; user pulls and runs on the factory
4×H800, then returns reports. Stage A need not occupy all four GPUs. Final
parallelism and memory settings are configurable for the user's existing setup.
Record main/dependency commits, checkpoint identity, tokenizer/prompt version,
retriever/index/corpus identity, decoding settings, config, sample manifests,
prefixes, original/visible documents, and per-run outputs. Never publish
credentials or factory-local data by default.


## Current audit task E — post-observation engineering, not Stage C RL

Factory step-0 partial calibration (8/175 reviewed, four sufficient HotpotQA
cases) is accepted for preparation only. Implemented public protocol and commands
are in `docs/E_EXECUTION_GUIDE.md`. Use existing development sources; independently
verify both failed and stable members at their displayed actual prefix. One
complete reviewed pair suffices for smoke; at most four pairs initially. Freeze
run/skip decisions and stable candidate order before inference. Preserve pending
semantic coverage and nonentity uncertainty; no population prevalence claim.

Start after max(parent intervention turn, earliest sufficient turn), retaining
actual perturbed history/raw generated token IDs and all spent budget. Do not
start before a later parent intervention. All selected original controls must
match each other and the historical suffix before any review arms. Compare
original, general review and original-question-constraint review with inherited
4-search/global-context caps and 512-token PER SEGMENT cap, reporting actual
incremental costs and stable harms. No historical compression, gold-input,
post-outcome reselection, new F evaluation, or RL is included. E matching is a
cross-question engineering control; matched E1 for content-specific inference
remains same-state/pre-intervention, not rematching on downstream mediators.

After E engineering and cost review, separately freeze F unseen question IDs,
independent confirmation and deployable prefix-only selection. Existing research
gates are unchanged and remain unestablished; passing E controls is not GO.
