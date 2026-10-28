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

## 2026-10-12 — Public main is authoritative; audit reported factory scale results

The user supplied a 2026-10-10 report from factory fork `f4a4bb5`, based on
`a17f1f3`, claiming 17 unpublished commits/35 changed files, A1 1000 questions,
A3 2000 states (1972 completed/28 unreachable), B baselines, 134 degraded states
and L3/recoverability proxy audits. Factory source and raw traces are unavailable
here. Public HEAD is `c4d9521`; no full fork import or factory-test verification
is claimed. The user explicitly answered that no code is available and public
developer main should be primary, with factory code replaced from it. Preserve
the old HEAD in a local backup branch before switching; preserve all outputs
and local configuration. Later ordinary pulls follow the public mainline.

Accept the reported runs as **legacy exploratory progress**, not as completed
semantic gates or independent confirmation. Continue CPU raw coverage/paired
integrity/score audits, question-clustered effects against E1 on the SAME eligible
states, and independent evidence sufficiency review of ALL degraded condition
events. Retain original gate definitions. E3's −10/178 signal does not establish
beyond-ordering degradation without the matched E1 contrast and semantic bridge
qualification; E2's aggregate gain does not establish equivalence. ~60% query
string change does not establish semantic off-target search, so gate 2 remains
unestablished. No RL GO or further scale rerun.

74/134=55.2% are answer-entity/no-answer L3 candidates. 74/92=80.4% is conditional
on proxy judgment coverage after excluding 42 states; not an all-state semantic
prevalence estimate. Review nonentity/alias cases and split the 14 empty answers
from 60 wrong answers. Resolve the different S1 133/134 evidence proxy instead
of asserting universal corpus accessibility. The 82 not repaired by tested
baselines are not proven irrecoverable; 52 witnessed repairs include restricted
oracle. A recovery ratio of 1 on a selected B0-failed witnessed subset is
constructional, not a new result. B3's 44 gains and 53 stable harms imply −9 on
those 583 states if B0 definitions agree; full-cohort net is unknown without all
1389 clean-wrong transitions. Check budget, prompts, gold isolation, action
starting points, oracle and outcome-selected condition assignment before method
claims. Mechanism/novelty hypotheses remain exploratory pending frozen controls
and disjoint confirmation; existing related papers are linked in the review.

Implementation: reconstruct the supplied `replay_intervention` signature as a
compatibility wrapper over the public shared continuation, add a local-branch
backup alignment script and tmp ignore, strict CPU raw paired audit with explicit
schema mappings, whole-question/dataset-stratified bootstrap and condition-matched
E1 contrast, pending semantic-review packets and compact B gain/harm/net audit.
Unknown schemas fail rather than invent missing controls or source fields. No
automatic semantic certification, old-score rewriting or GPU execution. Detailed
protocol and factory migration steps are in FACTORY_ALIGNMENT_AND_NEXT_STEPS.md.

Validation before publication: **46 CPU tests pass**, covering all public A1/A2/A3
regressions plus the compatibility API, duplicate/missing coverage, false
unreachability, raw control/score/history corruption, condition-matched E1,
question-cluster resampling, sparse-stratum interval abstention, B gain/harm and
partial coverage, source-file immutability/schema-only probing, preserved factory
backup commits/ignored results and refusal to discard uncommitted edits. Python
compilation, CLI help, shell syntax, JSON, documentation links and diff checks
passed. No actual factory raw audit or H800 inference was executed here; the
reported factory 77-test suite remains separately unverified.

## 2026-10-12 — Matched-E1 correction reported; content-specific claim remains unestablished

The user reports unpublished factory commit `839431d` found matched E3 excess
“+3, CI crossing zero.” The unit, interval bounds, matched question count and
raw output were not supplied. Record the literal finding rather than inventing
3 percentage points or an exact interval. It weakens the earlier unmatched
content-specific degradation story; it does not establish equivalence or prove
that no state can fail. Preserve original gate 1 and the not-GO decision. The
latest E1/E2 evidence-use candidate memberships (38/39) alone do not establish
weak association without exposure denominators and question/state dependence.

## 2026-10-12 — Reported C/D screening; separate semantic receipts and recovery witnesses

The user reports factory commit `113ca69`: all 175 degradation condition events
across 134 states screened; event labels 110/21/44 sufficient/insufficient/uncertain,
state labels 86/32/16; 72 evidence-use states; 32 empty-answer events split into
16 budget, 10 termination and 6 nonentity-answer cases. The Jonker example was
reported with 100% support-text overlap; `2wiki:75a44681` t1 was corrected to
budget exhaustion. These are supplied results, not locally recalculated raw
receipts. The user says code should follow the public implementation, without
providing factory source or clarification of counts/units.

Accept full-event coverage and terminal-status separation as reported progress,
but >=60% support-sentence body overlap remains a lexical proxy. Removing the
title reduces false positives; missing negation/date/role/comparison constraints
can still change the answer. Two manual examples do not validate full semantic
precision. The mechanism counts 72+29+15+12+9+4=141 do not form an exclusive
134-state partition; multilabel or cross-condition overlap remains unresolved.
Answer type is a separate axis from the reason for an empty terminal answer.
Condition memberships total 91 across 72 states and may overlap. Do not relabel
the 72 as true semantic L3 or as 72 witnessed-recoverable states. Link independently
confirmed evidence-use IDs with nonoracle recovery witnesses at the same prefix
and remaining budget; the earlier 52-witness union does not establish that link.

Public `semantic_review.py` independently implements versioned body-only token
recall diagnostics WITHOUT assigning semantic decisions, frozen-context/body-quote
review validation, explicit pending calibration coverage, and event-level primary
versus secondary mechanism receipts checked against original A3 status/content
and file hash. A state with different condition-specific primary labels becomes
an explicit mixed state; evidence-use requires actual post-evidence generation
and a nonempty wrong answer. Reviewer constraint/derivation attestations are not
automatic entailment proofs. Gold supporting facts are offline audit information,
never deployable trigger features. Do not claim reproduction of unavailable code.

Current next action: validate/calibrate C/D receipts and freeze small mechanism
controls E on existing development cases plus stable controls. Pin the actual
post-evidence token state; later evidence cannot label earlier triggers. A future
runner must validate the perturbed history, not bypass the clean A1 validator.
Only F requires new unseen questions after method/selection/cost freezing. These
audit-task letters do not authorize Stage C RL. No GPU runner/inference, new data
download, environment provisioning or RL work occurs in this update.

Validation before publication: **58 public CPU tests pass**. New checks cover
title-only false matches, high overlap despite missing negation, unchanged
semantic decisions, independent reviewer/body-quote requirements, frozen-context
and duplicate-event rejection, explicit pending calibration coverage, mixed-state
aggregation, budget/answer-type separation, empty-answer exclusion, generation
opportunity after evidence, nonentity uncertainty retention, and an end-to-end
native A3 receipt with hash/source preservation. Compilation, CLI help, shell,
JSON, document links and diff checks passed. No factory C/D count reproduction
or mechanism GPU experiment was run on the developer server.


## 2026-10-12 — Factory step-0 calibration accepted only for E engineering preparation

The latest supplied report is factory `0ea168a`, reportedly merging public
`6be41d4`. Raw artifacts and local source remain unavailable here. Factory adopts
the public lexical-only implementation and retracts automatic semantic labels,
including “72 true L3.” Reported diagnostics: 144/175 have at least one body
support-token recall >=0.6; eight independent reviews are 4 sufficient, 2
insufficient, 2 uncertain, 167 pending. Four sufficient cases are HotpotQA only.
Quote presence validation is progress, not automatic semantic entailment; this
is not full C/D coverage or measured precision across all strata.

Accept these cases as engineering seeds without requiring all 175 reviews.
Minimum GPU entry is one independently sufficient failed/stable pair, each
reviewed using ONLY information displayed at its actual start; maximum four
pairs initially. Stable sufficiency was not supplied and must be reviewed. The
fixed protocol uses existing development questions, not new F questions, and
cannot authorize triggering, a population mechanism prevalence, research GO or
RL. Numeric/comparison cases require explicit entity resolution and derivation;
uncertain is appropriate for missing proof, not an automatic permanent category.

Matched E3 now has supplied units: 178 states/151 questions; net delta sums E3
-10 and same-state E1 -13, excess +3 EM counts, reported question-cluster 95%
bootstrap count CI [-4,+10], 2000 draws, seed 42. Point mean is +3/178, not three
percentage points. Resampled cluster sizes can vary, so do not divide this count
interval by 178 and relabel it a mean interval. Same-state pre-intervention
prefix/budget equality is the required check; matching on post-intervention
length/search mediators would change the estimand. Gate 1 remains unestablished,
not evidence of equivalence. Separate cross-question E engineering matches use
fixed dataset/condition/turn/budget and length rules.

Correct two supplied explanations: the old six-category table literally totals
141 including the 29 uncertain; without them it is 112. Retire that distribution,
do not call it a nonuncertain partition. New raw status table is answered 143,
budget_exhausted 21, generation_limit 8, malformed_action 3. The 32 empty answers
therefore split 21 budget / 11 termination, with six nonentity answers being a
cross-axis subset, not a third exclusive cause. No motive of choosing not to
answer follows from terminal status. Reported 46 nonoracle witnesses versus old
union 52 and B5 successes 3 cannot all hold under identical units/eligibility;
the old union would require >=49 nonoracle. Retain this discrepancy for ID-level
reconciliation. The 25-state intersection uses a withdrawn lexical candidate
set and has no verified common post-evidence start.

## 2026-10-12 — Single public E runner implemented; actual perturbation and spent budget preserved

Record E engineering implementation as the next stage; original four research
gates and the not-granted RL decision remain unchanged. Public post_evidence
validates A1 identity/pre-intervention history, the actual one-observation change,
and the full perturbed token/observation history. Restore after
max(parent intervention turn, earliest reviewed sufficient turn). The reported
hotpotqa:5a83ed9b t2/E2 sufficient at t1 MUST start after t2, otherwise it would
skip the treatment. Generated raw token IDs are retained; observation IDs use
the frozen tokenizer as in the original rollout, next input length is checked,
and the subsequent saved prefix is checked when available. New post-observation
SHA is recorded with its reconstruction basis; do not invent a historical saved
post-observation SHA. Perturbed branches are never relabeled clean.

Preparation binds immutable source hashes and re-audits A1/A3. It selects at
most four independently reviewed answered-error questions in frozen A1 order,
then proposes at most three stable matches per case with the same dataset,
condition, parent/start turn and remaining budget, input length ratio 0.75–1.25.
Review truncated prefixes, select first sufficient stable proposal in frozen
order, and finalize all run/skip groups before GPU. These outcome-selected
cases are engineering inputs, not deployable selection features.

The runner inherits source manifest settings and asset/service identity. ALL
selected original A/B controls must match each other and historical continuation
before ANY general-review or original-question-constraint arm runs. Original
four-search budget is never reset; 512 remains a per-generation-segment cap,
not a whole-branch cap. The fourth observation still permits final generation.
No gold/review labels are inserted into policy input. Fixed instruction hashes,
real incremental instruction/input/generated/search costs, failures, full
planned denominators, and normal live retrieval audits are persisted atomically.
A changed parent observation is never used as a normal retrieval reference.
Unknown factory schemas fail explicitly rather than inventing fields. Existing
output directories are not overwritten. Detailed steps: E_EXECUTION_GUIDE.md.

Validation before publication: **71 public CPU tests pass**, including all
previous regressions and actual E1/E2/E3/E4 perturbed reconstruction, retained context-limit failures, t1 sufficiency/t2
intervention anchoring, raw token preservation, immutable source/context binding,
stable prefix-only review and frozen order, original budget/fourth-observation
answer opportunity, all-case control-first stopping, recorded partial
infrastructure failure, source-inherited backend settings, costs and gold/review
isolation. H800 runs and actual factory schema/count reproduction are pending;
no downloads, environment changes or training were performed.


## 2026-10-12 — E v1 all-skip reported; revise scope explicitly before implementing v2

Factory faf8494 reportedly merges c0a1e8b. Four sufficient failed cases were
reaffirmed prefix-only; all twelve nearest stable proposals were judged
insufficient; all four groups skipped, validation yielded zero pairs, runner
refused, no GPU execution. Accept the recorded engineering refusal, not a
population conclusion. Only twelve length-near proposals were reviewed, not
all eligible stable states. Global candidate reservation and top-three caps
also restrict coverage. Missing gold strings do not prove missing derivability:
number/comparison/inferred relations need an explicit constraint audit. Correct
final answers from a t1 start may rely on later retrieval, model knowledge or
reasoning; earlier observations are already included in the prefix. No isolated
parametric-knowledge or B3 harm mechanism follows. Factory reports 113 tests;
public c0a1e8b had 71 verified tests, and no factory source is imported.

Decision BEFORE code changes: retain immutable E v1 and strict sufficient-pair
validator. Do not bypass it or claim its engineering gate passed. Introduce E
v2 as a distinct outcome-selected development probe: primary sufficient failed
cases evaluate within-actual-state original versus general/constraint review;
secondary prefix-insufficient, historically stable cases evaluate sensitivity
and harm separately. No cross-state matching is required for the within-state
action contrast, and no pooled population NetEM is estimable from these selected
strata. Keep the same original candidate pool and fixed nearest candidate order,
anchor, prompts, asset/cost/global-budget protocol and all-case control-first
checks. No search-loop expansion to find favorable results.

V2 prepares at most four primary and four secondary cases (one frozen nearest
reviewed insufficient stable proposal per former group). Independently reaffirm
prefix-only review; an insufficient secondary case must identify missing question
constraints and consideration of alternative derivations. Literal answer absence
alone is inadequate. Uncertain reassessments are explicitly skipped rather than
forced into insufficient. Freeze all run/skip choices before GPU. Primary cases
can run without a secondary case, but then report that stable harm is unmeasured.
Begin one frozen group; extend to at most four only on engineering controls, not
repair success. Maintain E development/no-GO; F needs a separate method/trigger
and unseen-question protocol after review, RL remains ungranted.


Implementation completed: `prepare_e_probe` binds the immutable finalized v1
plan and original A1/A3 hashes, revalidates even skipped groups' review fields,
and freezes the first insufficient secondary and prefix-only case run/skip.
`run_e_probe` passes its plan loader and stratified metric function into the
existing `run_e` engine; token reconstruction, original-control gating, service
identity, global budget, costs and atomic failure receipts remain shared.
Manifest identifies v2, parent-plan hash, all frozen case decisions; metrics
explicitly retain v1_sufficient_pair_gate_passed=false and absent secondary harm
as unmeasured. General-versus-constraint EM/F1/token/search contrasts are separate
for each stratum; no pooled NetEM is produced. V1 rejects insufficient stable
pairs exactly as before.

Validation before publication: **77 public CPU tests pass**, retaining 71 v1
regressions and six synthetic v2 checks covering preserved all-skip v1 rejection,
missing-constraint/alternative-derivation requirements, uncertain skips and
primary-only coverage, immutable context/parent hashes, secondary-only refusal,
shared control-first execution and stopping on secondary control mismatch,
stratified gain/harm and EM/F1/cost contrasts, prompt-label isolation, untouched
source files and no-model/service check-only mode. Compilation, CLI help, shell
syntax, JSON, local documentation links and diff checks pass. No factory raw
counts or H800 inference were executed here; reported 113 factory tests remain
separate. No models/data/env changes or training.


## 2026-10-12 — E v2 report: engineering accepted, constraint-review method NO-GO

Factory 82fa3d5 reportedly merges public 755340e, completed smoke and all four
frozen groups (8 cases), with exact originals/history and complete engineering
coverage. Developer raw artifacts unavailable: supplied aggregates are recorded,
not independently recomputed. Smoke duplicates engineering pair_1, not a fifth
primary observation. Primary general repairs 3/4, constraint 2/4; paired general
only 2, constraint only 1, both 1, neither 0. Secondary general harms 0/4,
constraint harms 1/4, pair_3 with empty final answer/search_budget_exhausted.
No statistical superiority/inferiority conclusion from these selected four
questions per stratum; zero observed general harms is not established safety.

Decision BEFORE implementation: stop scaling and tuning the CURRENT constraint
review method, retain no research/RL GO, do not advance F, and do not introduce a
no-search prompt or four-GPU sharding to salvage this dev set. Original retrieval
content-specific and semantic off-target gates remain unestablished. The v1
12-candidate all-skip observation is not structural absence, nor an independent
population-level disproof. E v2 repairs do witness some budget-feasible recovery
at these same post-evidence states, but generic repair competes effectively and
no identifiable deployable selection/novel special-method advantage is shown.
This is a resource-allocation NO-GO for the method, not proof recovery is useless.

Cost interpretation: constraint saves primary tokens/search at lower primary
accuracy, so a tradeoff, not a blanket “no cost advantage.” Secondary is worse
in observed EM and cost, but only one harm case. Never pool selected strata into
population NetEM. Verify whether general repairs require later retrieval versus
reconsideration of the initial evidence; reviewer sufficiency at the anchor is
not proof that the model's eventual repair used those facts.

Confirm harm as EM loss due to budget/action-terminal failure, not an answered
wrong prediction or parametric-knowledge disruption. Budget was ALREADY fixed at
four total successful searches. Public rollout permits a generation after search
four and sets search_budget_exhausted when a further search is requested and
refused (normal non-error path). Audit the final generated action and spent
calls; two future successful searches alone cannot establish the terminal path.
A zero-additional-search intervention would change action space, and requires
matched generic/original controls; one post-hoc harm does not justify its launch.
The repeated old 32-empty split is corrected to 21 budget / 11 termination;
nonentity six is a cross-axis subset, not a third exclusive cause.

Continue only CPU closeout of EXISTING results: immutable source/plan/controls,
raw metric recomputation, paired outcomes, terminal attempts, and a small pending
manual evidence-use explanation packet. A budget/answer/search action-selection
question is only a future scoping possibility, requiring an observable state
rule, representative held-out development population, proper action controls,
full net benefit/cost/safety and novelty review. Not initiated here. No F/new
questions/GPU/RL. Hardware TP=1 source inheritance is an engineering fact;
parallel workers plus a global original-control barrier could preserve protocol,
so single-process absence of shards is not a scientific prohibition. At eight
cases no sharding investment is justified.


Additional paired insight: reported primary general-only 2 / constraint-only 1 /
both 1 implies a 4/4 union of these tested actions, so recovery is witnessed locally.
An oracle choosing the better action is not deployable, and no selector follows;
in particular, a post-hoc t2 rule is invalid and shares t2 secondary harm. V1 and
v2 share selected source data, not independent population disproofs. Preserve
these distinctions in E_V2_RESULTS_AND_DIRECTION.md.

Implementation: new audit_e_probe ONLY reads existing completed v2 JSON artifacts,
revalidates immutable plan/parent/A1/A3 bindings, actual past/identity, raw controls,
score/cost/metric sums and global search cap. It checks the final refused search
when terminal status is search_budget_exhausted, emits separate harm categories
and paired primary outcomes, and packages actual later Queries/observations and
pending human explanations; no automatic evidence-path mechanism judgment.
CPU SHA comparison is explicitly distinguished from runtime tokenizer verification.
No prompts, source metrics, continuation logic or GPU protocol changed.

Validation: **81 public CPU tests pass**. Four new synthetic closeout checks cover
source immutability/recomputed paired receipt, two successful later searches plus
actual refused fifth search, distinction of terminal harm, and rejected missing
last action, corrupted cost/metrics, controls or frozen plan. CLI help, Python
compilation, JSON, documentation links and diff checks pass. No developer GPU,
factory raw recomputation or additional semantic reviews executed.


## 2026-10-12 — Factory closeout complete; archive current method, no pending E tasks

User reports factory 8ceeda1 merging public 3664469: CPU closeout all eight cases
passed source/plan/control/score/cost/metric/retrieval checks, five event-case human
explanations completed, three non-event explanations pending. Developer does not
possess those artifacts; accept the supplied completed receipt, do not claim a
local rerun. Pending non-event explanations are not missing E eligibility review,
no new audit or GPU pass is required. Archive current constraint-review method
NO-GO, preserve generic baseline, immutable outputs and replay engineering assets;
no F, RL, prompt variant, forced-answer ablation or sharding implementation.

Reported general repairs pair_1/2/4 include 3/2/2 actual subsequent searches.
Pair_3 constraint makes one additional retrieval with the original question as
query and restates initial genus facts. Keep this as an evidence-restatement
observation after retrieval, not proof that later retrieval contributes nothing;
original question text need not equal an earlier actual query, and repeated
exposure/position effects are not excluded. No new mechanism experiment follows.
Pair_3 secondary spends two remaining successful calls then issues the refused
next search; empty terminal harm is budget_search_refusal, confirmed by factory
receipt. The spelling error in the refused query did not yield a retrieval result,
so does not prove retriever failure. Costs reported 151 vs 96 generated and
5404 vs 3027 input tokens. Preserve the local 4/4 repair union without selector
or population claims; retain unresolved historical 46/52/3, not repaired by E.

Next steps documented in POST_ARCHIVE_NEXT_STEPS.md: optional one-pass CPU set
reconciliation from actual B IDs/conditions/locators, unresolved if records are
missing; no mandatory experiment loop. Future work is a separately scoped
literature/observable-state hypothesis/protocol review before any new direction
is adopted. Budget answer/search/leave-unchanged remains uninitiated, not a renamed
resumption. No existing stopped-project controllers/artifacts are imported.
This update changes research records/documentation only; JSON/link/diff checks
are sufficient. Existing 81-test code validation remains unchanged; no code or
experiments modified.


## 2026-10-12 — User authorizes an independent implemented direction, SI1

The user explicitly asks the developer to read the adjacent papers, choose one actionable direction, implement it, publish runnable code and a factory guide, and use a new branch or repository if appropriate. This authorizes a NEW SI1 exploratory protocol and engineering implementation. It does not resume archived constraint review, old E/F, or RL.

Selected scope: natural post-first-observation states, actual same-prefix K/V/G/A actions, gain/harm supervision relative to the frozen original policy, and a prefix-only lightweight selector evaluated on question-disjoint validation. Literature already covers reflection, budget control and counterfactual branching; novelty remains a hypothesis, not a granted claim. Record literature reading and the exact fixed protocol before factory execution. Preserve historical artifacts and GMST independence.

Implementation starts on feat/selective-intervention. Factory executes existing offline Search-R1 assets; no developer GPU experiments or environment provisioning. Keep source IDs and normalized question exclusions, training/validation isolation, controls, costs and real coverage auditable.


## 2026-10-12 — SI1 implementation CPU-verified and ready for factory smoke

Implemented natural-state four-arm collector, global exact-control barrier, raw-prefix and global-budget preservation, explicit answer-only refusal, source/sample/resource bindings and full raw audit. Added train-only lightweight gain/harm/call-cost ridge fitting, observable-state-only selection before evaluation, hash/recomputed decision receipts and fixed/random/risk-neutral controls. Frozen 160/80 question-disjoint development protocol and 20 train-only smoke. First-observation no-generation context-limit cases remain in the full denominator and keep the policy unchanged.

95 CPU tests pass on the actual Linux development repository; shell syntax, compilation, CLI, links and diff checks pass. No real GPU, model, service, paper-baseline, independent confirmation or RL execution. Publish the branch with factory switching/backup and exact commands; user executes in existing factory environment and returns aggregates. Preserve all archived E conclusions.


## 2026-10-12 — User requires larger samples and four-card execution

The user questions the 160/80 allocation and explicitly requires four GPUs to accelerate the factory experiments. Expand the default cohort to 1200 fit + 800 held-out validation (HotpotQA/2Wiki each 600/400), preserve previous frozen outputs, and exclude previously observed SI1 questions. Implement four independent TP=1 workers with immutable round-robin whole-question shards, a GLOBAL original-controls barrier, cross-worker same-query drift checks, failure cancellation and exact ordered merge. Keep research actions/budgets/selector unchanged. New four-card smoke is required before expanded development. Four GPUs are execution resources, not four independent sample replicates. CPU verification pending at this entry; no new GPU execution by developer.


## 2026-10-12 — Four-worker implementation verified, default 1200/800 ready

108 CPU tests pass (95 previous + 13 new). Implemented immutable plan/GPU/question binding, four subprocess dispatch, two phases with all-worker exact-control barrier, cross-worker retrieval signatures, process-group cancellation of own workers only, complete ordered merge and raw audits. Expanded default samples to 1200/800; new dp4 output directories preserve old results and old default SI1 samples are conservatively excluded. New 20-question four-card smoke gates the expanded run. Runtime logs/progress and measured coordinator phase/wall timing added; no fixed 4x speed or real GPU execution claimed. Publish to the already explicitly authorized public branch; factory commands documented in SI1_DP4_QUICKSTART.md.


## 2026-10-13 — SI1 four-card report: engineering accepted, learned benefit not established

User reports 43473fc plus unpublished factory 285cac7 completed 1200-fit/800-validation
SI1 on four workers with global originals exact; mechanical checks pass, 80.8 min
development/134 s smoke. Raw artifacts and the named 27K feedback pack are not
present here. Keep 800 validation rows/model/settings/arms immutable. Prior E NO-GO
is unchanged; SI1 was separately authorized, not a resumed archived method.

Validation keep 230/800; review 223 (27 gain,34 harm); budget 172 (24,82),170
malformed; answer 21 (1,210),722 search refusals. Selector 228 (3 gain,5 harm),
net -2/800, reported 95% CI [-.010,+.00375]; selects keep649/review135/budget14/answer2.
Finite-action oracle271 gives 41 gain states beyond keep, not deployable labels
or guaranteed prefix-observable signal. It does not by itself authorize a larger
selector or RL. Current hypothesis of learned net benefit remains unestablished.

Correct positive inference: conditional harm 5/230 includes skipped correct states;
81.125% keep mechanically reduces exposure. Same-rate random and simple/risk-neutral
predeclared controls exist in validation_report but were omitted from the summary.
Need them before saying selection learned safe/beneﬁcial targeting, and no formal
significance claim from rates alone. Among151 interventions, gain3/harm5/unchangedEM143;
base-correct-trigger denominator must be measured separately. If all800 reachable,
a same-mixture uniform random policy has expected gains4.97875, harms7.6975 and
net -.0033984375; illustrative only, actual reachable/matched report is required.

At the fixed first-observation anchor every reachable row has spent1/remaining3;
there is no budget-stage variation or multi-query drift. The simplistic rule's
remaining<=1 branch is never reachable. 19 numeric plus64 hashed word features
cannot be assumed to detect semantic evidence sufficiency. Answer arm refuses
search requests rather than forces an actual answer:722 failures demonstrate poor
compatibility of this particular prompt/action-space contract, not proof answering
from evidence generally cannot work. Budget likewise is just a budget reminder,
not a full budget algorithm;170 malformed outputs require raw terminal-format
classification before an algorithmic-mechanism conclusion.

Continue only existing-artifact CPU diagnostics and reversible launcher/test-path
fixes. No new GPU, validation parameter/action tuning, automatic unseen confirmation,
new assets or RL. Any next learner/action protocol requires train-only model choice,
new unseen evaluation and separately recorded scope. Eight hundred rows now observed
cannot be recycled as fresh confirmation. Protocol/smoke uses20 current TRAIN
questions; exclude earlier observed cohorts, not claim current smoke is disjoint
from its own development training subset.


Implementation: launcher now honors SI1_PYTHON/PY without configuring environments;
all modes and spawned sys.executable inherit the same existing interpreter. Two
selective test fixtures resolve protocol under ROOT. New diagnose_selective reads
validated original/worker artifacts plus frozen selector/decision/report hashes,
reproduces selected arms and predeclared control metrics, and outputs true triggered
correct/wrong denominators, opportunity overlap/exclusive gains, train feature
variation, cost-only score crossings, observed frozen utility and dataset results.
It does not refit, change prompts/parameters or execute policies. Validation remains
observed development data. The supplied report does not prove significant targeting
or oracle reachability from available features.

111 public CPU tests pass (108 existing plus3 diagnostic checks), including exact
predeclared control recomputation, denominator separation, corruption rejection
and artifact immutability.30 selective tests also pass from tests/ cwd with workspace
on PYTHONPATH; a stub verifies interpreter override without service/GPU calls.
Shell/compile/CLI/JSON/docs/diff checks pass. No actual factory artifacts or archive
were received/read, no additional model/Retriever/GPU/NN/RL execution.


## 2026-10-13 — SI1 CPU diagnostics complete; archive current ridge prototype NO-GO

Source: user factory report d6435e4, merged public bfd5aa7; raw diagnostic/worker
artifacts unavailable here. Reported mechanical checks pass without fitting,
models/services/GPU; launcher/test-path fixes effective. Accept reported completion,
not developer raw verification. Current update is documentation/state only.

Matched-rate random and selector both trigger151/800 with same action counts.
Selector gain3/harm5/unchanged143 versus random gain4/harm5/unchanged142; EM .285
versus .28625. Difference -1/800 is exploratory, not statistical inferiority or
random equivalence. Trigger harm5/151 is3.31% for both; among triggered correct,
reported14.29% versus13.89%. Retire learned avoidance/significant safety claims from
5/230 aggregate harm. Risk-neutral gain9/harm11 has same net as3/5; simple rule's
one gain among8 triggers is not method GO. Dataset keep values alone do not support
dataset-specific selector direction. Train best fixed arm is keep; all train
fixed intervention nets negative. No universal feature/selection impossibility.

Oracle opportunity41, captured3, missed38 plus5 harms gives regret43; not proof
of prefix-observable signal. Do not mine observed validation opportunities to
choose features/actions. Four cost-term margin crossings concern predicted score,
not zero predicted/actual gains. Preserve cost term and all frozen results; no
posthoc deletion/refitting. Actual reported utility -.00859 and tiny mixed token/
call changes establish no practical advantage. Future quality versus efficiency
objectives must be separately specified in a new train-only protocol.

Answer722 search refusals limit this prompt/action contract; do not claim all
evidence-only answering impossible or all failures solely caused by compatibility.
Budget170 malformed may receive optional one-pass CPU raw-format descriptions,
without parser changes/retries/new prompts; not a prerequisite for archive.

Decision: archive current SI1 ridge prototype NO-GO; no mandatory next run or
additional feedback, no new GPU, SI2/confirmation/RL. Keep replay/DP4/barrier/random
and diagnostic assets; keep old E NO-GO and unresolved historical46/52/3 separate.
Any future project needs separately authorized scope, stable observable signal
across train splits, same-rate/random/keep/simple controls, explicit objectives
and new unseen confirmation. Record and report:
report/SI1_CPU_DIAGNOSTICS_AND_NO_GO_20261013.md.


## 2026-10-16 — New direction: path luck and placebo-controlled replay (PCR v1)

The user converged Work 2 on a new research question after reviewing all archived
results and recent literature: a substantial share of RL-trained search-agent
successes may be path luck rather than evidence-driven. The reported legacy
signals motivate it (not paper results, not independently recomputed here): E1
permutation flips in BOTH directions (75/583 degraded, 96/1389 improved); matched
E3 excess over E1 +3 counts, CI [-4,+10]; B3 repair 44/134 on outcome-selected
degraded states alongside 53/449 stable harms, a regression-to-the-mean confound.
The archived drift→recovery, E, and SI1 methods stay NO-GO and are not resumed.

Decision: open branch `feat/path-luck-pcr` with a pre-registered protocol
(research/PCR_PROTOCOL.md) frozen BEFORE any Phase 1 outcome. Every state forks the
same raw prefix; E0/E0 must reproduce the source exactly; six semantic-null
placebos (4 outcome-blind permutations + 2 byte-identical-content format variants)
estimate p(s); recovery arms (review, token-matched neutral, budget+2, and the
same on the perm_1 observation) and deletion arms (support vs non-support slot,
same replacement candidate) are assigned to every state before outcomes. Primary
estimands use the format-corrected view (alias/containment/F1>=0.8 flips removed).
Gates G1/G2/G3 and their dispositions are fixed in the protocol; analysis emits
mechanical values only, the GO decision is recorded here by a human. Default
sources reuse SI1 dev natural E0 sources (600 questions, turns 1/2). Phase 0 adds a
CPU re-analysis of existing A1/A3 raw files with the audit_pairs contract; the
unpublished B schema is not audited further, Exp-2 replaces it.

Implementation: work2/placebo.py, work2/pcr.py (prepare / N-GPU TP=1 workers with
resume / ordered merge), work2/pcr_analysis.py, work2/phase0_luck.py,
configs/pcr_v1.json, scripts/run_pcr.sh, docs/PCR_FACTORY_GUIDE.md. 124 CPU tests
pass (111 existing + 13 new: format/permutation invariants, token-matched neutral,
deletion candidate sharing, all-arm state with exact controls, budget+2, control
drift stop, turn-2 fork, analysis gates, tamper rejection, prepare→worker→merge,
Phase 0 collection). No H800 execution, no RL; Phase 2/3 (more checkpoints/data,
LDR training) require a recorded decision on G1–G3 first.

## 2026-10-16 — PCR Phase 0/1 reported: G1 NO-GO, G2/G3 pass; path-luck headline retired

Factory report (HEAD `d8ff019`, NOT in the public remote whose branch is `4d74af6`;
factory must return the log/diff before results move past provisional). 600 SI1-dev
questions, 1177 states, all E0/E0 exact, drift 0; turn-1 review arm 600/600 identical
to SI1. Reported arithmetic checks out (28/353, (28+30)/1177, net −25 / −13).

G1 false: semantic lucky share 7.93% [4.76, 11.37] (raw 9.07%). Per the frozen
protocol this is NO-GO for "a substantial share of correct answers is path luck".
No threshold lowering, no redefinition (any-placebo fragility 1−rEM/EM≈25% stays
descriptive), no sample enlargement to rescue. mEM≈EM: bidirectional flips cancel,
so aggregate-EM bias is negligible and the ranking-noise RQ loses its premise.
G3 true: drop_support NCE −9.0 pp vs drop_nonsupport +0.06 pp, difference −9.1 pp
[−13.6, −4.8]; the earlier "bridge removal ≈ permutation" reading is withdrawn.
G2 true: on perm_1-selected failures review repairs 30.8% (harm 11.8% on stable),
reproducing B3 32.8%/11.8%, while a second permutation alone repairs 41.0%;
placebo-corrected net −13. Natural-state review −1.0 pp (CI covers 0); budget2
+1.0 pp. Phase 0: E1 format flips 12/171; semantic degradation 11.8% [9.0, 14.8].

Decision: retire the path-luck headline; cancel Phase 3 LDR and the luck-prevalence
Phase 2 matrix. The protocol's permitted fallback (methodology short paper on
placebo-controlled replay: selection-induced mean reversion in recovery rates,
resolving power for evidence effects) is pending the user's choice; no new protocol
or code until then. Pack fix: analysis also writes flip_cases.json, pack converts
old .jsonl outputs. Details: research/PCR_PHASE1_RESULTS_AND_DECISION.md.

## 2026-10-16 — New direction CT1: interventional validation of turn-level credit signals

After PCR G1 NO-GO the user asked for a fresh, solid idea. Developer survey (kept
locally, not in this repo) found many 2026 turn-level credit/process-reward methods
for search agents (IGPO, IG-Search, TRACE, CIPO, KnowGPO, STAMP, E-GRPO,
CriticSearch) with no validation against REAL re-executed causal effects; QVal
(2606.32034) validates dense signals only in non-search environments, intervenes
on actions only, uses reference-optimal policies and leaves downstream RL
validity open. The user chose this direction (A) and asked for a frozen protocol,
the week 1–2 pilot code and a factory guide on a new branch.

Protocol research/CREDIT_PROTOCOL.md frozen before any CT1 outcome: 7B PPO,
150+150 outcome-blind questions from SI1 dev sources, temperature-1 on-policy
sampling, k=8. Truth A_t = V(s_t) − V(s_{t−1}), E_t = V(s_t) − V(s_t with a
semantic-null observation), closed-book V_cb with all retrievals nulled. Signals:
IGPO Eq.3–4 information gain (+dead-zone), optional TRACE-style reference TD,
CIPO-style EALR with a 4D attention mask (runtime self-check refuses ignored
masks), first-exposure fact/answer hits, random. Gates G-a (split-half
Spearman-Brown >= 0.6), G-b (reportable disagreement), G-c (cost); dispositions
fixed in the protocol. PCR code and results remain archived on feat/path-luck-pcr.

Implementation: work2/credit_testbed.py (lockstep batched multi-turn sampling,
batched live retrieval with drift checks, N-GPU workers with resume, merge),
work2/credit_signals.py (HF scoring), work2/credit_analysis.py, configs/credit_v1.json,
scripts/run_credit.sh, docs/CT1_FACTORY_GUIDE.md. Additive backend changes:
VLLMGenerator.sample_batch, SearchR1Retriever.search_batch (search() unchanged
in behaviour). 136 CPU tests pass (124 + 12 new: batched≡sequential rollout,
spent-budget carry-over, null-world isolation, null-document filtering, value
layout and token segments, drift stop, IGPO/EALR spans, fake-scorer EALR,
rank statistics, gates, tamper rejection, prepare→worker→merge). The torch/HF
scoring path cannot run on the developer server (no torch); its mask behaviour is
guarded by the runtime self-check. No H800 execution yet.

## 2026-10-16 — CT1 verified on the home server (searchr1 conda env, RTX 5090)

Environment: /mnt/wh/miniconda3/envs/searchr1 (python 3.12, torch 2.9.1+cu130,
transformers 4.47.1, vLLM 0.6.3, xformers 0.0.33); all 136 tests pass there.
vLLM 0.6.3's bundled FlashAttention rejects sm_120, so on the 5090 the run needs
VLLM_ATTENTION_BACKEND=XFORMERS (as Search-R1's own scripts set); H800 unaffected.
Real-model checks with Qwen2.5-0.5B-Instruct: batched seeded sampling works and
reproduces; the full CLI chain (prepare, check against a Search-R1-format HTTP
stub, 2-GPU MC workers, merge, 2-GPU signal workers, merge) completes.
Mask finding: with transformers 4.47.1, eager attention reproduces the default
causal mask EXACTLY when an explicit 4D mask is passed (max diff 0), while
sdpa+bf16 switches kernels and differs by up to 0.87 per-token log-prob (fp32
exact). Therefore the default is now eager, and the EALR visible side also uses
an explicit causal 4D mask so both sides share one kernel. On a hand-built
search trajectory EALR = +0.35, same sign as an observation-deletion contrast
(+0.71); the self-check rejects sdpa as designed. Not yet verified: a model that
actually follows the search protocol end to end (Search-R1 3B download stalled
at 65% when the 7890 reverse proxy dropped; resumes when the proxy is back).

## 2026-10-16 — CT1 smoke stopped on retrieval "drift": tie-order instability, fixed client-side

Factory (HEAD 0c6ac67, H800, torch 2.4.0, transformers 4.47.1) stopped the MC smoke
three times on `Live retriever drift` for the agent query "where is Doc Scott
raised": isolated calls were stable, but its Top-3 scores contain a tie
(0.8303, 0.8251, 0.8251) and under 4-worker concurrent batched load the service
swapped tied ranks; the service also returned a transient HTTP 500 at peak.
No code was changed by the factory.

Decision: not a service bug to patch and not a per-query exemption (ties can hit
any query, not just decoys). CT1 is a Monte Carlo over sampled branches, so what
matters is that the environment is a FIXED function of the query within a run.
LiveSearcher now (1) canonicalises order by (score rounded to 4 d.p. desc,
document_id), (2) caches the first canonical result per query per worker, so all
branches see identical observations, (3) every 50th cache hit re-fetches and
counts (non-fatally) Top-k id-set changes, reported as retrieval.audit_change_rate
(>5% flags an unstable service), and (4) retries transient HTTP/connection errors
with exponential backoff (4 attempts). Strict ordered drift checks in A2/SI1/PCR
are unchanged. Protocol estimands are unchanged; cached determinism is a stricter
environment, not a looser one. 139 tests pass (136 + 3 new; the old fatal-drift
test is replaced). Two-GPU stub end-to-end rerun on the home server passes.

## 2026-10-16 — CT1 k=8 pilot reported: G-a fails on A only; preregistered k=16 rerun

Factory HEAD 0e4f22e (public afc1c05 merged), 300 questions, 869 turns, 18,704
branches, MC 20.3 min, all trajectories reproduced, null eligible 300/300, EALR
mask self-check exact. Reliability: A 0.259 [0.040, 0.433], E 0.740 [0.660, 0.805]
-> G-a false. G-b true: rho vs E ig 0.089, ealr 0.043, fact_hit 0.079 (answer_hit
0.150, random 0.001); rho(ig, ealr) 0.037. G-c true (0.68 h for 600).
Interpretation: A's true variance is tiny (mean A 0.0014; on-policy resampling
from s_{t-1} mostly reproduces the turn), so noise dominates; Spearman-Brown
predicts A 0.41 at k=16 and k~34 for 0.6. Even so, disattenuated bounds already
exclude moderate IG-A correlation (upper ~0.13) and cap evidence-signal agreement
with E at ~0.12-0.18. 72% of turns sit in failed trajectories with E~0 (floor),
which must be addressed; the memory-stratum hypothesis is not supported.

Decision per CREDIT_PROTOCOL section 6: run the single preregistered k=16 rerun on
the same frozen questions (configs/credit_v1_k16.json; first 8 samples identical
by seed). If A still < 0.6, A-based claims are NO-GO and only E-based work may
continue under a new CT2 protocol on fresh questions. No gate is lowered and no
estimand is redefined on the seen 300 questions. Added for the rerun, labelled
exploratory and outside the gates: cf_ig (gold log-prob with the real vs the
stored null observation, a teacher-forced proxy of E), an informative-turn
stratum, disattenuated correlations and reliability prophecy. Records now store
null_observation_token_ids. 141 tests pass. Analysis:
research/CT1_PILOT_RESULTS_AND_DECISION.md.

## 2026-10-19 — CT1 k=16 rerun: G-a fails again -> CT1 v1 NO-GO; novelty threat found

Factory HEAD e279f63 (not public; reports 170 tests vs public 141 — factory must
return log/diff vs c39af3c before results leave provisional). 300 questions,
37,408 branches, 30 min, retrieval audit 0/107 changes. A reliability 0.492
[0.294, 0.639] (< 0.6), E 0.885. rho vs E: ig 0.087, ealr 0.064, fact_hit 0.154,
answer_hit 0.163, trace 0.060, cf_ig 0.026; informative-subset values similar;
disattenuated rho vs A ~0.03 for all signals.

Decision: per CREDIT_PROTOCOL section 6, CT1 v1 is NO-GO (second G-a failure).
No further k=25 rerun (one rerun was preregistered; the A-side conclusion would
not change). cf_ig dropped as a method candidate. Literature re-scan found
2608.19760 (executed-replay audit of step-level credit in ALFWorld: no signal
beats a shuffled control; preregistered training arms differ by dose not credit)
and 2609.02417 (shuffled per-turn rewards cost nothing on tau^2-bench). These
pre-empt CT's core claim in general agents; our remaining delta is search QA,
observation interventions (E) and search-specific signals. CT2 is NOT started;
the user chooses between (a) a consolidated interventional-audit analysis paper
(PCR + CT1 findings, confirmed on fresh questions and a second checkpoint) and
(b) a new direction. The earlier survey's gap claim was wrong for general
agents; recorded in the Chinese survey document.

## 2026-10-19 — User decision: start (a) AC1 confirmation, scan for (b) in parallel

The user chose to start the confirmation experiments for the consolidated
interventional-audit paper (a) and, in parallel, run a literature scan for a
new main-conference direction (b), deciding between them in two weeks. The (b)
scan runs as a separate read-only session (output in the user's local folder,
not this repo).

AC1 protocol research/AUDIT_CONFIRM_PROTOCOL.md frozen before any AC1 outcome:
1000 never-used HotpotQA/2Wiki dev questions (excluded by id AND normalized
question text against all historical cohorts), greedy E0 sources regenerated per
checkpoint; 7B PPO plus one second offline Search-R1 checkpoint (3B PPO > 7B GRPO
> 3B GRPO, chosen and logged before its results). Claims: H1 recovery repair
explained by placebo (>=1/3 or reversed net) for review or the new reflect
method; H2 support minus non-support deletion CI upper < 0; H3 on informative
turns ig/ealr/fact_hit rho vs E < 0.3 with CI upper < 0.4 given E reliability
>= 0.8. Continue (a) if >=2 of 3 confirm on the main checkpoint and >=1 in the
same direction on the second.

Implementation: work2/audit_confirm.py (freeze, multi-GPU E0 sources with resume,
report verdicts), PCR arm group "audit" (+reflect, neutral_reflect,
perm1_reflect, perm1_neutral_reflect with a token-matched placebo), PCR recovery
analysis generalised per method with a reperturb-difference CI,
configs/audit_confirm_v1.json, configs/pcr_audit_v1.json, scripts/run_audit.sh,
docs/AC1_FACTORY_GUIDE.md. 146 tests pass. No H800 execution yet.

## 2026-10-19 — AC1 on fresh questions (7B PPO only): H1 and H3 confirmed, H2 not

Factory f50a195 = public c21a997 plus factory-local dashboard/progress files (8
local commits c39af3c..e279f63; 29 local tests explain 170/175 vs 141/146); no
experiment path changed — accepted. No official second Search-R1 checkpoint
exists offline at the factory, so per protocol only the main checkpoint ran.
1000 fresh questions (3000 historical excluded), E0/E0 exact, drift 0.
H1 confirmed for review and reflect (D=56; review repair 35.7%, explained 1.25 —
the 1.25 implies reperturb P1 ~44.6%, not the 28.6% neutral figure the report
attributed it to; full fields requested). H2 NOT confirmed: support minus
non-support -2.1 pp [-5.0, +0.9] (dev set -9.1 pp). H3 confirmed: E reliability
0.849, informative-turn rho ig 0.082, ealr -0.038, fact_hit 0.019. Descriptive:
lucky share 6.6%, clean EM 0.253, A reliability 0.668 (descriptive only; CT1 v1
verdict unchanged).

Decision: (a) continues per AC1 section 4 with two claims (H1 mean reversion in
recovery evaluation; H3 evidence-credit signals do not track interventional
evidence value); H2 is reported as a non-replication, not a claim. Single
checkpoint is the main weakness; any second policy (Qwen2.5-7B-Instruct, or the
factory's own GRPO run if HF weights exist) must be registered before running.
Whether to start the (b)-A handoff-replay pilot now is the user's decision.
Details: research/AC1_RESULTS_AND_DECISION.md.

## 2026-10-19 — User starts the (b)-A handoff-replay pilot HR1

The user asked to freeze a protocol and run handoff replay between
Qwen2.5-7B-Instruct (I) and Search-R1 7B PPO (R), both available offline at the
factory. Protocol research/HANDOFF_PROTOCOL.md frozen before any HR1 outcome:
300 fresh HotpotQA/2Wiki dev questions disjoint from all history INCLUDING the
AC1 fresh set; temperature-1 sampling, k=8; one scored trajectory per policy;
handoff points t in {0,1,2,3} and end; arms real, placebo (all observations up
to t replaced by the frozen null observation), evonly (think text dropped).
Estimands: handoff curves, 2x2 Shapley acquisition/utilization at end, and the
evidence-free gap; gates G-A1 (martingale self-replay + placebo drop >= 5 pp),
G-A2 (component CI half-width <= 3 pp), G-A3 (evidence-free share >= 30% or a
dominant pathway >= 60% with CI low >= 50%); criterion (iii) not applicable
with a single RL checkpoint.
Pre-outcome amendment: the two tokenizers' chat templates differ (system
prompt "You are a helpful assistant." vs "You are Qwen, created by Alibaba
Cloud..."; identical 151,665 vocab), so both policies use the initial prompt
rendered by R's tokenizer; every handoff is then token-identical, and the
instruct-native mismatch is logged as a limitation.
Implementation: work2/handoff.py (freeze, three phases r1/i2/r3 with one model
per GPU process, resume, merge, smoke), work2/handoff_analysis.py,
configs/handoff_v1.json, scripts/run_handoff.sh, docs/HR1_FACTORY_GUIDE.md;
audit_confirm freeze refactored into select_fresh/write_selection. 149 tests pass.

## 2026-10-20 — AC1 second checkpoint (official Search-R1 3B PPO): 3/3 confirmed

The factory obtained SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3 through an
internal HF mirror and reran AC1 on the SAME frozen fresh questions (no
re-freeze). E0/E0 exact, drift 0. H1 confirmed (95 degraded states; explained
review 1.36, reflect 1.07); H2 confirmed (-7.5 pp [-11.2, -4.0]); H3 confirmed
(E reliability 0.937; ig 0.183, ealr 0.124, fact_hit 0.137; answer_hit 0.235).
clean EM 0.359, mean E 0.078, lucky share 0.063, A reliability 0.641. The
factory's own 3B GRPO run emits only malformed actions and is not a usable
checkpoint.

Decision: the AC1 continuation criterion is met (main 2/3, second 3/3 in the
same direction); the single-checkpoint limitation is removed. Paper claims:
H1 and H3 (confirmed on both checkpoints). H2 is reported per checkpoint as
"confirmed on 1 of 2, same sign": not robust, not a non-replication. The factory's
"power" explanation is incomplete — 3B depends more on evidence (mean E 0.078 vs
0.031) and the 7B dev-set estimate (-9.1 pp) was inflated relative to fresh
(-2.1 pp); the wording is that single-document deletion effects grow with
evidence dependence. HR1 remains pending as frozen. If an official 7B/3B GRPO
checkpoint is reachable through the mirror, it must be registered here before
running as an AC1 third checkpoint or for HR1's PPO-vs-GRPO criterion.

## 2026-10-20 — HR1 NO-GO; harness-fidelity check FC1 before writing (a)

Factory run outputs/hr1_pilot_v3 (frozen f61a8a4; records e54b0eb0...): 300 fresh
questions, 76,512 rollouts, template identical 300/300. Trajectory EM R 0.240 vs
I 0.250; gain_end -0.005 [-0.049, 0.040]; acquisition -0.024, utilization 0.019,
evidence_free 0.002, evidence_free_evonly -0.089 [-0.135, -0.046]. G-A1 fails
(placebo drop 1.2 pp < 5 pp; martingale checks pass), G-A2 fails (half-widths
3.28/3.00/4.08 pp), G-A3 fails, gain_end CI contains 0. Decision: HR1 NO-GO;
candidate A closed; per the (a)/(b) tree, return to (a). Candidate B not started.
Correction to the factory reading: evonly keeps observations and removes think
text, so R's evonly deficit means R depends on its own reasoning trace (or is
shifted more by its removal), not on observations. The placebo arm keeps think
text that restates evidence, a known conservative bias.
A v2 attempt was correctly refused by the code-freeze check after a mid-run
factory commit; its data is excluded.

Validity concern raised by HR1: R does not beat Instruct, and 7B (0.253) trails
3B v0.3 (0.359) in AC1. Our 7B is the original v0.1 release (no suffix); v0.3
7B PPO/GRPO exist. Our bounded prompt also deviates from the official Search-R1
template ("You can search as many times as your want"). Before writing (a) we
run FC1 (research/FIDELITY_PROTOCOL.md, frozen before outcomes): greedy EM on the
AC1 fresh questions for v0.1 7B PPO, v0.3 3B PPO, v0.3 7B PPO, v0.3 7B GRPO and
Qwen2.5-7B-Instruct, each with bounded and verbatim official prompts; rules D1
(prompt depresses an (a) checkpoint -> rerun AC1 with the official prompt), D2
(v0.3 7B PPO >= +3 pp over v0.1 -> new main 7B), D3 (RL vs Instruct, report;
HR1 re-registration is the user's call), D4 (GRPO usable as third checkpoint).
Remaining deviation reported as is: no 500-token observation truncation.
Implementation: rollout.question_prompt variant "official" (verbatim template,
prompt_version search_r1_official_base_v1; default unchanged), audit_confirm
accepts FC1 configs, work2/fidelity.py, configs/fidelity_{bounded,official}.json,
scripts/run_fidelity.sh, docs/FC1_FACTORY_GUIDE.md. 153 tests pass.
Factory process note: DECISION_LOG.md and STATE.json are maintained here; factory
execution notes go to factory-local files to avoid merge conflicts.

## 2026-10-22 — Target ACL 2027 main; upgrade plan confirmed; U1 implemented

The user restated the target: ACL 2027 main conference, not Findings. Developer
assessment: with 2 checkpoints, 2 datasets and our own recovery prompts, (a)
reads as "correct but incremental". Upgrade plan confirmed by the user: U1
(breadth: official Search-R1 agents across sizes, PPO/GRPO, base/instruct, and
five datasets), U2 (audit >= 2 published recovery methods; literature scan running
in a separate read-only session), U3 (training-side test) decided after a training
pipeline pilot, U4 (release the audit toolkit) during writing. Milestones 11-04,
11-25, 12-09 with stop rules are recorded in the user's status document.

U1 protocol research/U1_PROTOCOL.md frozen before any U1 outcome. The only
open choice, the prompt variant, is fixed by FC1 rule D1 before any run (the
runner refuses without U1_PROMPT). Questions: AC1 fresh HotpotQA/2Wiki reused
verbatim; NQ 500, MuSiQue 500, Bamboogle all from FlashRAG files, outcome-blind,
excluding historical question text. Roster (configs/u1_agents.json): v0.1 7B PPO,
v0.3 3B PPO/GRPO, 7B PPO/GRPO, 7B-it GRPO, 14B GRPO. Per agent: greedy sources;
PCR with the slim arm group "h1" (9 arms) on <= 400 searchable questions per
dataset; CT k=16 on 150 per dataset. Per-agent H1/H3 criteria as in AC1
(fact_hit skipped where undefined); claim holds on >= ceil(0.8 x included
agents); agents with >= 10% format/length failures are excluded and reported.
Implementation: multi-alias EM (best alias, FlashRAG convention; IGPO scores
the first alias), canonicalize_flashrag, official-prompt sources replay through
validate_prefix, PCR/CT prepare over any dataset set (small sets used in full and
recorded), recovery analysis works for the h1 group, work2/u1.py (freeze, check,
cross-agent summary with per-dataset blocks), configs, scripts/run_u1.sh,
docs/U1_FACTORY_GUIDE.md. Existing HotpotQA/2Wiki behaviour is unchanged.

## 2026-10-23 — FC1 results: prompt is fine (D1 no), main 7B -> v0.3 (D2 yes)

Factory 1dd463c (public 52ca62e + factory-local dashboard), 8 new runs frozen
there. Greedy EM on the 1000 AC1 fresh questions (bounded / official):
v0.1 7B PPO 0.254/0.270, v0.3 3B PPO 0.360/0.354, v0.3 7B PPO 0.408/0.405,
v0.3 7B GRPO 0.389/0.386, Qwen2.5-7B-Instruct 0.243/0.243.
D1 not triggered (official - bounded at most +1.6 pp, CI includes 0): U1_PROMPT =
bounded; no AC1 rerun with the official prompt. D2 triggered (v0.3 7B PPO - v0.1
= +13.5 [10.8, 16.4] official, +15.4 bounded): main 7B becomes v0.3 7B PPO; the
AC1 H1/H3 rerun on it is carried by its U1 run (same HotpotQA/2Wiki fresh
questions plus three datasets; H2 is no longer a claim). D3: RL - Instruct
(official) v0.1 +2.7 [-0.5, 5.8], v0.3 7B PPO +16.2, v0.3 7B GRPO +14.3, 3B +11.1;
HR1's null is attributed to the v0.1 checkpoint; re-registering HR1 on v0.3 is
the user's call. D4 passed (GRPO format/length failures 0.0%); v03_7b_grpo is the
third checkpoint and is already on the U1 roster. Note: v0.3 7B PPO uses ~3.97
searches per question vs GRPO ~1.9; report them separately.
AC1 supplementary fields confirm the earlier correction: explained 1.25 = P1
re-perturbation 44.6% / review 35.7%. Signals vs A are weak on both checkpoints.
Process: a factory FC1 run OOM-killed the user's own GRPO training (progress from
step 50 to 80 lost; resumable from global_step_50). U1 (4 GPUs, ~1.5-2 days) must
be scheduled with the user before launch. Details:
research/FC1_RESULTS_AND_DECISION.md.

## 2026-10-23 — GPU schedule: U1 first, user's GRPO resumes afterwards

The user decided to run U1 first on all four GPUs (~1.5-2 days, U1_PROMPT=bounded).
Their own GRPO training stays stopped until U1 finishes and is then resumed from
global_step_50. The factory must not start other GPU jobs while U1 runs.

## 2026-10-23 — Correction: the "GRPO training" was a GPU keep-alive program

The user clarified that the job the factory reported as "the user's GRPO training"
(OOM-killed during FC1, "resumable from global_step_50") is only the user's GPU
keep-alive program. It is not research work: no resume, no schedule dependency,
no record beyond this correction. The two previous entries' statements about
resuming it are void. Remaining rule: before any run, confirm the four GPUs are
free (nvidia-smi); the keep-alive is stopped and restarted by the user's own
practice. FC1_RESULTS_AND_DECISION.md section 5 and the U1 guide are corrected.

## 2026-10-23 — HR2 and U2 registered (code, protocols, factory guides)

User decision: re-register HR1 on v0.3, and implement the U2 audits directly.
Both protocols are frozen before any result. Execution order: U1 -> U2 -> HR2.

HR2 (research/HR2_PROTOCOL.md, configs/handoff_v2.json, scripts/run_hr2.sh,
docs/HR2_FACTORY_GUIDE.md): R = Search-R1 v0.3 7B PPO (main run) and v0.3 7B
GRPO (second run), I = Qwen2.5-7B-Instruct, bounded prompt (FC1 D1); 300
HotpotQA + 300 2Wiki fresh questions excluding AC1/HR1/U1 history (seed
20261023); k = 8, temperature 1.0. New arm evonly_placebo (thinking dropped,
queries kept, every observation nulled). Changes vs HR1, made because HR1's
placebo kept evidence paraphrased in <think>: G-A1(b) becomes evonly minus
evonly_placebo >= 5 pp with CI lower bound > 0; the primary evidence-free
component becomes the R-I gap under evonly_placebo (HR1's definition reported).
G-A3(iii) now applies: |share_PPO - share_GRPO| >= 20 pp with a paired CI
excluding 0 for acquisition or utilization (handoff_analysis --compare-with).
Other gates unchanged. Shared history exclusion now accepts the U1 FlashRAG
datasets (selective.exclusions previously rejected them).

U2 (research/U2_PROTOCOL.md, configs/u2_reseek.json, configs/u2_mrsearch.json,
work2/u2.py, work2/u2_analysis.py, scripts/run_u2.sh, docs/U2_FACTORY_GUIDE.md):
questions = U1 frozen set, first 300 per dataset by sha256(20261023:id).
ReSeek-3B (TencentBAC weights; system + user prompt verbatim from the repo,
checked byte-for-byte via ast): greedy source; from every post-observation state
k = 4 samples per arm nat / force_no / force_yes (same sampled pre-judge head,
judge value forced) / perm (P1). Primary set S_fail = wrong source with the
agent's own judge = Yes; also S_natNo (nat - force_yes) and S_ok (harm).
MR-Search test-time version on Search-R1 v0.3 7B PPO (main) and 3B PPO: greedy
episode 1 with the verbatim Search-R1 template; episode 2 via MR-Search's own
retry_str construction with k = 4 per arm reflect (retry_prompt verbatim) /
reask (reflection sentence replaced) / neutral (token-matched, P2) / fresh (no
history); final = last valid answer; primary set S_wrong. H1-U2 uses the AC1
rule (placebo / method >= 1/3, or placebo-corrected net <= 0) and counts only if
the fidelity gate passes (ReSeek judge after >= 80% of observations and <= 10%
format failures; MR episode-1 format failures <= 10% and reflect episode-2
answer rate >= 80%). Framing per the U2 scan: the methods report full-question
EM, so U2 states how much of each mechanism's repair on failure-selected states
same-state placebos reproduce. Real-tokenizer checks on the home server
(Search-R1 3B, Qwen2.5-7B-Instruct): MR user turn equals retry_str, neutral turn
142 vs 143 tokens, judge cut exact. CPU tests: all pass.

## 2026-10-27 — U1 / U2 / HR2 results and decisions

Factory runs frozen at factory HEAD 0f74626 (public 37bb169 + a factory-local
commit). Details: research/U1_U2_HR2_RESULTS_AND_DECISION.md.
U1 (7 agents x 2125 questions over 5 datasets): H1 confirmed 7/7 (best-placebo /
method repair >= 1 almost everywhere). H3 confirmed 3/7 (< 6 required): NOT
confirmed. v03_3b_grpo and v03_14b_grpo refute it (E reliability 0.90/0.97, ig rho
0.37/0.37); v03_7b_ppo and v03_7b_it_grpo are undetermined (E reliability 0.76/0.67
< 0.8; the factory quoted 0.75 by mistake). H2 is absent by design (U1 arm group h1
has no deletion arms). Decision: H1 is the paper's main claim with pooled per-placebo
effect sizes (new work2/paper_h1.py); H3 is demoted to a secondary analysis
("checkpoint-dependent, at most moderate, mostly no better than answer_hit"); no
further compute on E reliability; U3 (training with credit signals) is cancelled.
U2: all three runs pass fidelity. ReSeek-3B: H1 confirmed under the preregistered
rule (resample explains 0.38, perm 0.67), but force_no exceeds nat by +7.1 pp
[5.3, 8.9] and perm by +3.7 pp [2.0, 5.4] on S_fail, while harming S_ok by 16.3 pp;
forcing Yes beats natural behaviour on S_natNo and on all states (CIs pending).
MR-Search 7B/3B: H1 confirmed against neutral, fresh and reask; harm on S_right 4.7 /
17.3 pp. Both enter the paper, reported with the residual effects and harms.
HR2: PPO passes G-A1 and G-A2, fails G-A3 (i)/(ii) (acquisition 44%, utilization
56%, placebo-arm evidence-free 2.4 pp [-0.3, 5.2]; the evonly_placebo EF of -8.7 pp is
the bias anticipated in section 6) -> supplementary analysis of (a). GRPO fails G-A1
(martingale -3.5 pp [-5.3, -1.7]; GRPO-continued combos 35-41% format failures vs ~1%
for I). G-A3(iii) is mechanically True but NOT accepted: utilization contrasts GRPO-
vs I-continuation, exactly the term the format failures depress, and G-A1 exists to
bar invalid runs; the protocol omitted "(iii) requires both runs to pass G-A1" and the
intent is applied. No PPO-vs-GRPO mechanism claim. No HR2b scheduled.
Paper (ACL 2027 main) narrows to the recovery audit: protocol, H1 across 7 agents x 5
datasets, U2 published mechanisms, rescue/harm accounting; credit signals secondary;
HR2 PPO in the appendix. Next: factory CPU step docs/PAPER_TABLES_GUIDE.md; the user
decides on adding a non-Search-R1 agent for generality.

## 2026-10-27 — U1x registered: H1 on other training recipes

User decision: full plan, H1 only, four agents from three non-Search-R1 recipes
(research/U1X_PROTOCOL.md, configs/u1x_agents.json, scripts/run_u1x.sh,
docs/U1X_FACTORY_GUIDE.md): ZeroSearch wiki V2 Llama-3.2-3B and Qwen2.5-7B
(simulated-search RL; prompt = Search-R1 official verbatim), AutoRefine-Qwen2.5-7B-
Base (GRPO + retrieval reward, <documents>/<refine>), StepSearch-7B-Base (StePPO).
Candidates rejected for adapter cost: ReSearch/ReCall, R1-Searcher; web agents.
Implementation: agent formats in work2/rollout.py (FORMATS: prompt version,
observation tag, inner padding, strip, document style) threaded through
continue_record, replay.observation/validate_prefix, placebo.render, the CT engine
and audit_confirm; bounded/official/reseek keep their historical rendering. Prompts
read with ast from the upstream repos (no execution), sha256 pinned in tests.
Tokenizers without a chat template get BOS when tokenizer.encode adds it (Llama-3),
matching upstream; verified token-identical on a real Llama-3.2-3B tokenizer.
Harness: E5 + Wiki-18, top-3, 4 searches, U1 frozen questions, PCR h1 arm group
(<= 400 per dataset, turns 1-2); no H3. Fidelity references read from the papers
before any run (work2/u1x.py); macro gap < -10 pp marks an agent possibly
mis-harnessed (reported separately, not counted); format failures >= 10% exclude.
Generality claim: H1 confirmed on >= ceil(0.8 x included unflagged U1x agents);
pooled effect sizes over U1 + U1x via paper_h1 with both rosters. 177 CPU tests.

## 2026-10-27 — Paper tables back: H1 effect sizes, full U2 numbers

Factory ea6d6bf (CPU only). Pooled over 7 agents / 706 degraded states (semantic):
repair P1 48.0%, review 29.9%, reflect 33.1%, budget2 ~1.8%; review - P2 +2.1
[-1.5, 5.8], reflect - P2 +0.8 [-3.0, 4.7] (raw view +1.1 / +0.1), review - P1 -18.1,
reflect - P1 -14.9. Reading: recovery-prompt repairs equal token-matched content-free
text; re-drawing the luck repairs more; a no-text change (budget2) repairs ~nothing,
so repair comes from perturbing the deterministic decode. Per-agent content effects
range from about -10 pp (3B/14B GRPO) to +16 pp (v0.1 7B review, 3B PPO reflect).
Harm on stable states: review 26.9% vs P1 5.9% pooled, driven by 14B/3B GRPO; P1 is
not the matched control, so paper_h1 now adds the P2-matched harm difference.
ReSeek: nat - force_yes on S_natNo -2.4 [-3.9, -1.1], on all states -1.3 [-2.2,
-0.4]; judge No rate 21.8% with answer in the observation vs 56.9% without (detects,
but acting on it hurts). MR-Search S_wrong repair 7B reflect 3.3 / reask 1.9 /
neutral 1.7 / fresh 9.3; 3B 4.6 / 3.9 / 2.3 / 5.9; reflect - neutral +1.6 / +2.3
(significant), reflect - fresh -6.0 / -1.3; all-question net 7B ~0, 3B -2.5 pp.
HR2 GRPO sources: 18.2% failed (102 generation_limit, 7 malformed) -> stays invalid.
Decision: the paper states H1 as effect sizes against the matched placebo; the
verdict rule is unchanged. Details: research/U1_U2_HR2_RESULTS_AND_DECISION.md
section 6.

## 2026-10-28 — U1x results: H1 on 3/3 measurable non-Search-R1 recipes

Factory 9b5ea45, zero commits during the runs. Fidelity fine (format failures 0-2.4%;
paper EM gap AutoRefine -1.1 pp, StepSearch -3.0 pp). H1 confirmed on zs_llama32_3b,
zs_qwen25_7b and stepsearch_7b (explained 1.46-2.26). Preregistered generality claim
NOT met mechanically: 3/4 < ceil(0.8 x 4) = 4. autorefine_7b: review, reflect AND
both content-free P2 arms repair 0% and break 99-100% of stable states, while P1
(47.2% repair, 4.2% harm) and budget2 are normal -> inserting any text after the
observation breaks this agent; its False verdict is the 0/0 edge of the explained
fraction. Post hoc validity condition, labelled as post hoc in the paper: agents whose
P2 harm on stable states exceeds 50% are not measurable for insertion arms (only
AutoRefine triggers it; 14B GRPO's P2 is normal). With it, H1 holds on 3/3 measurable
U1x agents and 10/10 overall; both the mechanical 3/4 and the condition are reported.
No AutoRefine prompt adaptation (post hoc tuning); its brittleness is reported as a
finding. Content effects vary by agent (ZeroSearch-Llama +10.5 / +13.2 pp, D = 38).
Pooled 11 agents: review - P2 +1.8 [-1.0, 4.6], reflect - P2 +2.0 [-0.9, 5.1]; the
paper table pools the 10 measurable agents (paper_h1 --exclude, factory CPU step
docs/U1X_FOLLOWUP_GUIDE.md, with work2/arm_diagnose.py on AutoRefine). Experiments for
the main claims are complete; next is the paper outline. The user's GPU keep-alive
was stopped by the factory and left stopped for the user to decide.

## 2026-10-28 — 10-agent H1 table, matched harm, AutoRefine diagnosis; validity rule corrected

Factory 6944151, CPU only. 10 agents (AutoRefine excluded), 911 degraded states:
review - P2 +2.0 [-1.1, 4.9], reflect - P2 +2.2 [-1.2, 5.7]; vs P1 -19.8 / -16.9.
Harm on 8072 stable states, method - P2: review +8.6 [7.9, 9.4], reflect +9.2
[8.4, 10.0] pooled, dominated by GRPO models; per agent CI > 0 in 6/10 (review) and
8/10 (reflect), CI < 0 in 0/10, median about +2.1 / +3.4 (the factory's "three
reversed agents" is a misreading). Asymmetry for the paper: content adds no repair on
failure states but adds harm on stable states -> net content effect <= 0.
AutoRefine: after any inserted text it emits 1-2 tokens and stops (malformed) on
99-100% of stable states; budget2 is 100% normal; v03_7b_ppo answers >= 99% on all
arms. Correction: the 2026-10-28 claim that only AutoRefine crosses "P2 harm > 50%"
was wrong (14B GRPO P2 harm 52.7% / 51.6%). The post hoc validity rule is replaced,
before seeing any 14B diagnosis, by a format rule: an agent is insertion-valid if its
well-formed answer rate under content-free P2 on stable states is >= 50%
(work2/arm_diagnose.py VALIDITY_MIN_ANSWERED); harm with well-formed answers is a
real effect, not invalidity. paper_h1 gains --validity and an agent-level summary.
Main table = valid agents; sensitivity: 11, 10 and 9 (also without 14B) agents.
Factory CPU step: docs/H1_VALIDITY_GUIDE.md.
