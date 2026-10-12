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
