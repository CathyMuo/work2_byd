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

## Stage A1 — clean baseline (current; runner implemented, factory execution pending)

- Prefer a Search-R1-trained Qwen2.5-3B checkpoint. Compare to untrained
  Qwen2.5-3B-Instruct later; do not assume the two have equivalent tool behavior.
- Freeze E5-base-v2, Wiki-18, FAISS HNSW, Top-3 visible passages, maximum 4
  search turns, 8192 context tokens, 512 generated tokens per turn initially.
- Verify tool-call validity, multi-turn frequency, EM/F1, context truncation,
  search counts, and termination. A lack of usable multi-turn traces must be
  diagnosed before interpreting perturbation effects.
- Diagnose corpus-missing facts, retriever misses, and agent evidence misuse
  separately. Title equality alone cannot establish supporting-fact recall.

Required output: `baseline_metrics.json`, clean trajectories, and immutable run
metadata. No baseline result exists at initialization.

Implementation and factory commands are documented in `docs/EXECUTION_GUIDE.md`.
Use `work2.prepare_pilot`, `work2.launch_retriever` (or an existing service),
and `work2.run_baseline`. Run a 10-question smoke subset before all 200 frozen
pilot questions. Dependencies/resources are user-managed; no model or dataset
download is performed. A1 emits E0 only. A2/A3 code and their scientific gates
are not passed by this implementation update.

## Stage A2 — paired replay sanity

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

## Stage A3 — diagnostic pilot

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
