# Work 2 development rules

This is an independent project rooted at this directory. The user explicitly
requested that Work 2 contain no GMST methods, training, experiments, or selected
artifacts. Do not modify the stopped research projects elsewhere in the workspace.

Active task (2026-10-19): AC1 confirmation of the interventional-audit claims on
`feat/credit-testbed`; read `research/AUDIT_CONFIRM_PROTOCOL.md` and `docs/AC1_FACTORY_GUIDE.md`
first. CT1 and PCR are archived NO-GO (their confirmed findings feed AC1). HR1 handoff replay
is NO-GO (2026-10-20). Current task: FC1 harness fidelity (`research/FIDELITY_PROTOCOL.md`,
`docs/FC1_FACTORY_GUIDE.md`). Factory: do not edit research/DECISION_LOG.md or research/STATE.json. Also read
`research/EXPERIMENT_PLAN.md`, `research/STATE.json`, and
`research/DECISION_LOG.md` before changing research code or proposing experiments.
Continue the recorded stage. Stage C requires evidence from stages A and B and
an explicit recorded GO decision; repository setup is not a research GO decision.

## Collaboration

- Develop, check, commit, and push code on the development server. The user has
  authorized pushing this public repository as part of the continuing workflow.
- The user pulls public HTTPS repositories and runs experiments on a 4×H800
  factory server. Never require the factory server to push, store GitHub write
  credentials, or run Codex.
- The factory already has a working Search-R1 environment, managed by the user.
  Provide a recommended configuration list only; do not configure/rebuild it or
  require an environment report before development. Environment inspection and
  feedback-packaging scripts are optional troubleshooting utilities.
- Keep paths configurable. Models, Wikipedia, indexes, outputs, and credentials
  are machine-local and must never be committed.
- Record the main repository commit, pinned dependency commits, configuration,
  and model/index versions with each future run. Do not invent results.
- Keep `third_party/` submodules pinned and unmodified by default. New experiment
  code belongs to this repository; any required dependency patch must be tracked
  here and documented so a fresh public clone reproduces the change.
- Preserve upstream notices and licenses. Do not assume all dependencies can be
  installed in one Python environment.
- Update state and decision log when a stage or research decision changes.
- Prefer CPU/static checks here; report H800 execution as pending until the user
  provides real results.
