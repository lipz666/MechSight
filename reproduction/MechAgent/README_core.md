# Mech-Agent Planner-Executor

This repository contains a Planner-Executor agent for stepwise organic reaction
mechanism inference on PGL `ReactionState` graphs. The current implementation is
under `develop/`; historical design notes and experiment summaries are under
`doc/`.

## Current Design

The runtime uses two inference agents and one optional evaluator.

- Planner: compares the initial and target PGL states, infers the reaction
  family, writes a global mechanism summary, and proposes natural-language
  `elementary_steps`.
- Executor: receives only the current step and current PGL state, then proposes
  one concrete PGL `ReactionTransformation`.
- Planner review: checks the Executor proposal for global consistency, step
  consistency, and electron-transfer consistency before the environment applies
  the action.
- Environment: applies approved local graph edits, updates the current PGL
  state, and records successful actions.
- Completion check: verifies strict atom connectivity, bond order, formal
  charge, and explicit/nonzero H representation. A separate boundary-completion
  path can ignore omitted independent byproducts such as H2, but only when the
  runtime's own diff agrees that nothing remains on an atom shared with the
  target. An unformed bond on the principal product is a missing event, not an
  annotation boundary, and the runtime overrides the Planner's verdict.
- Post-hoc evaluator: after rollout termination, compares the initial Planner
  plan and accepted Executor actions against annotated GT transformations.

The latest default input mode is `structures_only`. The model sees only initial
and target PGL states; reaction names, reaction classes, conditions, and
class-bearing reaction IDs are hidden. The old `metadata` mode is retained only
for label-exposure ablations.

The current version deliberately does not use Planner skills, Executor skills,
or trajectory-mined skill updates.

## Deterministic Structural Evidence

`develop/state_diff.py` diffs the current and target PGL graphs and gives the
Planner an explicit list of every bond to form or break, every bond-order,
formal-charge and recorded-Hcount change, and every atom present on only one
side. Kekule alternations are separated out: a component of bond-order changes
in which no atom's total bond order moves is the same molecule redrawn, not an
outstanding event.

The list is supplied at planning, replanning, directive and completion time,
and it is roughly eight times shorter than the two PGL states it summarizes.
The reviewer additionally receives which outstanding changes the proposed
action touches. That is the only reference it has that it did not write itself;
without it the review approved every proposal it ever saw.

The list states what must change, never how many steps it takes, and never
which arrows to draw.

## Evaluation Fields

The post-hoc evaluator now reports three separate judgments.

- `planner_global_mechanism_judgment`: broad mechanism understanding. This asks
  whether the Planner inferred the correct reaction family, overall
  transformation, and major mechanistic stages.
- `planner_elementary_steps_judgment`: strict Planner step alignment. It
  normalizes the Planner's natural-language steps and GT transformations into
  ordered chemical events. Text-level splitting or merging is allowed only if no
  real event is added, omitted, duplicated, or reordered. Truly
  chemical-environment-equivalent atoms are allowed.
- `predicted_events_judgment`: strict comparison of accepted Executor
  `ReactionTransformation`s against the ordered GT electron-transfer events.

This split is important. Earlier evaluation used a broad Planner judgment, which
often marked a plan correct when it identified the reaction type but did not
match GT elementary-event order or step boundaries.

The LLM judgments are diagnostic only. On repeats that produced byte-identical
normalized action sequences, the evaluator changed its verdict on 12% of tasks,
so it cannot resolve differences smaller than that. Headline numbers come from
`develop/metrics.py` instead.

## Deterministic Metrics

`develop/evaluate_gt.py` scores a run against the annotated path and reports
four questions separately rather than as one number:

- `L1_transfer_set`: the predicted path moves exactly the annotated electrons,
  ignoring how they were grouped into steps and in what order they were emitted.
- `L2_transfer_sequence`: L1 plus the same causal order. Transfers inside one
  `ReactionTransformation` are concerted, so their order within a step is
  ignored.
- `L3_step_partition`: L2 plus the annotation's own step boundaries. This equals
  the previous `full_path_exact` metric; run with `--literal-h` to reproduce
  historical numbers exactly.
- transfer-level precision/recall/F1 at both the boundary-agnostic and the
  step-position-bound level.

Two normalizations are applied. Endpoint dialects are reconciled, because the
annotation writes `source='O205'` while the Executor is required to write
`source=('O205', 'lone_pair')`. Hydrogen labels are collapsed to `H*` by
default, because protons are indistinguishable and the annotated
`H900003`-style labels are generated rather than observed; `--literal-h` turns
this off.

Separating the levels matters because MechFinder emits a fixed step count per
reaction class, so step boundaries measure agreement with an annotation
convention rather than chemistry. The same SN2 event is annotated as a
one-step path for a halide and as step 4 of 5 for a tosylate.

Confidence intervals resample whole reactions, not rollouts. Repeats of one
reaction are not independent observations, so a 60-reaction design does not
become a 180-sample design by repeating each task three times.

## Compact Output

For large experiments, use `--analysis-output`. It appends one compact JSONL
record per completed rollout.

Each line has four top-level fields:

- `reaction`: reaction ID, repeat index, input mode, task text, initial/target
  PGL, success flag, and stop reason.
- `gt`: ordered GT `ReactionTransformation`s and the evaluator assessment.
- `planner_plan`: the initial Planner global plan and elementary steps.
- `actions`: every Executor proposal, including validation result, Planner
  decision, whether it entered the environment, edit validity, and whether it
  entered accepted history.

Full prompts are not saved. Trace files, outputs, model weights, and local
reference checkouts are ignored by `.gitignore`.

## Running

Start the local vLLM OpenAI-compatible server:

```bash
cd <path/to/MechAgent>
./develop/serve_vllm.sh
```

Run the current 60-class MechFinder evaluation, three rollouts per reaction:

```bash
mkdir -p outputs/v9

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
python -m develop.run_agent \
    --data-source core \
    --split all \
    --core-subset mechfinder \
    --mechfinder-num-classes 60 \
    --mechfinder-per-class 1 \
    --max-tasks 60 \
    --num-runs 60 \
    --repeats-per-task 3 \
    --task-input-mode structures_only \
    --planner-base-url http://127.0.0.1:8001/v1 \
    --executor-base-url http://127.0.0.1:8001/v1 \
    --planner-model gemma4-31b \
    --executor-model gemma4-31b \
    --planner-max-tokens 8192 \
    --executor-max-tokens 8192 \
    --max-planner-review-rounds 3 \
    --analysis-output outputs/v9/mechfinder60x3_structures_only.jsonl \
    > outputs/v9/mechfinder60x3_console.log 2>&1
```

Use `--task-input-mode metadata` only when explicitly testing the effect of
revealing reaction labels.

For a closed OpenAI-compatible API, replace the base URLs, models, and keys, and
add `--omit-thinking-extra-body` so vLLM/Gemma-specific thinking kwargs are not
sent.

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
PLANNER_API_KEY="$YOUR_API_KEY" \
EXECUTOR_API_KEY="$YOUR_API_KEY" \
GT_EVALUATOR_API_KEY="$YOUR_API_KEY" \
python -m develop.run_agent \
    --data-source core \
    --split all \
    --core-subset mechfinder \
    --mechfinder-num-classes 60 \
    --mechfinder-per-class 1 \
    --max-tasks 5 \
    --num-runs 5 \
    --repeats-per-task 1 \
    --task-input-mode structures_only \
    --planner-base-url https://api.example.com/v1 \
    --executor-base-url https://api.example.com/v1 \
    --gt-evaluator-base-url https://api.example.com/v1 \
    --planner-model CLOSED_MODEL_NAME \
    --executor-model CLOSED_MODEL_NAME \
    --gt-evaluator-model CLOSED_MODEL_NAME \
    --planner-max-tokens 4096 \
    --executor-max-tokens 4096 \
    --gt-evaluator-max-tokens 4096 \
    --omit-thinking-extra-body \
    --analysis-output outputs/closed_smoke.jsonl
```

## Current Results

Two recent experiment settings are documented in
`doc/recent_design_and_results_0816.md`.

Metadata mode, completed 180 rollouts:

- Environment success: 129/180 = 71.7%.
- Planner broad mechanism correct: 146/177 = 82.5%.
- Executor events correct: 52/177 = 29.4%.

Structures-only mode, current partial run with 149/180 saved rollouts:

- Environment success: 102/149 = 68.5%.
- Planner global correct: 128/147 = 87.1%.
- Planner strict steps correct: 45/147 = 30.6%.
- Executor events correct: 23/147 = 15.6%.

The structures-only result shows that the model can often infer the broad
reaction family from PGL alone, but strict step planning and executable
electron-transfer prediction remain much harder.

## Known Issues

- The broad Planner score overestimates stepwise correctness. The strict
  Planner score is the better metric for GT elementary-event alignment.
- Executor actions are the main bottleneck. Common failures include missed
  later steps, extra cleanup steps, incorrect proton/H handling, wrong bond
  order changes, and PGL electron-flow errors.
- `CONFIRM_COMPLETE_WITH_BOUNDARY_DIFFERENCE` is too permissive. It correctly
  handles omitted independent byproducts such as H2, but it also lets through
  some genuinely incomplete or wrong trajectories.
- Environment success does not imply mechanism correctness. Many successful
  trajectories reach the target state or boundary-completion condition while
  differing from GT event order.
- The runtime is slow with local Gemma 31B because every rollout uses many
  sequential Planner, Executor, review, completion, and evaluator calls, with
  thinking enabled and large token limits.

## Tests

Run local protocol and state-machine tests:

```bash
python -m pytest -q develop/tests
```

The latest test suite has 15 tests and does not require vLLM or GPUs.

## Repository Scope

The intended GitHub scope is:

- `develop/`
- `doc/`
- `README.md`
- `.gitignore`

Large outputs, traces, weights, checkpoints, benchmark snapshots, and reference
checkouts are ignored and should not be uploaded.
