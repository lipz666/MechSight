# Planner–Executor PGL runtime

This directory implements the elementary-step Planner–Executor protocol in
[`doc/planner_executor_elementary_steps_v3.md`](../doc/planner_executor_elementary_steps_v3.md).
The earlier macro-step design remains available as historical context in
[`doc/planner_executor_design_0803_v2.md`](../doc/planner_executor_design_0803_v2.md).

The first version deliberately uses no Planner skills and no Executor skills.
Both agents call an OpenAI-compatible Chat Completions endpoint. Reasoning is
recorded separately from `message.content`; only content is parsed as the
`<plan>`, `<directive>`, `<review>`, and Executor protocol blocks.
Saved model events and JSONL traces contain response text, reasoning, usage,
and call metadata, but deliberately omit the full input messages.

## Components

- `client.py`: generic OpenAI-compatible client with Gemma thinking kwargs.
- `prompts.py`: shared PGL definition and role-specific message builders.
- `agents.py`: Planner and Executor protocol adapters.
- `orchestrator.py`: shared state and Planner review loop.
- `chemistry.py`: thin bridge to the existing SkillRL PGL environment.
- `run_agent.py`: CLI for core or extracted mechanism tasks.
- `serve_vllm.sh`: vLLM OpenAI server launcher.

## 1. Start the vLLM server

Use an environment where `vllm` supports the local Gemma 4 checkpoint:

```bash
cd <path/to/MechAgent>
./develop/serve_vllm.sh
```

Defaults:

```text
checkpoint: /path/to/model/gemma-4-31B-it
served model: gemma4-31b
URL: http://127.0.0.1:8000/v1
tensor parallel size: 4
max model length: 30000
max batched tokens: 8192
Planner/Executor max generated tokens per request: 8192
```

The checkpoint stores its chat template in `chat_template.jinja`, so the
launcher passes that file explicitly and enables vLLM's `gemma4` reasoning
parser. Override any server setting with the prefixed environment variables or
append additional `vllm serve` arguments:

```bash
MECH_TENSOR_PARALLEL_SIZE=8 \
MECH_SERVER_PORT=8010 \
./develop/serve_vllm.sh --dtype bfloat16
```

## 2. Run one mechanism

```bash
cd <path/to/MechAgent>
python -m develop.run_agent \
  --data-source core \
  --core-subset named_val \
  --num-runs 1 \
  --planner-base-url http://127.0.0.1:8000/v1 \
  --executor-base-url http://127.0.0.1:8000/v1 \
  --planner-model gemma4-31b \
  --executor-model gemma4-31b \
  --planner-trace outputs/planner_trace.jsonl \
  --executor-trace outputs/executor_trace.jsonl \
  --gt-evaluator-trace outputs/gt_evaluator_trace.jsonl \
  --output outputs/planner_executor_result.json
```

Every request uses normal OpenAI messages and, by default:

```python
response = client.chat.completions.create(
    model="gemma4-31b",
    messages=messages,
    temperature=0.2,
    max_tokens=8192,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True,
        }
    },
)
```

For a closed OpenAI-compatible provider, point the Planner and Executor flags
at its model, URL, and key. If it rejects vLLM-specific request fields, pass
`--omit-thinking-extra-body`.

Thinking is enabled by default. Do not pass `--disable-planner-thinking` or
`--disable-executor-thinking` when testing it. The protocol parser accepts the
requested tagged JSON and the single fenced-JSON form sometimes emitted by a
thinking-enabled Gemma template; malformed output still receives one repair
attempt with thinking disabled.

After each rollout terminates, one independent post-hoc model call compares
the initial Planner plan and accepted executed actions with annotated GT. The
GT is loaded only after inference and is never placed in Planner, Executor,
review, replan, or completion prompts. Disable this report with
`--disable-gt-assessment`. The report separates broad global-mechanism
understanding (`planner_global_mechanism_judgment`), strict normalized
Planner-step alignment (`planner_elementary_steps_judgment`), and executed
ReactionTransformation alignment (`predicted_events_judgment`).

Completion first uses an exact state comparison over atoms, connectivity,
bond order, formal charge, and recorded nonzero Hcount. If all planned steps
have finished and the principal product is correct, the Planner may separately
confirm an annotation-boundary difference for a participant or byproduct that
the target omits (for example H2 left after NaH deprotonation). Such a species
must not trigger a fabricated deletion, bond cleavage, or cleanup transfer.

Use `--repeats-per-task 3` for independent stability runs. Inference defaults
to `--task-input-mode structures_only`: reaction name, class, conditions, and
the class-bearing identifier are hidden, and the Planner must infer the family
from initial/target PGL. Use `--task-input-mode metadata` only for the explicit
label-exposure ablation.

For analysis without the large runtime state and model-event payloads, pass:

```bash
--analysis-output outputs/mechanism_analysis.jsonl
```

One compact JSON object is appended after every completed rollout and its
optional post-hoc evaluation. Its four top-level fields are
`reaction`, `gt`, `planner_plan`, and `actions`. `gt` contains the ordered
annotated ReactionTransformations and the structured evaluator assessment
(`null` when GT assessment is disabled); `planner_plan` is the initial global plan;
and every Executor proposal records its Planner decision, environment-execution
status, edit validity, and whether it entered accepted action history. This
JSONL therefore retains every rollout whose optional evaluation has completed
when a long multi-rollout process is interrupted. Evaluator parse failures are
stored as an error object instead of dropping the rollout.
If `--analysis-output` is used without `--output`, the full result payload is
not printed or retained; only a short final summary is printed.

## 3. Run tests

The protocol and state machine tests do not require GPUs, vLLM, or API calls:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q develop/tests
```

The state machine never executes a proposal unless deterministic validation
passes and the Planner returns `APPROVE`. A rejected environment edit is fed
back into the next directive but is not appended to successful action history.
The runtime owns the active `elementary_step`: one approved, valid, non-repeated
environment action completes exactly one step and advances to the next one.
The Planner can return `EXECUTE` or `REPLAN`, but cannot select, skip, merge, or
repeat plan steps through `STAY/ADVANCE` decisions.
If a model response is truncated or lacks its required protocol block, the
agent records the protocol error and retries once with thinking disabled.

The runtime reuses the existing chemistry environment from
`${SKILLRL_ROOT:-./external/SkillRL}` by default. Override it with
`--skillrl-root` when needed.
