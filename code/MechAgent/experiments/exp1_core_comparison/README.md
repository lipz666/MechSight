# Experiment 1: direct reasoning vs executable agent vs agent with skills

The preparation step freezes 80 new MechFinder records across 40 known families,
their endpoint-only uniform-explicit-H inputs, a separate GT file, and the actual
local Agent/79-skill code snapshot. This is sample-held-out evaluation, not an
unseen-family benchmark or a claim of absence of pretraining contamination.

See the run directory's `PROTOCOL.md` before interpreting scores. In particular,
the proxy reports thinking separately and can exceed the per-request max_tokens;
the shared 98304-generation-token threshold stops subsequent calls, rather than
guaranteeing exactly equal cost. Actual usage and overshoots must be reported.

Install `openai`, `rdkit`, and `pytest` in a virtual environment. In this run the
environment is `/tmp/mech-exp1-env`; do not assume it persists after reboot.

From the repository root:

```sh
python experiments/exp1_core_comparison/prepare.py \
  --output outputs/publication_exp1_20261018
```

The frozen dataset already exists; `prepare.py` refuses to overwrite it. Supply
`OPENAI_API_KEY` via the environment (never put credentials into source files).

```sh
python -u experiments/exp1_core_comparison/run.py \
  --output outputs/publication_exp1_20261018 --workers 2
python experiments/exp1_core_comparison/summarize.py \
  --output outputs/publication_exp1_20261018
```

Results are written atomically per arm/case/repeat and completed jobs are skipped
on resume. Existing transport-failed smoke runs are retained separately, rather
than being presented as scientific results. Do not mix run configurations or
silently count repeated infrastructure attempts as independent rollouts.

```sh
python -m pytest -q experiments/exp1_core_comparison/test_experiment.py
python -m unittest discover -s develop-2/tests -q
```

No LLM GT judge is called, and inference code never opens `gold.jsonl`. Post-hoc
scoring restores blinded heavy-atom labels and uses the existing arrow metrics.
The raw proposed baseline path, self-review versions, rejected agent proposals,
usage, reasoning, and response model identifiers are retained for audit.
