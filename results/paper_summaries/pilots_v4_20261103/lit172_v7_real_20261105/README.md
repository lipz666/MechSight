# 172-paper literature continuation (2026-11-05)

Scope: only the fixed 172 usable cases from `p20_lit150.cases()`; no USPTO jobs.
Auto uses the existing MechSight v5 pipeline and caches. Guided uses the real
literature observations for the 166 cases with clues. The six remaining cases
participate in Direct/Auto evaluation only.

The user selected real literature observations instead of simulated HRMS. Thus
this is a **literature adaptation of Guided v7**, not the frozen simulated-HRMS
experiment from P25/P27. It keeps four hypothesis samples, three votes,
RDKit checks, formula bookkeeping, Auto-first revision and one retry. Differences
and hashes are recorded in `MANIFEST.json`. No author-intermediate structures
are passed to prediction functions; author structures are used only in scoring.

In particular, a formula must not be invented from the author reference. Without
a measured formula, candidates are checked for valence, reactant atom budget and
endpoint novelty. Calculated formulas and decomposition hypotheses are supplied
to voting, and are labeled as computations rather than observations. Voting can
reject every candidate. Revision distinguishes a trapped adduct from its inferred
precursor. Both parsing failures and unavailable outputs remain in denominators
with zero recall, and availability/parsing coverage are reported separately.

Only the specified 117 gateway and `gemini-3.8-flash` are enabled. There is a
single worker pool with at most 300 workers, and sequential stage barriers.
Existing completed calls are reused. Credentials are inherited through the
environment and are not written into the source or run manifest.

Files:

- `status.json`: latest stage and supervisor process ID.
- `console.log`: append-only execution log.
- `MANIFEST.json`: frozen IDs, input/source hashes, model and method details.
- `SUMMARY_LIT172_V7_REAL.json`: generated on completion, with per-paper scores
  and paired bootstrap confidence intervals.
- `../lit150/G7L_*`: separate literature-v7 caches and API traces.

To resume, set `FALLBACK_API_KEY` in the environment and run from the project root:

```sh
FAILOVER_PREFER=117 FAILOVER_CAP_117=300 FAILOVER_CAP_VPS=0 \
  external/venv-llm/bin/python -u \
  MechAgent/experiments/pilots_v4/p28_lit172_v7.py --workers 300
```

An exclusive lock prevents duplicate supervisors for this run. Failed API jobs
are retried, and unresolved errors stop the dependent stages. A restart resumes
completed per-call caches. The source hash is checked on restart to prevent
mixing different methods in this run directory.
