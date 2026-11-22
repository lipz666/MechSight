#!/usr/bin/env bash
set -euo pipefail

# Restores the raw output directory tree and creates all symlinks expected by
# the frozen reproduction scripts under reproduction/MechAgent/ and figures/:
#   results/raw_outputs/pilots_v4_20261103/
#   results/raw_outputs/03_early_exploration_storyline_package/
#   reproduction/MechAgent/outputs/pilots_v4_20261103 -> results/raw_outputs/pilots_v4_20261103
#   reproduction/MechAgent/outputs/{fukuyama,heldout_fukuyama_20261024,p23_planning_probe_20261024,agent_v2_J4,benchmarks_20261103}/...

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCH="$ROOT/results/raw_outputs_archives"
DEST="$ROOT/results/raw_outputs/pilots_v4_20261103"
OUT_ROOT="$ROOT/reproduction/MechAgent/outputs"
S02_DATA="$ROOT/si_experiments/S02_benchmarks_and_splits/data"

echo "Verifying archive SHA-256 checksums..."
(cd "$ARCH" && shasum -a 256 -c SHA256SUMS.txt)

mkdir -p "$DEST"
for tgz in "$ARCH"/01_*.tar.gz "$ARCH"/02_*.tar.gz "$ARCH"/03_*.tar.gz "$ARCH"/04_*.tar.gz "$ARCH"/05_*.tar.gz; do
  [ -e "$tgz" ] || continue
  echo "Extracting $(basename "$tgz") -> results/raw_outputs/pilots_v4_20261103/"
  tar -xzf "$tgz" -C "$DEST"
done

# Copy canonical uncompressed analyses & top-level summaries into the restored pilots_v4_20261103 tree
cp -R "$ROOT/results/canonical_analyses/v4_analyses_20261108" "$DEST/"
cp -R "$ROOT/results/canonical_analyses/v3_analyses_20261107" "$DEST/"
cp -R "$ROOT/results/canonical_analyses/stepwise_20261108" "$DEST/"
cp -R "$ROOT/results/canonical_analyses/nodft_case_20261108" "$DEST/"
cp "$ROOT/results/canonical_analyses/pilots_v4_summaries/"* "$DEST/"

if [ -e "$ARCH/06_early_exploration_storyline.tar.gz" ]; then
  echo "Extracting 06_early_exploration_storyline.tar.gz -> results/raw_outputs/"
  tar -xzf "$ARCH/06_early_exploration_storyline.tar.gz" -C "$ROOT/results/raw_outputs/"
fi

# Wire symlinks inside reproduction/MechAgent/outputs/ so frozen scripts run out-of-the-box
mkdir -p \
  "$OUT_ROOT/fukuyama" \
  "$OUT_ROOT/heldout_fukuyama_20261024" \
  "$OUT_ROOT/p23_planning_probe_20261024" \
  "$OUT_ROOT/agent_v2_J4" \
  "$OUT_ROOT/benchmarks_20261103"

ln -sfn "$DEST" "$OUT_ROOT/pilots_v4_20261103"
ln -sfn "$S02_DATA/fukuyama_clean.jsonl" "$OUT_ROOT/fukuyama/clean.jsonl"
ln -sfn "$S02_DATA/fukuyama_heldout.jsonl" "$OUT_ROOT/heldout_fukuyama_20261024/heldout.jsonl"
ln -sfn "$S02_DATA/fukuyama_registration.json" "$OUT_ROOT/heldout_fukuyama_20261024/REGISTRATION.json"
ln -sfn "$S02_DATA/p23_cases.json" "$OUT_ROOT/p23_planning_probe_20261024/cases.json"
ln -sfn "$S02_DATA/ome_gold_pgl_cases.jsonl" "$OUT_ROOT/agent_v2_J4/cases.jsonl"
ln -sfn "$S02_DATA/fukuyama_splits.json" "$OUT_ROOT/benchmarks_20261103/splits.json"

echo "Done! Restored raw outputs to $DEST and wired symlinks in $OUT_ROOT"
