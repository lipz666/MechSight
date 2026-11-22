#!/usr/bin/env bash
set -euo pipefail

# Restores the exact raw output directory tree expected by the reproduction scripts:
#   results/raw_outputs/pilots_v4_20261103/
#   results/raw_outputs/03_early_exploration_storyline_package/

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARCH="$ROOT/results/raw_outputs_archives"
DEST="$ROOT/results/raw_outputs/pilots_v4_20261103"

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

echo "Done! Restored raw outputs to $DEST"
