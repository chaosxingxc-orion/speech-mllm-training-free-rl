#!/usr/bin/env bash
set -euo pipefail
# Data root — override with SPEECHRL_DATA_DIR (e.g. the umbrella's speechrl-data on /mnt/d,
# or an ext4 copy under $HOME). Matches docs/data.md's ${SPEECHRL_DATA_DIR:-...} convention.
DATA="${SPEECHRL_DATA_DIR:-$HOME/speechrl-data}"

echo "=== MODELS ==="
for d in "$DATA/models"/*/; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  count=$(find "$d" -type f ! -name '.*' 2>/dev/null | wc -l)
  size=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf '  %-40s %5s files  %6s\n' "$name" "$count" "$size"
done

echo ""
echo "=== DATASETS ==="
for d in "$DATA/datasets"/*/; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  count=$(find "$d" -type f ! -name '.*' 2>/dev/null | wc -l)
  size=$(du -sh "$d" 2>/dev/null | cut -f1)
  printf '  %-40s %5s files  %6s\n' "$name" "$count" "$size"
done

echo ""
echo "=== REPOS ==="
for d in "$DATA/repos"/*/; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  has_git='N'
  [[ -d "$d/.git" ]] && has_git='Y'
  printf '  %-40s git=%s\n' "$name" "$has_git"
done

echo ""
echo "=== DISK ==="
df -h "$DATA" 2>/dev/null || true
