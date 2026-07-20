#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required=(
  "data/linked/manifest.csv"
  "data/linked/embedding_index.csv"
  "data/linked/embeddings"
  "data/linked/progression_pairs.csv"
  "data/linked/horizon_matched_pairs.csv"
)

for relative_path in "${required[@]}"; do
  absolute_path="${repo_root}/${relative_path}"
  if [[ ! -e "${absolute_path}" ]]; then
    echo "Missing linked input: ${absolute_path}" >&2
    exit 1
  fi
  echo "OK ${relative_path} -> $(readlink -f "${absolute_path}")"
done
