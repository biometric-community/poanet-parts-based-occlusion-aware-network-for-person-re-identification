#!/usr/bin/env bash
# Measure required dataset roots vs the paper-2-model 5 GiB full-train gate.
# Usage:
#   bash scripts/check_dataset_size.sh [path ...]
#   DATA_ROOTS="path1 path2" bash scripts/check_dataset_size.sh
# Exit 0 = full_train (< 5 GiB); exit 3 = skip (>= 5 GiB); exit 1/2 = error.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
THRESHOLD=$((5 * 1024 * 1024 * 1024))
OUT_DIR="${ROOT}/outputs/logs"
mkdir -p "$OUT_DIR"
OUT_JSON="${OUT_DIR}/dataset_size.json"

resolve_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
    return
  fi
  if [[ -e "${ROOT}/${p}" ]]; then
    printf '%s\n' "${ROOT}/${p}"
    return
  fi
  # common monorepo relative from projects/papers/<slug>/: ../../datasets/...
  local normalized
  normalized="$(cd "${ROOT}" && realpath -m "$p" 2>/dev/null || true)"
  if [[ -n "${normalized}" && -e "${normalized}" ]]; then
    printf '%s\n' "${normalized}"
    return
  fi
  printf '%s\n' "${ROOT}/${p}"
}

paths=("$@")
if [[ ${#paths[@]} -eq 0 ]]; then
  if [[ -n "${DATA_ROOTS:-}" ]]; then
    # shellcheck disable=SC2206
    paths=($DATA_ROOTS)
  elif [[ -f configs/default.yaml ]]; then
    mapfile -t paths < <(python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
root = cfg.get("data", {}).get("root")
if root:
    print(root)
PY
)
  fi
fi

if [[ ${#paths[@]} -eq 0 || -z "${paths[0]:-}" ]]; then
  echo "No dataset paths given. Pass paths or set data.root / DATA_ROOTS." >&2
  exit 2
fi

resolved=()
for p in "${paths[@]}"; do
  rp="$(resolve_path "$p")"
  if [[ ! -e "$rp" ]]; then
    echo "Missing dataset path: $rp (from $p)" >&2
    exit 1
  fi
  resolved+=("$rp")
done

python3 - "$OUT_JSON" "$THRESHOLD" "${resolved[@]}" <<'PY'
import json, subprocess, sys
out_json, threshold = sys.argv[1], int(sys.argv[2])
roots = sys.argv[3:]
items = []
total = 0
for path in roots:
    bytes_ = int(subprocess.check_output(["du", "-sb", path], text=True).split()[0])
    items.append({"path": path, "bytes": bytes_})
    total += bytes_
decision = "full_train" if total < threshold else "skip_full_train_size_gte_5gib"
payload = {
    "threshold_bytes": threshold,
    "threshold_label": "5 GiB",
    "total_bytes": total,
    "total_gib": round(total / (1024 ** 3), 4),
    "decision": decision,
    "roots": items,
}
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"Wrote {out_json}")
print(f"total_bytes={total} total_gib={payload['total_gib']} decision={decision}")
sys.exit(0 if decision == "full_train" else 3)
PY
