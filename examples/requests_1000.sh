#!/usr/bin/env bash
# End-to-end demo on the 1000-photo batch:
#
#   1. create the batch (two 50-photo chains + 900 singletons)
#   2. add link p0000-p0050, merging the two chains into one 100-photo group
#      (every chain photo's previous nearest neighbour loses eligibility)
#   3. remove the same link, splitting the group again
#
# Both corrections must show ZERO newly computed squared distances because
# every per-photo ranking was built once at batch creation and is only
# re-filtered by group eligibility. The batch creation itself is counted once
# per unordered photo pair: 1000 * 999 / 2 = 499500.
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCH_ID="${BATCH_ID:-demo-1000}"

echo "== 1. Create 1000-photo batch (version 1) =="
# Use a unique batch id when re-running against an existing database.
python3 - "$HERE/create_batch_1000.json" "$BATCH_ID" "$HERE/create_batch_1000.request.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
payload["batch_id"] = sys.argv[2]
json.dump(payload, open(sys.argv[3], "w"))
PY
curl -sS -X POST "$BASE_URL/batches" \
  -H 'Content-Type: application/json' \
  --data @"$HERE/create_batch_1000.request.json" \
  | python3 -m json.tool > "$HERE/v1_1000_evaluation.json"
python3 - "$HERE/v1_1000_evaluation.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["photo_count"] == 1000, body["photo_count"]
# Two 50-photo groups + 900 singleton groups.
assert body["group_count"] == 902, body["group_count"]
assert body["version"] == 1
print("version:", body["version"],
      "photos:", body["photo_count"],
      "groups:", body["group_count"],
      "evaluable:", body["evaluable"],
      "overall_accuracy:", body["overall_accuracy"])
PY

echo
echo "== 2. Compute-stats after batch creation =="
curl -sS "$BASE_URL/batches/$BATCH_ID/compute-stats" \
  | python3 -m json.tool > "$HERE/v1_1000_stats.json"
python3 - "$HERE/v1_1000_stats.json" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
print("cache limit:", stats["cache_limit"],
      "cached ranking records (all batches):", stats["cached_ranking_records"],
      "for this batch:", stats["batch_cached_ranking_records"])
print("new squared-distance calculations:",
      stats["new_squared_distance_calculations"])
print("bulk pair calculations:", stats["bulk_pair_calculations"])
assert stats["bulk_pair_calculations"] == 1000 * 999 // 2
assert stats["new_squared_distance_calculations"] == 1000 * 999 // 2
assert stats["ranking_records_built"] == 1000
assert stats["batch_cached_ranking_records"] == 1000
assert stats["cached_ranking_records"] <= stats["cache_limit"]
PY

echo
echo "== 3. Snapshot v1 neighbours of chain photos =="
curl -sS "$BASE_URL/batches/$BATCH_ID/evaluation" \
  | python3 -m json.tool > "$HERE/v1_1000_evaluation.json"
python3 - "$HERE/v1_1000_evaluation.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
by_id = {p["photo_id"]: p for p in body["predictions"]}
chain = [f"p{i:04d}" for i in range(100)]
v1_neighbours = {pid: by_id[pid]["nearest_photo_id"] for pid in chain}
# Every chain photo's v1 nearest neighbour must be inside the *other* chain,
# so the upcoming merge disqualifies all 100 old nearest neighbours.
for pid in (f"p{i:04d}" for i in range(50)):
    assert "p0050" <= v1_neighbours[pid] <= "p0099", (pid, v1_neighbours[pid])
for pid in (f"p{i:04d}" for i in range(50, 100)):
    assert "p0000" <= v1_neighbours[pid] <= "p0049", (pid, v1_neighbours[pid])
print("all 100 chain photos have their v1 nearest neighbour in the other chain")
print("sample:", {pid: v1_neighbours[pid] for pid in ("p0000", "p0049", "p0050", "p0099")})
open("/tmp/v1_neighbours.json", "w").write(json.dumps(v1_neighbours))
PY

echo
echo "== 4. Correction: ADD p0000-p0050 (merge the two 50-photo groups) =="
curl -sS -X POST "$BASE_URL/batches/$BATCH_ID/corrections" \
  -H 'Content-Type: application/json' \
  --data '{"expected_version": 1, "action": "add", "photo_a": "p0000", "photo_b": "p0050"}' \
  | python3 -m json.tool > "$HERE/merge_1000_response.json"
python3 - "$HERE/merge_1000_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["version"] == 2
assert body["group_count"] == 901  # two groups became one
assert len(body["recomputed_photo_ids"]) == 100, body["recomputed_photo_ids"]
print("version:", body["previous_version"], "->", body["version"],
      "groups:", body["group_count"],
      "recomputed:", len(body["recomputed_photo_ids"]), "photos")
PY

curl -sS "$BASE_URL/batches/$BATCH_ID/compute-stats" \
  | python3 -m json.tool > "$HERE/merge_1000_stats.json"
python3 - "$HERE/merge_1000_stats.json" "$HERE/v1_1000_evaluation.json" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
event = stats["events"][-1]
print("merge new squared-distance calculations:",
      event["new_squared_distance_calculations"])
print("merge ranking records reused:", event["ranking_records_reused"],
      "rebuilt:", event["ranking_records_rebuilt"])
assert event["phase"] == "correction" and event["action"] == "add"
assert event["new_squared_distance_calculations"] == 0
assert event["ranking_records_reused"] == 100
assert event["ranking_records_rebuilt"] == 0
# Cumulative total must not have moved either.
assert stats["new_squared_distance_calculations"] == 1000 * 999 // 2
PY

echo
echo "== 5. Verify old nearest neighbours are now ineligible =="
curl -sS "$BASE_URL/batches/$BATCH_ID/evaluation" \
  | python3 -m json.tool > "$HERE/v2_1000_evaluation.json"
python3 - "$HERE/v2_1000_evaluation.json" /tmp/v1_neighbours.json <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
v1_neighbours = json.load(open(sys.argv[2]))
by_id = {p["photo_id"]: p for p in body["predictions"]}
disqualified = 0
for pid, old_nn in v1_neighbours.items():
    # The merged group contains exactly p0000..p0099.
    assert by_id[pid]["nearest_photo_id"] != old_nn, (pid, old_nn)
    assert by_id[pid]["nearest_photo_id"] >= "p0100", by_id[pid]
    disqualified += 1
print(f"{disqualified} chain photos lost their old nearest neighbour;")
print("each now reaches outside the merged 100-photo group, e.g.",
      {pid: by_id[pid]["nearest_photo_id"]
       for pid in ("p0000", "p0049", "p0050", "p0099")})
PY

echo
echo "== 6. Correction: REMOVE p0000-p0050 (split the group again) =="
curl -sS -X POST "$BASE_URL/batches/$BATCH_ID/corrections" \
  -H 'Content-Type: application/json' \
  --data '{"expected_version": 2, "action": "remove", "photo_a": "p0000", "photo_b": "p0050"}' \
  | python3 -m json.tool > "$HERE/split_1000_response.json"
python3 - "$HERE/split_1000_response.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
assert body["version"] == 3
assert body["group_count"] == 902
assert len(body["recomputed_photo_ids"]) == 100
print("version:", body["previous_version"], "->", body["version"],
      "groups:", body["group_count"],
      "recomputed:", len(body["recomputed_photo_ids"]), "photos")
PY

curl -sS "$BASE_URL/batches/$BATCH_ID/compute-stats" \
  | python3 -m json.tool > "$HERE/split_1000_stats.json"
python3 - "$HERE/split_1000_stats.json" <<'PY'
import json, sys
stats = json.load(open(sys.argv[1]))
event = stats["events"][-1]
print("split new squared-distance calculations:",
      event["new_squared_distance_calculations"])
print("split ranking records reused:", event["ranking_records_reused"],
      "rebuilt:", event["ranking_records_rebuilt"])
assert event["phase"] == "correction" and event["action"] == "remove"
assert event["new_squared_distance_calculations"] == 0
assert event["ranking_records_reused"] == 100
assert event["ranking_records_rebuilt"] == 0
assert stats["new_squared_distance_calculations"] == 1000 * 999 // 2
PY

echo
echo "== 7. Version 3 must equal version 1 exactly =="
curl -sS "$BASE_URL/batches/$BATCH_ID/evaluation" \
  | python3 -m json.tool > "$HERE/v3_1000_evaluation.json"
python3 - "$HERE/v1_1000_evaluation.json" "$HERE/v3_1000_evaluation.json" /tmp/v1_neighbours.json <<'PY'
import json, sys
v1 = json.load(open(sys.argv[1]))
v3 = json.load(open(sys.argv[2]))
v1_neighbours = json.load(open(sys.argv[3]))
assert v3["group_count"] == v1["group_count"] == 902
by_id = {p["photo_id"]: p for p in v3["predictions"]}
for pid, old_nn in v1_neighbours.items():
    assert by_id[pid]["nearest_photo_id"] == old_nn, pid
assert v3["overall_accuracy"] == v1["overall_accuracy"]
print("version 3 matches version 1: groups, neighbours and accuracy restored")
PY

echo
echo "ALL CHECKS PASSED"
