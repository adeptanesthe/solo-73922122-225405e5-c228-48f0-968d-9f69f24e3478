# Wildlife Camera Evaluation API

Same-session camera photos are near-duplicates of each other. Scoring them
with a plain leave-one-out nearest neighbour would let each photo "recognise"
its own burst and inflate accuracy. This service treats directly or indirectly
linked photos as one group and scores every photo only against photos
**outside** its group.

Stack: FastAPI, NumPy, SQLite. A batch holds 1-1000 photos; each photo has a
client-supplied ID, a known species, and 32 signed 16-bit integer features.
After a batch exists, the only mutation is adding/removing one photo link.

## Run with Docker Compose

```bash
docker compose up --build
# docs: http://127.0.0.1:8000/docs
```

The SQLite file lives on the `eval-data` volume at
`/data/evaluations.sqlite3`. Override with the `EVALUATION_DB` env var.
The in-process ranking cache keeps at most `DISTANCE_CACHE_LIMIT` records
(default 32000; set to `0` to disable caching and always rebuild).

Without Docker:

```bash
pip install -r requirements.txt
EVALUATION_DB=data/evaluations.sqlite3 uvicorn app.main:app --port 8000
```

## API

### `POST /batches` → 201

```json
{
  "batch_id": "demo-batch",          // optional; generated when omitted
  "photos": [{"photo_id": "p01", "species": "deer", "features": [0, 0, "... 32 int16 values"]}],
  "pairs":  [{"photo_a": "p01", "photo_b": "p02"}]
}
```

Returns the version-1 evaluation. Validation (all 422): 1-1000 photos,
unique photo IDs, exactly 32 strict integers in `[-32768, 32767]`, pair
endpoints must exist in the batch, duplicate/self pairs rejected. Reusing a
`batch_id` returns 409.

### `GET /batches/{batch_id}/evaluation` → 200

Current committed version: `version`, `groups`, every photo's prediction,
per-group accuracy, and `overall_accuracy` (equal-weight mean of group
accuracies). With a single group the response is `evaluable: false`,
`overall_accuracy: null`, `predictions: []`, and an explanatory `message`.

### `POST /batches/{batch_id}/corrections` → 200

```json
{"expected_version": 1, "action": "add", "photo_a": "p01", "photo_b": "p03"}
```

- `expected_version` is the reviewer's current version; a stale value returns
  **409 `version_conflict`** and the old version stays fully intact.
- `add` of an existing link / `remove` of a missing link returns 409;
  unknown endpoints return 422.
- Only photos whose same-group membership changed are re-predicted; the list
  is returned as `recomputed_photo_ids`. Adding a cycle edge inside one
  component changes nobody (empty list). Ending up with one group flips the
  batch to `evaluable: false` and drops predictions.
- Grouping, predictions, and the overall score switch versions atomically in
  one `BEGIN IMMEDIATE` transaction. Concurrent corrections against the same
  version: exactly one commits, the rest get 409.
- Reads use SQLite WAL snapshots, so `GET .../evaluation` keeps serving the
  old version while a correction is being computed and switches only on commit.

Predictions: nearest Euclidean distance over photos outside the group, using
exact integer squared distances; ties are broken by the smallest photo ID.

### `GET /batches/{batch_id}/compute-stats` → 200

Distance-work counters for one batch, plus the process-wide cache footprint:

- `new_squared_distance_calculations`: cumulative number of squared distances
  actually evaluated (bulk batch creation is counted **once per unordered
  photo pair**, so a 1000-photo batch reports `1000*999/2 = 499500`).
- `bulk_pair_calculations`, `ranking_records_built`,
  `ranking_records_rebuilt`, `ranking_records_reused`: how the work split
  between batch warming, on-demand rebuilds, and cache hits.
- `cache_limit`, `cached_ranking_records`: the process cap (default 32000)
  and the current number of retained records across all batches.
- `events`: one entry per batch creation / committed correction, each with
  its own counters. Failed corrections roll back and append no event.

Counters live in process memory and reset on restart; predictions themselves
are stored in SQLite and stay exact after a restart.

## Distance ranking cache

Features are fixed once a batch is created, so for every photo the service
builds one full ranking of all other photos by exact squared distance
(secondary sort key: photo index, which implements the smallest-ID tie
break). Corrections never change a ranking — they only change which entries
are group-eligible:

- **Merge:** the old nearest neighbours that now belong to the merged group
  are skipped, and the search continues down the same cached ranking.
- **Split:** photos that regain out-of-group eligibility re-enter the
  candidate set through the same cached ranking; no distance is recomputed.

The cache is a single-process, thread-safe LRU holding at most
`DISTANCE_CACHE_LIMIT` records (default **32000**, one record is one photo's
complete ranking). When a record is missing (evicted or never built in this
process, e.g. after a restart) it is rebuilt from the batch features stored
in SQLite, so every answer remains exact; only the work counters increase.

## Demonstration (before/after a correction)

```bash
BASE_URL=http://127.0.0.1:8000 bash examples/requests.sh
```

`examples/create_batch.json` has 10 photos in 6 groups (two same-session pairs
at each of two sites plus two singletons). Version 1 scores
`4/6 = 0.6667`; adding `p01-p03` merges two deer pairs into one 4-photo group,
recomputes exactly `p01,p02,p03,p04`, and the score drops to `2/5 = 0.4`.
The script also shows a stale version-1 correction rejected with 409.
`examples/requests.sh` writes the raw responses next to the payloads.

## 1000-photo merge/split demonstration

`examples/create_batch_1000.json` (regenerate with
`python3 examples/generate_batch_1000.py`) contains 1000 photos: two 50-photo
chains (`p0000..p0049`, `p0050..p0099`) placed in one tight feature cluster,
plus 900 far-away singletons. At version 1 every chain photo's nearest
out-of-group neighbour is in the *other* chain.

```bash
BASE_URL=http://127.0.0.1:8000 bash examples/requests_1000.sh
```

The script:

1. creates the batch — bulk work counted as `499500` unordered photo pairs,
   and all 1000 ranking records are cached;
2. adds `p0000-p0050`, merging the two 50-photo groups: all 100 chain photos
   lose their previous nearest neighbour and are re-predicted against photos
   outside the new 100-photo group, with **zero** new squared-distance
   calculations (100 records reused);
3. removes the same link, splitting the group so the cross-chain neighbours
   become eligible again — again **zero** new calculations;
4. asserts version 3 reproduces version 1 exactly.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest -q
```

32 tests cover ingestion/validation, the 1000-photo limit, grouping, exact tie
breaking, brute-force cross-checks, incremental recomputation, version
conflicts, concurrent writers, rollback on failure, reads during an
in-flight correction, ranking-cache reuse on merge/split, LRU eviction and
on-demand rebuild, empty-cache restart exactness, the global 32000-record
cap, compute-stats accounting, orphan-record cleanup on failed creation, and
rollback of work counters on failure.
