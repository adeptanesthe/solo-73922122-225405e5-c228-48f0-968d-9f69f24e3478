"""Distance-ranking cache: reuse, LRU eviction, restart safety, stats API."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from app.database import initialize
from app.distance_cache import DEFAULT_CACHE_LIMIT, distance_cache
from app.grouping import build_groups, group_index
from app.models import BatchCreateRequest, CorrectionRequest, PhotoIn
from app.predictions import predict_nearest_outside, predict_nearest_outside_cached
from app.service import apply_correction, create_batch, load_evaluation

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples" / "create_batch.json").read_text())


@pytest.fixture(autouse=True)
def _reset_cache():
    # Every test starts from an empty, default-sized process cache, exactly
    # like a freshly started service process; the default limit is restored
    # afterwards even when a test shrank it.
    distance_cache.reset()
    yield
    distance_cache.reset()


def _random_batch(batch_id: str, size: int, seed: int) -> tuple[
    BatchCreateRequest, list[str], dict[str, str], np.ndarray
]:
    rng = random.Random(seed)
    photo_ids = [f"q{i:03d}" for i in range(size)]
    photos = [
        PhotoIn(
            photo_id=photo_id,
            species=rng.choice(["a", "b", "c"]),
            features=[rng.randint(-4000, 4000) for _ in range(32)],
        )
        for photo_id in photo_ids
    ]
    request = BatchCreateRequest.model_validate(
        {"batch_id": batch_id, "photos": [p.model_dump() for p in photos], "pairs": []}
    )
    features = np.asarray([p.features for p in photos], dtype=np.int64)
    species = {p.photo_id: p.species for p in photos}
    return request, photo_ids, species, features


def test_batch_creation_counts_pairs_once_and_caches_all_rankings(tmp_path):
    db_path = str(tmp_path / "cache.sqlite3")
    initialize(db_path)
    request, photo_ids, _, _ = _random_batch("warm", 20, seed=1)
    create_batch(db_path, request)

    stats = distance_cache.stats_snapshot("warm")
    n = 20
    assert stats["bulk_pair_calculations"] == n * (n - 1) // 2
    assert stats["new_squared_distance_calculations"] == n * (n - 1) // 2
    assert stats["ranking_records_built"] == n
    assert stats["batch_cached_ranking_records"] == n
    assert stats["cached_ranking_records"] >= n
    assert stats["cache_limit"] == DEFAULT_CACHE_LIMIT
    assert stats["events"][0]["phase"] == "batch_created"


def test_merge_correction_reuses_rankings_with_zero_new_calculations(tmp_path):
    db_path = str(tmp_path / "cache.sqlite3")
    initialize(db_path)
    request, photo_ids, species, features = _random_batch("merge", 30, seed=2)
    # Two 10-photo groups plus ten singletons.
    pair_dicts = [
        {"photo_a": f"q{i:03d}", "photo_b": f"q{i + 1:03d}"} for i in range(0, 9)
    ] + [
        {"photo_a": f"q{i:03d}", "photo_b": f"q{i + 1:03d}"} for i in range(10, 19)
    ]
    request = BatchCreateRequest.model_validate(
        {**request.model_dump(), "pairs": pair_dicts}
    )
    created = create_batch(db_path, request)
    assert created["group_count"] == 12

    response = apply_correction(
        db_path,
        "merge",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="q000", photo_b="q010"
        ),
    )
    assert set(response["recomputed_photo_ids"]) == {f"q{i:03d}" for i in range(20)}

    stats = distance_cache.stats_snapshot("merge")
    correction_events = [e for e in stats["events"] if e["phase"] == "correction"]
    assert len(correction_events) == 1
    event = correction_events[0]
    assert event["new_squared_distance_calculations"] == 0
    assert event["ranking_records_reused"] == 20
    assert event["ranking_records_rebuilt"] == 0

    # Cached merge result is identical to a from-scratch brute-force oracle.
    edges = {
        (min(p.photo_a, p.photo_b), max(p.photo_a, p.photo_b))
        for p in request.pairs
    } | {("q000", "q010")}
    groups = build_groups(photo_ids, edges)
    oracle = predict_nearest_outside(
        photo_ids, species, features, group_index(groups), photo_ids
    )
    actual = {p["photo_id"]: p for p in load_evaluation(db_path, "merge")["predictions"]}
    for photo_id in photo_ids:
        assert actual[photo_id]["nearest_photo_id"] == oracle[photo_id]["nearest_photo_id"]
        assert actual[photo_id]["distance"] == pytest.approx(oracle[photo_id]["distance"])


def test_split_correction_re_includes_newly_eligible_photos_with_zero_new_work(tmp_path):
    db_path = str(tmp_path / "cache.sqlite3")
    initialize(db_path)
    request, photo_ids, species, features = _random_batch("split", 24, seed=3)
    # One chain q000..q009 and a separate chain q010..q019, four singletons.
    pair_dicts = [
        {"photo_a": f"q{i:03d}", "photo_b": f"q{i + 1:03d}"} for i in range(0, 9)
    ] + [
        {"photo_a": f"q{i:03d}", "photo_b": f"q{i + 1:03d}"} for i in range(10, 19)
    ]
    request = BatchCreateRequest.model_validate(
        {**request.model_dump(), "pairs": pair_dicts}
    )
    create_batch(db_path, request)
    apply_correction(
        db_path,
        "split",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="q000", photo_b="q010"
        ),
    )

    # Removing the bridge splits the 20-photo group again: photos in the two
    # halves regain cross-group eligibility and must be compared once more.
    response = apply_correction(
        db_path,
        "split",
        CorrectionRequest(
            expected_version=2, action="remove", photo_a="q000", photo_b="q010"
        ),
    )
    assert response["version"] == 3
    assert set(response["recomputed_photo_ids"]) == {f"q{i:03d}" for i in range(20)}

    events = [e for e in distance_cache.stats_snapshot("split")["events"]
              if e["phase"] == "correction"]
    assert events[-1]["new_squared_distance_calculations"] == 0
    assert events[-1]["ranking_records_reused"] == 20

    groups = build_groups(
        photo_ids,
        {
            (min(p.photo_a, p.photo_b), max(p.photo_a, p.photo_b))
            for p in request.pairs
        },
    )
    oracle = predict_nearest_outside(
        photo_ids, species, features, group_index(groups), photo_ids
    )
    actual = {p["photo_id"]: p for p in load_evaluation(db_path, "split")["predictions"]}
    for photo_id in photo_ids:
        assert actual[photo_id] == oracle[photo_id]


def test_eviction_rebuilds_rankings_and_keeps_exact_results(tmp_path):
    distance_cache.reset(limit=4)
    db_path = str(tmp_path / "evict.sqlite3")
    initialize(db_path)
    request, photo_ids, species, features = _random_batch("evict", 18, seed=4)
    create_batch(db_path, request)
    assert distance_cache.cached_count() == 4  # LRU cap applied during warming

    response = apply_correction(
        db_path,
        "evict",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="q000", photo_b="q001"
        ),
    )
    changed = response["recomputed_photo_ids"]
    assert changed == ["q000", "q001"]

    event = [
        e for e in distance_cache.stats_snapshot("evict")["events"]
        if e["phase"] == "correction"
    ][0]
    # Every ranking had been evicted: both targets are rebuilt from features.
    assert event["ranking_records_rebuilt"] == 2
    assert event["new_squared_distance_calculations"] == 2 * (len(photo_ids) - 1)

    groups = build_groups(photo_ids, {("q000", "q001")})
    oracle = predict_nearest_outside(
        photo_ids, species, features, group_index(groups), photo_ids
    )
    actual = {p["photo_id"]: p for p in load_evaluation(db_path, "evict")["predictions"]}
    for photo_id in photo_ids:
        assert actual[photo_id]["nearest_photo_id"] == oracle[photo_id]["nearest_photo_id"]
        assert actual[photo_id]["distance"] == pytest.approx(oracle[photo_id]["distance"])


def test_restart_with_empty_cache_still_produces_exact_predictions(tmp_path):
    db_path = str(tmp_path / "restart.sqlite3")
    initialize(db_path)
    request, photo_ids, species, features = _random_batch("restart", 22, seed=5)
    create_batch(db_path, request)

    # Simulate a service restart: the in-process cache and counters are gone,
    # while SQLite keeps the committed batch state.
    distance_cache.reset()
    assert distance_cache.cached_count() == 0

    response = apply_correction(
        db_path,
        "restart",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="q002", photo_b="q003"
        ),
    )
    assert response["recomputed_photo_ids"] == ["q002", "q003"]
    event = distance_cache.stats_snapshot("restart")["events"][0]
    assert event["phase"] == "correction"
    assert event["ranking_records_rebuilt"] == 2
    assert event["new_squared_distance_calculations"] == 2 * (len(photo_ids) - 1)

    groups = build_groups(photo_ids, {("q002", "q003")})
    oracle = predict_nearest_outside(
        photo_ids, species, features, group_index(groups), photo_ids
    )
    actual = {
        p["photo_id"]: p
        for p in load_evaluation(db_path, "restart")["predictions"]
    }
    for photo_id in photo_ids:
        assert actual[photo_id]["nearest_photo_id"] == oracle[photo_id]["nearest_photo_id"]


def test_lru_cap_is_global_across_batches(tmp_path):
    distance_cache.reset(limit=10)
    db_path = str(tmp_path / "global.sqlite3")
    initialize(db_path)
    request_a, _, _, _ = _random_batch("alpha", 8, seed=6)
    request_b, _, _, _ = _random_batch("beta", 8, seed=7)
    create_batch(db_path, request_a)
    create_batch(db_path, request_b)
    assert distance_cache.cached_count() == 10

    stats_a = distance_cache.stats_snapshot("alpha")
    stats_b = distance_cache.stats_snapshot("beta")
    assert stats_a["ranking_records_built"] == 8
    assert stats_b["ranking_records_built"] == 8


def test_failed_batch_creation_leaves_no_cached_records(tmp_path, monkeypatch):
    import sqlite3

    from app.service import ServiceError, create_batch

    db_path = str(tmp_path / "orphan.sqlite3")
    initialize(db_path)
    request, _, _, _ = _random_batch("orphan", 16, seed=9)
    create_batch(db_path, request)
    assert distance_cache.cached_count() == 16

    # Same id again: rejected with 409 (pre-check), nothing new was warmed.
    with pytest.raises(ServiceError) as exc:
        create_batch(db_path, request)
    assert exc.value.status_code == 409
    assert distance_cache.cached_count() == 16

    # Force the in-transaction race path: the INSERT INTO batches fails with an
    # integrity error after cache warming has already happened.
    import app.service as service

    real_connect = service.connect

    class FailingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].lstrip().startswith(
                "INSERT INTO batches"
            ):
                raise sqlite3.IntegrityError(
                    "UNIQUE constraint failed: batches.batch_id"
                )
            return self._inner.execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def close(self):
            self._inner.close()

    def flaky_connect(path):
        return FailingConnection(real_connect(path))

    monkeypatch.setattr(service, "connect", flaky_connect)
    request2, _, _, _ = _random_batch("race-orphan", 14, seed=10)
    with pytest.raises(ServiceError) as exc:
        create_batch(db_path, request2)
    assert exc.value.status_code == 409
    monkeypatch.undo()

    # The warmed rankings for the never-committed batch are fully cleaned up.
    assert distance_cache.cached_count() == 16
    orphan_stats = distance_cache.stats_snapshot("race-orphan")
    assert orphan_stats["events"] == []
    assert orphan_stats["ranking_records_built"] == 0


def test_failed_correction_does_not_change_counters(tmp_path, monkeypatch):
    db_path = str(tmp_path / "fail.sqlite3")
    initialize(db_path)
    payload = dict(EXAMPLE)
    payload["batch_id"] = "rollback-stats"
    request = BatchCreateRequest.model_validate(payload)
    create_batch(db_path, request)
    before = distance_cache.stats_snapshot("rollback-stats")

    import app.service as service

    def explode(*args, **kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(service, "_upsert_predictions", explode)
    with pytest.raises(RuntimeError):
        apply_correction(
            db_path,
            "rollback-stats",
            CorrectionRequest(
                expected_version=1, action="add", photo_a="p01", photo_b="p03"
            ),
        )
    after = distance_cache.stats_snapshot("rollback-stats")
    assert after["events"] == before["events"]
    for field in (
        "new_squared_distance_calculations",
        "bulk_pair_calculations",
        "ranking_records_built",
        "ranking_records_rebuilt",
        "ranking_records_reused",
    ):
        assert after[field] == before[field]


def test_compute_stats_endpoint_and_zero_work_corrections(client):
    response = client.post("/batches", json=EXAMPLE)
    assert response.status_code == 201
    batch_id = EXAMPLE["batch_id"]

    stats = client.get(f"/batches/{batch_id}/compute-stats").json()
    n = len(EXAMPLE["photos"])
    assert stats["cache_limit"] == DEFAULT_CACHE_LIMIT
    assert stats["new_squared_distance_calculations"] == n * (n - 1) // 2
    assert stats["bulk_pair_calculations"] == n * (n - 1) // 2
    assert stats["ranking_records_built"] == n
    assert stats["batch_cached_ranking_records"] == n
    assert stats["cached_ranking_records"] == n  # fixture resets the cache
    assert len(stats["events"]) == 1

    correction = client.post(
        f"/batches/{batch_id}/corrections",
        json={
            "expected_version": 1,
            "action": "add",
            "photo_a": "p01",
            "photo_b": "p03",
        },
    )
    assert correction.status_code == 200
    stats = client.get(f"/batches/{batch_id}/compute-stats").json()
    assert stats["events"][-1]["new_squared_distance_calculations"] == 0
    assert stats["events"][-1]["ranking_records_reused"] == 4

    assert client.get("/batches/missing/compute-stats").status_code == 404


def test_1000_photo_batch_merge_and_split_need_zero_new_calculations(tmp_path):
    """Contract behind examples/requests_1000.sh, at the service layer."""
    payload_path = ROOT / "examples" / "create_batch_1000.json"
    payload = json.loads(payload_path.read_text())
    payload["batch_id"] = "pytest-1000"
    request = BatchCreateRequest.model_validate(payload)
    db_path = str(tmp_path / "thousand.sqlite3")
    initialize(db_path)

    created = create_batch(db_path, request)
    assert created["photo_count"] == 1000
    # Two 50-photo chains plus 900 singletons.
    assert created["group_count"] == 902

    stats = distance_cache.stats_snapshot("pytest-1000")
    pair_count = 1000 * 999 // 2
    assert stats["bulk_pair_calculations"] == pair_count
    assert stats["new_squared_distance_calculations"] == pair_count
    assert stats["ranking_records_built"] == 1000
    assert distance_cache.cached_count() == 1000

    by_id = {p["photo_id"]: p for p in created["predictions"]}
    # Every chain photo's v1 nearest neighbour lives in the other chain.
    for index in range(50):
        assert "p0050" <= by_id[f"p{index:04d}"]["nearest_photo_id"] <= "p0099"
    for index in range(50, 100):
        assert "p0000" <= by_id[f"p{index:04d}"]["nearest_photo_id"] <= "p0049"

    merged = apply_correction(
        db_path,
        "pytest-1000",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="p0000", photo_b="p0050"
        ),
    )
    assert merged["group_count"] == 901
    assert len(merged["recomputed_photo_ids"]) == 100
    merge_event = distance_cache.stats_snapshot("pytest-1000")["events"][-1]
    assert merge_event["new_squared_distance_calculations"] == 0
    assert merge_event["ranking_records_reused"] == 100
    # All 100 old nearest neighbours are now inside the merged group.
    v2 = {p["photo_id"]: p for p in load_evaluation(db_path, "pytest-1000")["predictions"]}
    for index in range(100):
        assert v2[f"p{index:04d}"]["nearest_photo_id"] >= "p0100"

    split = apply_correction(
        db_path,
        "pytest-1000",
        CorrectionRequest(
            expected_version=2, action="remove", photo_a="p0000", photo_b="p0050"
        ),
    )
    assert split["group_count"] == 902
    assert len(split["recomputed_photo_ids"]) == 100
    split_event = distance_cache.stats_snapshot("pytest-1000")["events"][-1]
    assert split_event["new_squared_distance_calculations"] == 0
    assert split_event["ranking_records_reused"] == 100

    v3 = load_evaluation(db_path, "pytest-1000")
    assert v3["group_count"] == created["group_count"]
    v3_by_id = {p["photo_id"]: p for p in v3["predictions"]}
    for photo_id, prediction in by_id.items():
        assert v3_by_id[photo_id]["nearest_photo_id"] == prediction["nearest_photo_id"]


def test_tie_break_still_smallest_photo_id_through_cache():
    ids = ["a", "b", "c", "d"]
    features = np.asarray(
        [[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.int64
    )
    species = {i: i for i in ids}
    mapping = {"a": 0, "b": 0, "c": 1, "d": 2}
    cache = distance_cache
    cache.reset()
    cache.warm_batch("ties", ids, features)
    result, delta = predict_nearest_outside_cached(
        "ties", ids, species, features, mapping, ["a"]
    )
    assert result["a"]["nearest_photo_id"] == "c"
    assert delta["ranking_records_reused"] == 1
    cache.reset()


def test_cache_limit_zero_means_always_rebuild_but_exact(tmp_path):
    distance_cache.reset(limit=0)
    db_path = str(tmp_path / "nolimit.sqlite3")
    initialize(db_path)
    request, photo_ids, species, features = _random_batch("zero", 12, seed=8)
    create_batch(db_path, request)
    assert distance_cache.cached_count() == 0

    response = apply_correction(
        db_path,
        "zero",
        CorrectionRequest(
            expected_version=1, action="add", photo_a="q000", photo_b="q001"
        ),
    )
    assert response["recomputed_photo_ids"] == ["q000", "q001"]
    event = distance_cache.stats_snapshot("zero")["events"][-1]
    assert event["ranking_records_rebuilt"] == 2
    assert event["new_squared_distance_calculations"] == 2 * (len(photo_ids) - 1)

    groups = build_groups(photo_ids, {("q000", "q001")})
    oracle = predict_nearest_outside(
        photo_ids, species, features, group_index(groups), photo_ids
    )
    actual = {p["photo_id"]: p for p in load_evaluation(db_path, "zero")["predictions"]}
    for photo_id in photo_ids:
        assert actual[photo_id]["nearest_photo_id"] == oracle[photo_id]["nearest_photo_id"]
