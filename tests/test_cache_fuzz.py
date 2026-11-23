"""Fuzz: cached predictions equal brute force across random group changes."""

from __future__ import annotations

import random

import numpy as np

from app.distance_cache import DistanceRankingCache
from app.grouping import build_groups, group_index
from app.predictions import predict_nearest_outside, predict_nearest_outside_cached


def test_cached_predictions_match_brute_force_under_random_edits():
    rng = random.Random(2026)
    size = 40
    ids = [f"z{i:03d}" for i in range(size)]
    # Small feature range deliberately creates many exact distance ties.
    photos = np.asarray(
        [[rng.randint(-3, 3) for _ in range(32)] for _ in ids], dtype=np.int64
    )
    species = {photo_id: rng.choice(("a", "b")) for photo_id in ids}
    cache = DistanceRankingCache(limit=55)
    cache.warm_batch("fuzz", ids, photos)

    edges: set[tuple[str, str]] = set()
    for step in range(60):
        a, b = rng.sample(ids, 2)
        edge = (min(a, b), max(a, b))
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)
        groups = build_groups(ids, edges)
        if len(groups) < 2:
            continue
        mapping = group_index(groups)
        targets = rng.sample(ids, rng.randint(1, size))

        cached, delta = predict_nearest_outside_cached(
            "fuzz", ids, species, photos, mapping, targets, cache=cache
        )
        oracle = predict_nearest_outside(ids, species, photos, mapping, targets)
        assert sorted(cached) == sorted(oracle)
        for photo_id, prediction in cached.items():
            assert prediction == oracle[photo_id], (step, photo_id)
        # Every target was served either from the cache or rebuilt exactly.
        assert (
            delta["ranking_records_reused"] + delta["ranking_records_rebuilt"]
            == len(cached)
        )
        assert cache.cached_count() <= 55


def test_cached_path_exact_after_cache_reset_mid_sequence():
    rng = random.Random(77)
    size = 25
    ids = [f"z{i:03d}" for i in range(size)]
    photos = np.asarray(
        [[rng.randint(-100, 100) for _ in range(32)] for _ in ids], dtype=np.int64
    )
    species = {photo_id: "s" for photo_id in ids}
    cache = DistanceRankingCache(limit=10)
    cache.warm_batch("reset-fuzz", ids, photos)

    edges = {(ids[i], ids[i + 1]) for i in range(0, 12)}
    groups = build_groups(ids, edges)
    mapping = group_index(groups)

    # Simulate a restart halfway: all rankings disappear.
    cache.reset(limit=10)
    assert cache.cached_count() == 0
    cached, delta = predict_nearest_outside_cached(
        "reset-fuzz", ids, species, photos, mapping, ids, cache=cache
    )
    oracle = predict_nearest_outside(ids, species, photos, mapping, ids)
    assert cached == oracle
    assert delta["ranking_records_rebuilt"] == len(ids)
    assert delta["new_squared_distance_calculations"] == len(ids) * (size - 1)
