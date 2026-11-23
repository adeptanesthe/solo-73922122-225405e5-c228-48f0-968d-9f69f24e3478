"""Process-wide LRU cache of per-photo sorted neighbour rankings.

Features of a batch are immutable once the batch has been created, so a
photo's complete ranking of every other photo by exact squared Euclidean
distance never goes stale. A correction only changes *which* ranking entries
are eligible (photos in the same group are skipped), so merge/split
corrections can reuse the cached ranking instead of recomputing any distance.

Each cached record is one photo's full ranking:
``(neighbour indices sorted by (squared distance, photo index), distances)``.
At most ``limit`` records (default 32000) are retained across all batches in
the process; the least-recently-used record is evicted first. An evicted or
never-built record is rebuilt from the batch features on demand, so results
stay exact after eviction and after a process restart.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_CACHE_LIMIT = 32_000

# Fields carried by every statistics delta / event.
DELTA_FIELDS = (
    "new_squared_distance_calculations",
    "bulk_pair_calculations",
    "ranking_records_built",
    "ranking_records_rebuilt",
    "ranking_records_reused",
)


def _env_limit() -> int:
    raw = os.environ.get("DISTANCE_CACHE_LIMIT")
    if raw is None or not raw.strip():
        return DEFAULT_CACHE_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CACHE_LIMIT
    return max(0, value)


def zero_delta() -> dict[str, int]:
    return {field: 0 for field in DELTA_FIELDS}


def _merge_delta(total: dict[str, int], delta: dict[str, int]) -> None:
    for field in DELTA_FIELDS:
        total[field] += delta.get(field, 0)


def _squared_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Exact integer squared Euclidean distances (int64)."""
    norms = np.einsum("ij,ij->i", matrix, matrix)
    return norms[:, None] + norms[None, :] - 2 * (matrix @ matrix.T)


def _build_ranking(matrix: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank every other photo for one photo.

    Primary key is the exact squared distance; the secondary key is the array
    index. Callers pass photos sorted by ID, so the secondary key implements
    the smallest-photo-ID tie break exactly.
    """
    size = matrix.shape[0]
    differences = matrix - matrix[index]
    distances = np.einsum("ij,ij->i", differences, differences)
    indices = np.arange(size)
    # lexsort uses its last key as the primary sort key.
    order = np.lexsort((indices, distances))
    order = order[order != index]
    return order.astype(np.int32), distances[order].astype(np.int64)


class DistanceRankingCache:
    """LRU cache of full neighbour rankings shared by all requests."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = _env_limit() if limit is None else max(0, int(limit))
        self._lock = threading.RLock()
        self._records: OrderedDict[
            tuple[str, str], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._stats: dict[str, dict[str, Any]] = {}
        # Work delta of the most recent cached prediction call per batch.
        self._last_delta: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------ state
    def reset(self, limit: int | None = None) -> None:
        """Drop every record and counter (used to simulate a restart).

        Without an argument the default process limit is restored; an explicit
        limit (including 0) is honoured as-is.
        """
        with self._lock:
            self.limit = _env_limit() if limit is None else max(0, int(limit))
            self._records.clear()
            self._stats.clear()
            self._last_delta.clear()

    def cached_count(self) -> int:
        with self._lock:
            return len(self._records)

    def drop_batch(self, batch_id: str) -> None:
        """Remove every cached record of a batch that failed to commit."""
        with self._lock:
            for key in [key for key in self._records if key[0] == batch_id]:
                del self._records[key]
            self._stats.pop(batch_id, None)
            self._last_delta.pop(batch_id, None)

    def _batch_stats(self, batch_id: str) -> dict[str, Any]:
        stats = self._stats.get(batch_id)
        if stats is None:
            stats = {field: 0 for field in DELTA_FIELDS}
            stats["events"] = []
            self._stats.setdefault(batch_id, stats)
        return stats

    def _store(self, key: tuple[str, str], record: tuple[np.ndarray, np.ndarray]) -> None:
        if self.limit == 0:
            return
        self._records[key] = record
        self._records.move_to_end(key)
        while len(self._records) > self.limit:
            self._records.popitem(last=False)

    # ----------------------------------------------------------------- writes
    def warm_batch(
        self,
        batch_id: str,
        photo_ids: Sequence[str],
        features: np.ndarray,
    ) -> dict[str, int]:
        """Build and cache a full ranking for every photo in a new batch.

        Bulk work is counted per unordered photo pair: ``n*(n-1)/2`` squared
        distances for ``n`` photos, even though the matrix multiply evaluates
        both directions.
        """
        size = len(photo_ids)
        delta = zero_delta()
        if size < 2 or self.limit == 0:
            return delta

        matrix = np.asarray(features, dtype=np.int64)
        squared = _squared_distance_matrix(matrix)
        indices = np.arange(size)
        pair_count = size * (size - 1) // 2
        delta["new_squared_distance_calculations"] = pair_count
        delta["bulk_pair_calculations"] = pair_count

        with self._lock:
            for index, photo_id in enumerate(photo_ids):
                order = np.lexsort((indices, squared[index]))
                order = order[order != index]
                record = (order.astype(np.int32), squared[index, order].astype(np.int64))
                self._store((batch_id, photo_id), record)
                delta["ranking_records_built"] += 1
        return delta

    def nearest_outside_batch(
        self,
        batch_id: str,
        photo_ids: Sequence[str],
        features: np.ndarray,
        group_by_id: dict[str, int],
        target_photo_ids: Sequence[str],
    ) -> tuple[dict[str, tuple[str, int]], dict[str, int]]:
        """Exact out-of-group nearest neighbour for each target.

        Returns ``{target_id: (nearest_id, squared_distance)}`` and a statistics
        delta. A cached ranking is reused for free; a missing record is rebuilt
        from the immutable features (``n-1`` new pair calculations) and cached,
        subject to LRU eviction.
        """
        matrix = np.asarray(features, dtype=np.int64)
        size = len(photo_ids)
        id_to_index = {photo_id: index for index, photo_id in enumerate(photo_ids)}
        group_array = np.asarray(
            [group_by_id[photo_id] for photo_id in photo_ids], dtype=np.int64
        )
        result: dict[str, tuple[str, int]] = {}
        delta = zero_delta()
        if size < 2 or not target_photo_ids:
            return result, delta

        with self._lock:
            for target_id in sorted(set(target_photo_ids)):
                target_index = id_to_index[target_id]
                key = (batch_id, target_id)
                record = self._records.get(key)
                if record is None:
                    neighbours, distances = _build_ranking(matrix, target_index)
                    self._store(key, (neighbours, distances))
                    delta["ranking_records_rebuilt"] += 1
                    # A full ranking computes one distance per other photo,
                    # i.e. one unordered pair each.
                    delta["new_squared_distance_calculations"] += size - 1
                else:
                    self._records.move_to_end(key)
                    neighbours, distances = record
                    delta["ranking_records_reused"] += 1

                own_group = group_array[target_index]
                eligible = group_array[neighbours] != own_group
                first = int(np.flatnonzero(eligible)[0])
                nearest_index = int(neighbours[first])
                result[target_id] = (photo_ids[nearest_index], int(distances[first]))
            # Expose the work delta through a guarded one-shot slot so the
            # service entry point keeps its historical single-dict shape.
            self._last_delta[batch_id] = dict(delta)
        return result, delta

    def consume_last_delta(self, batch_id: str) -> dict[str, int]:
        """Take and clear the work delta of the latest cached prediction call."""
        with self._lock:
            return self._last_delta.pop(batch_id, zero_delta())

    # ------------------------------------------------------------- statistics
    def commit_stats(self, batch_id: str, delta: dict[str, int]) -> None:
        """Add a work delta to the batch's cumulative counters."""
        with self._lock:
            stats = self._batch_stats(batch_id)
            _merge_delta(stats, delta)

    def record_event(
        self,
        batch_id: str,
        phase: str,
        at_version: int,
        delta: dict[str, int],
        action: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            stats = self._batch_stats(batch_id)
            event = {
                "phase": phase,
                "at_version": at_version,
                "action": action,
                **{field: int(delta.get(field, 0)) for field in DELTA_FIELDS},
            }
            stats["events"].append(event)
            return dict(event)

    def stats_snapshot(self, batch_id: str) -> dict[str, Any]:
        with self._lock:
            stats = self._stats.get(batch_id)
            if stats is None:
                counters = zero_delta()
                events: list[dict[str, Any]] = []
            else:
                counters = {field: stats[field] for field in DELTA_FIELDS}
                events = [dict(event) for event in stats["events"]]
            batch_records = sum(
                1 for cached_batch_id, _ in self._records if cached_batch_id == batch_id
            )
            return {
                "batch_id": batch_id,
                "cache_limit": self.limit,
                "cached_ranking_records": len(self._records),
                "batch_cached_ranking_records": batch_records,
                **counters,
                "events": events,
            }


# Process-wide singleton: rankings are reusable across requests but never leave
# the process, and nothing here is persisted.
distance_cache = DistanceRankingCache()
