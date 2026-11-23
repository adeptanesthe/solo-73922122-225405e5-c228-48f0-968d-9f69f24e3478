#!/usr/bin/env python3
"""Generate the deterministic 1000-photo demo payload.

Layout (1000 photos, each with 32 int16 features):

* ``p0000..p0049`` form chain A (50 photos, 49 links)
* ``p0050..p0099`` form chain B (50 photos, 49 links)
* ``p0100..p0999`` are 900 singletons

The two chains are generated as jittered copies of the same tight cluster, so
at version 1 every chain photo's nearest outside neighbour lives in the other
chain. Adding ``p0000-p0050`` therefore merges them and disqualifies each
chain photo's previous nearest neighbour. Removing the bridge later restores
the original eligibility.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPECIES_CLUSTER = "deer"
SPECIES_SINGLETON = "boar"
CLUSTER_CENTER = 2000
SINGLETON_CENTER = -12000
JITTER = 18
SINGLETON_SPREAD = 120
SIZE = 1000
DIMENSIONS = 32


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, value))


def build_payload(seed: int = 20261123) -> dict:
    rng = random.Random(seed)
    center = [
        _clamp_int16(CLUSTER_CENTER + rng.randint(-JITTER, JITTER))
        for _ in range(DIMENSIONS)
    ]

    photos: list[dict] = []
    for index in range(SIZE):
        photo_id = f"p{index:04d}"
        if index < 100:
            # Two chains share one tight cluster centre; use a per-photo jitter
            # so distances are distinct while cross-chain pairs stay very close.
            features = [
                _clamp_int16(value + rng.randint(-JITTER, JITTER))
                for value in center
            ]
            species = SPECIES_CLUSTER
        else:
            features = [
                _clamp_int16(
                    SINGLETON_CENTER + rng.randint(-SINGLETON_SPREAD, SINGLETON_SPREAD)
                )
                for _ in range(DIMENSIONS)
            ]
            species = SPECIES_SINGLETON
        photos.append(
            {"photo_id": photo_id, "species": species, "features": features}
        )

    pairs = [
        {"photo_a": f"p{i:04d}", "photo_b": f"p{i + 1:04d}"}
        for i in range(0, 49)
    ] + [
        {"photo_a": f"p{i:04d}", "photo_b": f"p{i + 1:04d}"}
        for i in range(50, 99)
    ]
    return {"batch_id": "demo-1000", "photos": photos, "pairs": pairs}


def main() -> None:
    here = Path(__file__).resolve().parent
    payload = build_payload()
    out = here / "create_batch_1000.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out} with {len(payload['photos'])} photos and {len(payload['pairs'])} pairs")


if __name__ == "__main__":
    main()
