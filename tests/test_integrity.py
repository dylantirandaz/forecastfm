"""Tests for deterministic experiment-artifact hashing."""

from forecastfm.integrity import canonical_json, canonical_sha256


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    first = {"alpha": 1, "beta": {"yes": 0.6, "no": 0.4}}
    second = {"beta": {"no": 0.4, "yes": 0.6}, "alpha": 1}

    assert canonical_json(first) == canonical_json(second)
    assert canonical_sha256(first) == canonical_sha256(second)
