"""Exact Shapley values on textbook games (part emissions)."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dc_decom_analysis import shapley as sh  # noqa: E402


def _game(players, v):
    return {frozenset(S): float(v(frozenset(S)))
            for k in range(len(players) + 1) for S in combinations(players, k)}


def test_glove_game():
    # player 1 holds a left glove, players 2 and 3 a right glove; a pair is worth 1
    players = (1, 2, 3)
    vals = _game(players, lambda S: 1.0 if 1 in S and len(S) >= 2 else 0.0)
    phi = sh.shapley(vals, players)
    assert phi[1] == pytest.approx(2.0 / 3.0, abs=1e-15)
    assert phi[2] == pytest.approx(1.0 / 6.0, abs=1e-15)
    assert phi[3] == pytest.approx(1.0 / 6.0, abs=1e-15)
    assert sh.oat_on(vals, players) == {1: 0.0, 2: 0.0, 3: 0.0}
    assert sh.oat_off(vals, players) == {1: 1.0, 2: 0.0, 3: 0.0}


def test_efficiency_symmetry_null_player():
    players = ("a", "b", "c", "d", "e", "f")
    # a and b symmetric, f a null player, interactions between c, d and e
    weights = {"a": 2.0, "b": 2.0, "c": 1.5, "d": 0.7, "e": 3.1, "f": 0.0}

    def v(S):
        x = sum(weights[p] for p in S)
        if {"c", "d"} <= S:
            x += 1.3
        if {"c", "d", "e"} <= S:
            x -= 0.4
        return x

    vals = _game(players, v)
    phi = sh.shapley(vals, players)
    full = frozenset(players)
    assert sum(phi.values()) == pytest.approx(vals[full] - vals[frozenset()], abs=1e-12)
    assert phi["a"] == pytest.approx(phi["b"], abs=1e-15)
    assert phi["f"] == pytest.approx(0.0, abs=1e-15)
    # additive part and the interaction split equally among its members
    assert phi["c"] == pytest.approx(1.5 + 1.3 / 2 - 0.4 / 3, abs=1e-12)
    assert phi["e"] == pytest.approx(3.1 - 0.4 / 3, abs=1e-12)


def test_missing_coalition():
    vals = {frozenset(): 0.0, frozenset([1]): 1.0}
    with pytest.raises(KeyError):
        sh.shapley(vals, (1, 2))
