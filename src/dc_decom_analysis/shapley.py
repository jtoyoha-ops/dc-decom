"""Exact Shapley values of a cooperative game given on all coalitions (part emissions).

shapley  phi_i = sum over S not containing i of |S|! (n - |S| - 1)! / n! (v(S + i) - v(S))
oat_on   v({i}) - v(empty): effect of adding one player to the empty coalition
oat_off  v(N) - v(N - i): effect of removing one player from the grand coalition

``values`` maps every coalition (a frozenset of players) to v(S); all 2^n
coalitions must be present. The Shapley value is the average of the marginal
contributions of i over all n! orders in which the players can be added, so
it does not depend on an order (efficiency: sum phi = v(N) - v(empty)).
Definitions: config/experiments_v35.yaml section emissions (defined before the runs).
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Hashable, Mapping, Sequence, TypeVar

P = TypeVar("P", bound=Hashable)


def _check(values: Mapping[frozenset, float], players: Sequence[P]) -> None:
    n = len(players)
    if len(set(players)) != n:
        raise ValueError("players must be distinct")
    for k in range(n + 1):
        for S in combinations(players, k):
            if frozenset(S) not in values:
                raise KeyError(f"coalition {sorted(map(str, S))} missing")


def shapley(values: Mapping[frozenset, float], players: Sequence[P]) -> Dict[P, float]:
    _check(values, players)
    n = len(players)
    w = [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
    out: Dict[P, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        phi = 0.0
        for k in range(n):
            for S in combinations(others, k):
                s = frozenset(S)
                phi += w[k] * (values[s | {i}] - values[s])
        out[i] = phi
    return out


def oat_on(values: Mapping[frozenset, float], players: Sequence[P]) -> Dict[P, float]:
    _check(values, players)
    e = values[frozenset()]
    return {i: values[frozenset([i])] - e for i in players}


def oat_off(values: Mapping[frozenset, float], players: Sequence[P]) -> Dict[P, float]:
    _check(values, players)
    full = frozenset(players)
    return {i: values[full] - values[full - {i}] for i in players}
