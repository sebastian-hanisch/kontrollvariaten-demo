"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvr_scenario import Netz  # noqa: E402


def _netz(name, n, edges):
    return Netz(name, n, tuple(edges), tuple((float(i), 0.0) for i in range(n)))


@pytest.fixture
def dreieck():
    """Dreieck: 3 Knoten, 3 Kanten (Spannbaum = die ersten zwei). Bei p = 1/2 von Hand: E X = 1/2, Var X = 1/4, E K = 3/2, E iso = 3/4, E tree = 1/4;
    mit K: Cov = 3/8, beta* = 1/2, Faktor 1/4; mit iso: Cov = -3/8, Var iso = 15/16, beta* = -2/5, Faktor 2/5."""
    return _netz("Dreieck", 3, [(0, 1), (1, 2), (0, 2)])


@pytest.fixture
def pfad2():
    """Pfad mit zwei Kanten und drei Knoten: X = beide halten. Bei p = 1/2: E X = 1/4, Var X = 3/16, Var K = 1/2, Cov(X, K) = 1/4, beta* = 1/2, Faktor 1/3."""
    return _netz("Pfad", 3, [(0, 1), (1, 2)])


@pytest.fixture
def kreis4():
    return _netz("Kreis", 4, [(0, 1), (1, 2), (2, 3), (3, 0)])


@pytest.fixture
def k4():
    return _netz("K4", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
