"""Orakel-Tests (ORACLE-PLAYBOOK §5): Kernroutinen gegen einen UNABHÄNGIGEN Rechenweg.

  Zusammenhang, isolierte Knoten  gegen networkx (is_connected, isolates); Spannbaum gegen networkx (is_tree)
  exakte Momente                  Optimales beta und R^2 gegen eine GEWICHTETE KLEINSTE-QUADRATE-Anpassung (lstsq mit sqrt(w)-Skalierung, andere Formel als
                                  Kovarianzmatrix mal Kovarianzvektor) und gegen Bruchrechnung (fractions) über alle Kantenmengen
  bekannte Erwartungswerte        gegen die Aufzählung
  Schätzer                        Simulation: Orakel-Schätzer ist erwartungstreu
Mini-Instanzen, feste Stände (range statt Zufall), insgesamt wenige Sekunden."""

import itertools
import random
from fractions import Fraction

import numpy as np
import pytest

import kvr_algorithm as A
from kvr_scenario import grid_netz, named_netz, parse_edges


def _table(netz):
    tree = A.spanning_tree(netz.n, netz.edges)
    return tree, A.enumerate_table(netz.n, netz.edges, tree)


def _random_graph(rng, n, m):
    alle = list(itertools.combinations(range(n), 2))
    return n, tuple(rng.sample(alle, m))


def test_zusammenhang_isolierte_knoten_und_spannbaum_gegen_networkx():
    nx = pytest.importorskip("networkx")
    rng = random.Random(2026)
    for _ in range(30):
        n = rng.randint(3, 7)
        n, edges = _random_graph(rng, n, rng.randint(2, min(10, n * (n - 1) // 2)))
        tree = A.spanning_tree(n, edges)
        stats = np.array([[rng.random() < 0.6 for _ in edges] for _ in range(25)])
        v = A.control_values(stats, n, edges, tree)
        for i, row in enumerate(stats):
            g = nx.Graph()
            g.add_nodes_from(range(n))
            g.add_edges_from(e for e, keep in zip(edges, row) if keep)
            assert bool(v["X"][i]) == nx.is_connected(g)
            assert int(v["iso"][i]) == len(list(nx.isolates(g)))
            assert int(v["K"][i]) == g.number_of_edges()
        full = nx.Graph()
        full.add_nodes_from(range(n))
        full.add_edges_from(edges)
        forest = nx.Graph()
        forest.add_nodes_from(range(n))
        forest.add_edges_from(edges[e] for e in tree)
        assert nx.is_forest(forest) and nx.number_connected_components(forest) == nx.number_connected_components(full)


def _wls(table, m, p, names):
    """Gewichtete Kleinste-Quadrate-Anpassung von X auf (1, C): beta und R^2 über sqrt(w)-skalierte Zeilen, ohne Kovarianzmatrix."""
    w = p ** table["k"] * (1 - p) ** (m - table["k"])
    sw = np.sqrt(w)
    design = np.column_stack([np.ones(len(w))] + [table[nm] for nm in names])
    coef, *_ = np.linalg.lstsq(design * sw[:, None], table["X"] * sw, rcond=None)
    resid = table["X"] - design @ coef
    var_x = float(np.sum(w * table["X"] ** 2) - np.sum(w * table["X"]) ** 2)
    return coef[1:], 1.0 - float(np.sum(w * resid**2)) / var_x


@pytest.mark.parametrize("gitter", ["2x3", "3x3"])
@pytest.mark.parametrize("names", [["K"], ["iso"], ["tree"], ["K", "iso", "tree"]])
@pytest.mark.parametrize("p", [0.4, 0.7, 0.95])
def test_optimales_beta_und_faktor_gegen_gewichtete_kleinste_quadrate(gitter, names, p):
    netz = named_netz(gitter)
    tree, table = _table(netz)
    mo = A.exact_moments(table, netz.m, p, names)
    beta, r2 = _wls(table, netz.m, p, names)
    assert [mo["beta"][nm] for nm in names] == pytest.approx(list(beta), abs=1e-8)
    assert mo["factor"] == pytest.approx(1.0 - r2, abs=1e-8)


def _moments_fractions(netz, p: Fraction, name):
    """Exakte Kovarianz von X mit der Kontrollvariate `name` und Varianz der Kontrollvariate durch Bruchrechnung über alle Kantenmengen."""
    tree = A.spanning_tree(netz.n, netz.edges)
    ex = ec = exc = ecc = Fraction(0)
    for mask in range(1 << netz.m):
        up = np.array([[bool(mask >> e & 1) for e in range(netz.m)]])
        v = A.control_values(up, netz.n, netz.edges, tree)
        k = int(up.sum())
        w = p**k * (1 - p) ** (netz.m - k)
        x, c = Fraction(int(v["X"][0])), Fraction(int(v[name][0]))
        ex += w * x
        ec += w * c
        exc += w * x * c
        ecc += w * c * c
    return ex, ec, exc - ex * ec, ecc - ec * ec


@pytest.mark.parametrize("name", ["K", "iso", "tree"])
def test_beta_und_erwartungswerte_gegen_bruchrechnung_auf_dreieck_und_kreis(name):
    for netz in (grid_netz(2, 2), parse_edges("0-1\n1-2\n0-2")):
        for p in (Fraction(1, 3), Fraction(3, 4)):
            ex, ec, cov, var_c = _moments_fractions(netz, p, name)
            tree, table = _table(netz)
            mo = A.exact_moments(table, netz.m, float(p), [name])
            assert mo["EX"] == pytest.approx(float(ex), abs=1e-12) and mo["EC"][name] == pytest.approx(float(ec), abs=1e-12)
            assert mo["beta"][name] == pytest.approx(float(cov / var_c), abs=1e-12)


@pytest.mark.parametrize("gitter", ["2x3", "3x3", "2x5"])
def test_bekannte_erwartungswerte_gegen_die_aufzaehlung(gitter):
    netz = named_netz(gitter)
    if netz.m > 13:
        pytest.skip("zu groß für diesen Test")
    tree, table = _table(netz)
    for p in (0.3, 0.7, 0.95):
        mo = A.exact_moments(table, netz.m, p, ["K", "iso", "tree"])
        mu = A.known_means(netz.n, netz.edges, tree, p)
        for nm in ("K", "iso", "tree"):
            assert mo["EC"][nm] == pytest.approx(mu[nm], abs=1e-12), (gitter, p, nm)


def test_bekannte_erwartungswerte_fuer_eigene_netze_mit_mehrfachkanten_und_wald():
    for text in ("0-1\n1-2\n2-0\n2-3", "0-1\n2-3", "0-1\n1-2"):
        netz = parse_edges(text)
        tree, table = _table(netz)
        mo = A.exact_moments(table, netz.m, 0.6, ["K", "iso", "tree"])
        mu = A.known_means(netz.n, netz.edges, tree, 0.6)
        assert [mo["EC"][nm] for nm in ("K", "iso", "tree")] == pytest.approx([mu[nm] for nm in ("K", "iso", "tree")], abs=1e-12)


def test_orakel_schaetzer_ist_erwartungstreu_in_der_simulation():
    netz = named_netz("2x3")
    tree, table = _table(netz)
    for names in (["iso"], ["K", "iso", "tree"]):
        mo = A.exact_moments(table, netz.m, 0.8, names)
        mu = A.known_means(netz.n, netz.edges, tree, 0.8)
        sp = A.spread(3, 600, 100, 0.8, netz.n, netz.edges, tree, names, mu, mo["beta"], mo["EX"])
        se = sp["est"]["oracle"].std(ddof=1) / np.sqrt(600)
        assert abs(sp["est"]["oracle"].mean() - mo["EX"]) < 4 * se


def test_varianz_des_orakel_schaetzers_gleich_faktor_mal_einfacher_varianz():
    netz = named_netz("2x3")
    tree, table = _table(netz)
    mo = A.exact_moments(table, netz.m, 0.8, ["iso"])
    mu = A.known_means(netz.n, netz.edges, tree, 0.8)
    sp = A.spread(4, 1500, 100, 0.8, netz.n, netz.edges, tree, ["iso"], mu, mo["beta"], mo["EX"])
    ratio = sp["est"]["oracle"].var(ddof=1) / sp["est"]["plain"].var(ddof=1)
    assert ratio == pytest.approx(mo["factor"], rel=0.2)                          # Rauschen bei 1500 Wiederholungen: rund 5 %
