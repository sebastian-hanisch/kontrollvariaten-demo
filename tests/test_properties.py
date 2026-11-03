"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für ALLE Eingaben gelten müssen, an festen Zufallsständen geprüft."""

import random

import numpy as np
import pytest

import kvr_algorithm as A
import kvr_constants as C
import kvr_evaluation as ev
from kvr_scenario import grid_netz, named_netz

PS = (0.3, 0.5, 0.7, 0.9, 0.99)


def _table(netz):
    tree = A.spanning_tree(netz.n, netz.edges)
    return tree, A.enumerate_table(netz.n, netz.edges, tree)


@pytest.mark.parametrize("gitter", ["2x3", "3x3"])
def test_mit_exaktem_beta_ist_der_schaetzer_im_erwartungswert_exakt(gitter):
    """E[X - beta (C - mu)] = E X: bei exakt bekanntem mu und festem beta, hier mit exakten gewichteten Summen über alle Kantenmengen."""
    netz = named_netz(gitter)
    tree, table = _table(netz)
    for p in PS:
        for names in (["K"], ["iso"], ["tree"], ["K", "iso", "tree"]):
            mo = A.exact_moments(table, netz.m, p, names)
            mu = A.known_means(netz.n, netz.edges, tree, p)
            w = p ** table["k"] * (1 - p) ** (netz.m - table["k"])
            y = table["X"] - sum(mo["beta"][nm] * (table[nm] - mu[nm]) for nm in names)
            assert float(np.sum(w * y)) == pytest.approx(mo["EX"], abs=1e-12), (gitter, p, names)
            assert float(np.sum(w * y * y) - np.sum(w * y) ** 2) == pytest.approx(mo["factor"] * mo["VarX"], abs=1e-12)


@pytest.mark.parametrize("gitter", ["2x3", "3x3"])
def test_faktor_liegt_in_null_eins_und_mehr_kontrollvariaten_machen_ihn_nie_schlechter(gitter):
    netz = named_netz(gitter)
    tree, table = _table(netz)
    for p in PS:
        f = {k: A.exact_moments(table, netz.m, p, names)["factor"] for k, names in C.CONTROLS.items()}
        assert all(0.0 <= v <= 1.0 + 1e-12 for v in f.values()), (p, f)
        assert f["alle"] <= min(f["K"], f["iso"], f["tree"]) + 1e-12
        two = A.exact_moments(table, netz.m, p, ["K", "iso"])["factor"]
        assert f["alle"] <= two + 1e-12 and two <= min(f["K"], f["iso"]) + 1e-12


def test_bei_einer_kontrollvariate_ist_der_faktor_eins_minus_korrelation_zum_quadrat():
    netz = named_netz("3x3")
    tree, table = _table(netz)
    for p in PS:
        for nm in ("K", "iso", "tree"):
            mo = A.exact_moments(table, netz.m, p, [nm])
            assert mo["factor"] == pytest.approx(1.0 - mo["corr"][nm] ** 2, abs=1e-12)


def test_faktor_ist_invariant_gegen_umskalieren_und_verschieben_der_kontrollvariate():
    netz = named_netz("3x3")
    tree, table = _table(netz)
    skaliert = dict(table)
    skaliert["K"] = 7.0 * table["K"] - 3.0
    for p in (0.5, 0.9):
        a = A.exact_moments(table, netz.m, p, ["K"])
        b = A.exact_moments(skaliert, netz.m, p, ["K"])
        assert b["factor"] == pytest.approx(a["factor"], abs=1e-12) and b["beta"]["K"] == pytest.approx(a["beta"]["K"] / 7.0, abs=1e-12)


def test_faktor_haengt_nicht_von_der_kantenreihenfolge_ab():
    rng = random.Random(8)
    netz = grid_netz(2, 3)
    basis = A.exact_moments(_table(netz)[1], netz.m, 0.8, ["iso"])["factor"]
    for _ in range(5):
        edges = list(netz.edges)
        rng.shuffle(edges)
        tree = A.spanning_tree(netz.n, edges)
        table = A.enumerate_table(netz.n, edges, tree)
        assert A.exact_moments(table, netz.m, 0.8, ["iso"])["factor"] == pytest.approx(basis, abs=1e-12)


def test_vorzeichen_der_korrelationen_im_mittleren_bereich():
    netz = named_netz("3x3")
    tree, table = _table(netz)
    mo = A.exact_moments(table, netz.m, 0.7, ["K", "iso", "tree"])
    assert mo["corr"]["K"] > 0 and mo["corr"]["iso"] < 0 and mo["corr"]["tree"] > 0


def test_beta_null_liefert_den_einfachen_schaetzer():
    X = np.array([1.0, 0.0, 1.0])
    Cm = np.array([[3.0], [0.0], [5.0]])
    assert A.cv_estimate(X, Cm, np.array([123.0]), np.array([0.0])) == X.mean()


def test_gleicher_strom_gleiche_zahlen():
    netz = named_netz("2x3")
    tree = A.spanning_tree(netz.n, netz.edges)
    mu = A.known_means(netz.n, netz.edges, tree, 0.8)
    mo = A.exact_moments(_table(netz)[1], netz.m, 0.8, ["iso"])
    a = A.spread(9, 30, 40, 0.8, netz.n, netz.edges, tree, ["iso"], mu, mo["beta"], mo["EX"])
    b = A.spread(9, 30, 40, 0.8, netz.n, netz.edges, tree, ["iso"], mu, mo["beta"], mo["EX"])
    assert all(np.array_equal(a["est"][k], b["est"][k]) for k in a["est"]) and a["coverage"] == b["coverage"]


def test_plugin_hat_bei_mittlerem_p_und_kleinem_n_eine_negative_verzerrung():
    """Gemessen (Vormessung): Verzerrung ≈ -0,36 / N bei einer Kontrollvariate; hier nur die Richtung bei N = 30 und 800 Wiederholungen."""
    netz = named_netz("3x4")
    tree, table = _table(netz)
    mo = A.exact_moments(table, netz.m, 0.7, ["iso"])
    mu = A.known_means(netz.n, netz.edges, tree, 0.7)
    sp = A.spread(3, 800, 30, 0.7, netz.n, netz.edges, tree, ["iso"], mu, mo["beta"], mo["EX"])
    plug = sp["est"]["plug"]
    assert plug.mean() - mo["EX"] < -2 * plug.std(ddof=1) / np.sqrt(800)


def test_analyse_ist_deterministisch():
    s = ev.Settings(n=100)
    assert ev.analyse(s) == ev.analyse(s)
