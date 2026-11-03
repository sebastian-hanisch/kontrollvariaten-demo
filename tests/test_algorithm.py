"""Kern gegen Handrechnung auf den Mini-Instanzen (conftest.py) und gegen geschlossene Formeln; je Einheit mindestens ein Test."""

import numpy as np
import pytest

import kvr_algorithm as A


# --- Zufallsstrom --------------------------------------------------------------------------------------------------------------------------

def test_splitmix64_bekannter_startwert():
    assert A.mix_int(A.GAMMA) == 0xE220A8397B1DCDAF       # Seed 0, erste Ausgabe des Verfahrens


@pytest.mark.parametrize("seed", [0, 7, 2**63 + 12345])
def test_vektorisierter_strom_gleich_ganzzahl_referenz(seed):
    ref = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 301)]
    assert np.array_equal(A.uniforms(seed, 300), np.array(ref))


def test_edge_states_sind_ein_praefix_und_folgen_der_schwelle():
    a, b = A.edge_states(5, 3, 0.7, 4), A.edge_states(5, 10, 0.7, 4)
    assert np.array_equal(a, b[:3])
    assert np.array_equal(a, A.uniforms(5, 12).reshape(3, 4) < 0.7)


def test_stream_seed_ist_deterministisch_und_unterscheidet_schluessel():
    assert A.stream_seed(1, 2) == A.stream_seed(1, 2)
    assert len({A.stream_seed(1, 2), A.stream_seed(2, 1), A.stream_seed(1), A.stream_seed(1, 3)}) == 4


# --- Ziel und Kontrollvariaten -----------------------------------------------------------------------------------------------------------

def test_zusammenhang_dreieck_von_hand(dreieck):
    z = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    assert list(A.connected(z, 3, dreieck.edges)) == [True, True, False, False]


def test_ein_knoten_ist_zusammenhaengend():
    assert A.connected(np.zeros((2, 0), dtype=bool), 1, ()).all()


def test_spannbaum_hat_n_minus_eins_kanten_und_verbindet(k4):
    tree = A.spanning_tree(k4.n, k4.edges)
    assert len(tree) == 3
    z = np.zeros((1, k4.m), dtype=bool)
    z[0, tree] = True
    assert A.connected(z, k4.n, k4.edges)[0]


def test_spannwald_bei_nicht_zusammenhaengendem_netz():
    assert A.spanning_tree(4, ((0, 1), (2, 3))) == [0, 1]
    assert A.spanning_tree(4, ((0, 1), (0, 1), (2, 3))) == [0, 2]            # die parallele Kante gehört nicht dazu


def test_kontrollvariaten_dreieck_von_hand(dreieck):
    tree = A.spanning_tree(dreieck.n, dreieck.edges)
    assert tree == [0, 1]
    z = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    v = A.control_values(z, 3, dreieck.edges, tree)
    assert v["X"].tolist() == [1, 0, 0, 1]
    assert v["K"].tolist() == [2, 1, 0, 3]
    assert v["iso"].tolist() == [0, 1, 3, 0]            # nur Kante 0-2 hält: Knoten 1 ist isoliert
    assert v["tree"].tolist() == [1, 0, 0, 1]           # Spannbaum = Kanten 0-1 und 1-2


def test_bekannte_erwartungswerte_dreieck_bei_p_gleich_einhalb(dreieck):
    tree = A.spanning_tree(dreieck.n, dreieck.edges)
    mu = A.known_means(dreieck.n, dreieck.edges, tree, 0.5)
    assert mu == {"K": 1.5, "iso": 0.75, "tree": 0.25}


# --- Wahrheit ----------------------------------------------------------------------------------------------------------------------------

def _moments(netz, p, names):
    tree = A.spanning_tree(netz.n, netz.edges)
    return A.exact_moments(A.enumerate_table(netz.n, netz.edges, tree), netz.m, p, names)


def test_exakte_momente_dreieck_kantenzahl_von_hand(dreieck):
    mo = _moments(dreieck, 0.5, ["K"])
    assert mo["EX"] == pytest.approx(0.5) and mo["VarX"] == pytest.approx(0.25)
    assert mo["EC"]["K"] == pytest.approx(1.5)
    assert mo["beta"]["K"] == pytest.approx(0.5) and mo["factor"] == pytest.approx(0.25) and mo["R2"] == pytest.approx(0.75)
    assert mo["corr"]["K"] == pytest.approx(np.sqrt(0.75))


def test_exakte_momente_dreieck_isolierte_knoten_von_hand(dreieck):
    mo = _moments(dreieck, 0.5, ["iso"])
    assert mo["EC"]["iso"] == pytest.approx(0.75)
    assert mo["beta"]["iso"] == pytest.approx(-0.4) and mo["factor"] == pytest.approx(0.4)
    assert mo["corr"]["iso"] == pytest.approx(-np.sqrt(0.6))


def test_exakte_momente_pfad_mit_zwei_kanten_von_hand(pfad2):
    mo = _moments(pfad2, 0.5, ["K"])
    assert mo["EX"] == pytest.approx(0.25) and mo["VarX"] == pytest.approx(3 / 16)
    assert mo["beta"]["K"] == pytest.approx(0.5) and mo["factor"] == pytest.approx(1 / 3)


def test_ohne_kontrollvariaten_gibt_es_nur_den_erwartungswert(dreieck):
    mo = _moments(dreieck, 0.5, [])
    assert set(mo) == {"EX", "VarX", "names"} and mo["EX"] == pytest.approx(0.5)


def test_netz_ohne_streuung_hat_faktor_nan():
    from kvr_scenario import Netz

    getrennt = Netz("zwei Paare", 4, ((0, 1), (2, 3)), tuple((float(i), 0.0) for i in range(4)))
    mo = _moments(getrennt, 0.7, ["K", "iso"])
    assert mo["EX"] == 0.0 and np.isnan(mo["factor"]) and np.isnan(mo["corr"]["K"])


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

def test_cv_estimate_von_hand():
    X = np.array([1.0, 0.0, 1.0, 1.0])
    Cm = np.array([[2.0], [0.0], [2.0], [3.0]])
    assert A.cv_estimate(X, Cm, np.array([1.5]), np.array([0.5])) == pytest.approx(0.75 - 0.5 * (1.75 - 1.5))
    assert A.cv_estimate(X, Cm, np.array([1.5]), np.array([0.0])) == pytest.approx(0.75)


def test_plugin_beta_von_hand_und_randfaelle():
    X = np.array([0.0, 1.0, 0.0, 1.0])
    assert A.plugin_beta(X, X[:, None]) == pytest.approx([1.0])
    assert A.plugin_beta(X, 3 * X[:, None] + 1) == pytest.approx([1 / 3])                    # skaliert und verschoben
    assert A.plugin_beta(np.ones(4), X[:, None]).tolist() == [0.0]                          # X konstant
    assert A.plugin_beta(X, np.ones((4, 1))).tolist() == [0.0]                              # C konstant (Kovarianzmatrix singulär)
    assert A.plugin_beta(X[:2], X[:2, None]).tolist() == [0.0]                              # zu wenige Ziehungen
    Cm = np.stack([X, 2 * X], axis=1)
    assert A.plugin_beta(X, Cm).tolist() == [0.0, 0.0]                                      # kollineare Kontrollvariaten


def test_cv_stderr_von_hand():
    X = np.array([0.0, 1.0, 0.0, 1.0])
    # beta = 0: Residuen = X, Summe der Abweichungsquadrate 1, Freiheitsgrade N - 1 - k = 2: sqrt(1 / 2 / 4)
    assert A.cv_stderr(X, X[:, None], np.array([0.5]), np.array([0.0])) == pytest.approx(np.sqrt(1 / 8))


def test_spread_ist_deterministisch_und_hat_die_richtige_form(dreieck):
    tree = A.spanning_tree(dreieck.n, dreieck.edges)
    mu = A.known_means(dreieck.n, dreieck.edges, tree, 0.7)
    mo = _moments(dreieck, 0.7, ["K"])
    a = A.spread(5, 20, 50, 0.7, dreieck.n, dreieck.edges, tree, ["K"], mu, mo["beta"], mo["EX"])
    b = A.spread(5, 20, 50, 0.7, dreieck.n, dreieck.edges, tree, ["K"], mu, mo["beta"], mo["EX"])
    assert all(np.array_equal(a["est"][k], b["est"][k]) for k in a["est"]) and a["est"]["plain"].shape == (20,)
    assert set(a["coverage"]) == {"plain", "oracle", "plug"} and 0.0 <= a["nonzero_share"] <= 1.0


def test_spread_gleich_unabhaengige_neuberechnung_einer_wiederholung(dreieck):
    tree = A.spanning_tree(dreieck.n, dreieck.edges)
    mu = A.known_means(dreieck.n, dreieck.edges, tree, 0.7)
    mo = _moments(dreieck, 0.7, ["K"])
    sp = A.spread(5, 3, 40, 0.7, dreieck.n, dreieck.edges, tree, ["K"], mu, mo["beta"], mo["EX"])
    u = A.uniforms(A.stream_seed(1, 700, 40, 1, 5), 40 * 3).reshape(40, 3)                  # Wiederholung 1
    present = u < 0.7
    X = (present.sum(axis=1) >= 2).astype(float)                                           # Dreieck: mindestens 2 von 3 Kanten
    K = present.sum(axis=1).astype(float)
    assert sp["est"]["plain"][1] == pytest.approx(X.mean())
    assert sp["est"]["oracle"][1] == pytest.approx(X.mean() - mo["beta"]["K"] * (K.mean() - 2.1))


# --- CUPED ---------------------------------------------------------------------------------------------------------------------------------

def test_normalzahlen_haben_mittel_null_und_streuung_eins():
    z = A.normals(3, (200_000,))
    assert abs(z.mean()) < 0.01 and z.std() == pytest.approx(1.0, abs=0.01)
    assert np.array_equal(z, A.normals(3, (200_000,)))


def test_cuped_varianzverhaeltnis_ist_eins_minus_rho_quadrat():
    for rho in (0.2, 0.5, 0.8):
        r = A.cuped_experiment(rho, 20_000, 50, 11)
        assert r["var_fixed"] / r["var_raw"] == pytest.approx(1 - rho**2, abs=0.04)
        assert abs(r["mean_raw"]) < 0.01 and abs(r["mean_fixed"]) < 0.01 and abs(r["mean_hat"]) < 0.01
