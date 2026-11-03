"""Jede Zahl aus der README und den App-Texten, nachgerechnet über dieselben Auswertungsfunktionen wie die App (DEMO-PLAYBOOK §4).

Ströme sind fest (SplitMix64) und die Rechnung besteht aus Vergleichen, Zählungen und Summen: die Zahlen sind plattformunabhängig bis auf die letzte Stelle der
Gleitkommasummen. Verhältnisse verrauschter Messungen haben trotzdem Bänder (Rauschen der Varianzverhältnisse bei 200 bis 300 Wiederholungen: 8 bis 10 %).
Zeiten stehen nie in einem Test."""

import pytest

import kvr_algorithm as A
import kvr_constants as C
import kvr_evaluation as ev
from kvr_scenario import named_netz, parse_edges

STANDARD = ev.Settings()


def _setup(gitter="3x4"):
    netz = named_netz(gitter)
    tree, table = ev.netz_table(netz)
    return netz, tree, table


# --- Die vier Presets ----------------------------------------------------------------------------------------------------------------------

def test_preset_standardfall():
    a = ev.analyse(ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]))
    assert a.truth == pytest.approx(0.94451, abs=5e-6) and a.factor_exact == pytest.approx(0.2316, abs=5e-5)
    assert a.beta_star["iso"] == pytest.approx(-0.923, abs=0.002) and a.corr["iso"] == pytest.approx(-0.877, abs=0.002)
    assert [a.sd_plain, a.sd_oracle, a.sd_plug] == pytest.approx([0.00714, 0.00325, 0.00322], abs=3e-5)
    assert a.ratio_oracle == pytest.approx(0.207, abs=0.01) and a.ratio_plug == pytest.approx(0.203, abs=0.01)
    assert a.coverage == pytest.approx({"plain": 0.965, "oracle": 0.945, "plug": 0.945}, abs=0.006)
    kind, text = ev.verdict_main(a, 1000)
    assert kind == "success" and "auf rund 23 %" in text


def test_preset_seltener_ausfall():
    a = ev.analyse(ev.Settings(**C.PRESETS["Seltener Ausfall"]))
    assert a.truth == pytest.approx(0.99958, abs=5e-6) and a.factor_exact == pytest.approx(0.0269, abs=5e-5)
    assert a.corr["iso"] == pytest.approx(-0.986, abs=0.002) and a.beta_star["iso"] == pytest.approx(-0.999, abs=0.002)
    assert a.ratio_oracle == pytest.approx(0.025, abs=0.01) and a.ratio_plug == pytest.approx(0.277, abs=0.03)
    assert a.nonzero_share == pytest.approx(0.65, abs=0.02)
    assert a.coverage == pytest.approx({"plain": 0.66, "oracle": 0.035, "plug": 0.035}, abs=0.01)
    assert ev.verdict_main(a, 3000)[0] == "warning"


def test_preset_die_naheliegende_wahl():
    a = ev.analyse(ev.Settings(**C.PRESETS["Die naheliegende Wahl"]))
    assert a.factor_exact == pytest.approx(0.9899, abs=5e-5) and a.corr["K"] == pytest.approx(0.101, abs=0.002) and a.beta_star["K"] == pytest.approx(0.005, abs=0.001)
    assert a.ratio_oracle == pytest.approx(0.981, abs=0.02) and a.ratio_plug == pytest.approx(1.0, abs=0.02)
    kind, text = ev.verdict_main(a, 3000)
    assert kind == "info" and "bringt hier fast nichts" in text


def test_preset_kleine_stichprobe_drei_kontrollvariaten():
    a = ev.analyse(ev.Settings(**C.PRESETS["Kleine Stichprobe, drei Kontrollvariaten"]))
    assert a.truth == pytest.approx(0.45602, abs=5e-6) and a.factor_exact == pytest.approx(0.4828, abs=5e-5)
    assert a.beta_star == pytest.approx({"K": 0.112, "iso": -0.241, "tree": 0.047}, abs=0.002)
    assert a.corr == pytest.approx({"K": 0.654, "iso": -0.633, "tree": 0.155}, abs=0.002)
    assert a.ratio_oracle == pytest.approx(0.436, abs=0.03) and a.ratio_plug == pytest.approx(0.714, abs=0.03)
    assert a.nonzero_share == pytest.approx(0.49, abs=0.02) and a.bias_plug == pytest.approx(-0.0116, abs=0.002)
    assert ev.verdict_main(a, 30)[0] == "warning"


def test_rauschen_der_varianzverhaeltnisse():
    import numpy as np

    assert 100 / np.sqrt(C.REPS / 2) == pytest.approx(10.0) and 100 / np.sqrt(C.SWEEP_REPS / 2) == pytest.approx(8.16, abs=0.01)


# --- Experiment 1: Gewinn gegen N -------------------------------------------------------------------------------------------------------------

def test_experiment_1_gewinn_gegen_n_bei_p_0_9():
    netz, tree, table = _setup()
    exp = ev.n_sweep(STANDARD, netz, tree, table)
    assert exp["factor_exact"] == pytest.approx(0.2316, abs=5e-5)
    assert [r["N"] for r in exp["rows"]] == [30, 100, 300, 1000]
    assert [r["ratio_oracle"] for r in exp["rows"]] == pytest.approx([0.179, 0.270, 0.242, 0.211], abs=0.015)
    assert [r["ratio_plug"] for r in exp["rows"]] == pytest.approx([0.365, 0.251, 0.243, 0.212], abs=0.015)
    assert exp["rows"][0]["nonzero"] == pytest.approx(0.74, abs=0.03) and exp["rows"][1]["nonzero"] == 1.0
    assert ev.verdict_n_sweep(exp)[0] == "info"


# --- Experiment 2: Wahl der Kontrollvariate ------------------------------------------------------------------------------------------------------

def test_experiment_2_exakte_faktoren_am_3x4_gitter():
    netz, tree, table = _setup()
    exp = ev.choice_curve(netz, tree, table)
    assert [r["p"] for r in exp["rows"]] == list(C.CHOICE_PS)
    assert [r["K"] for r in exp["rows"]] == pytest.approx([0.8028, 0.6436, 0.5719, 0.6552, 0.8453, 0.9361, 0.9899], abs=5e-4)
    assert [r["iso"] for r in exp["rows"]] == pytest.approx([0.8975, 0.7658, 0.5988, 0.4205, 0.2316, 0.1247, 0.0269], abs=5e-4)
    assert [r["tree"] for r in exp["rows"]] == pytest.approx([0.9923, 0.9856, 0.9759, 0.9681, 0.9731, 0.9839, 0.9964], abs=5e-4)
    assert [r["alle"] for r in exp["rows"]] == pytest.approx([0.7975, 0.6330, 0.4828, 0.3581, 0.2185, 0.1226, 0.0269], abs=5e-4)
    assert [r["R"] for r in exp["rows"]] == pytest.approx([0.0599, 0.2019, 0.4560, 0.7464, 0.9445, 0.9879, 0.9996], abs=5e-5)
    kind, text = ev.verdict_choice(exp)
    assert "Bei p = 0,70 gewinnt die Kantenzahl am meisten (Faktor 0,572)" in text


def test_experiment_2_faktoren_anderer_gitter_bei_p_0_99():
    for gitter, erwartet in (("3x3", 0.0290), ("2x3", 0.3420)):
        netz, tree, table = _setup(gitter)
        assert A.exact_moments(table, netz.m, 0.99, ["iso"])["factor"] == pytest.approx(erwartet, abs=5e-4), gitter


# --- Experiment 3: β schätzen -----------------------------------------------------------------------------------------------------------------------

def test_experiment_3_verzerrung_und_ueberdeckung_bei_p_0_7():
    netz, tree, table = _setup()
    exp = ev.beta_estimation(ev.Settings(p=0.7), netz, tree, table)
    r = {row["N"]: row for row in exp["rows"]}
    assert r[30]["bias"] == pytest.approx(-0.01117, abs=0.0015) and r[100]["bias"] == pytest.approx(-0.00327, abs=0.0008)
    assert -0.5 < r[30]["bias_times_n"] < -0.2 and -0.5 < r[100]["bias_times_n"] < -0.15            # etwa -0,33 / N
    assert r[30]["bias_se"] < -2.5
    assert r[30]["coverage"] == pytest.approx({"plain": 0.948, "oracle": 0.945, "plug": 0.928}, abs=0.02)
    assert r[1000]["coverage"] == pytest.approx({"plain": 0.965, "oracle": 0.958, "plug": 0.952}, abs=0.02)


def test_experiment_3_ueberdeckung_bei_p_0_9_unterdeckt_bei_kleinem_n():
    netz, tree, table = _setup()
    exp = ev.beta_estimation(STANDARD, netz, tree, table)
    r = {row["N"]: row for row in exp["rows"]}
    assert r[30]["bias"] == pytest.approx(0.01005, abs=0.0015) and r[30]["bias_se"] > 4                  # bei p = 0,9 positiv (anderes Vorzeichen als bei p = 0,7)
    assert r[30]["coverage"] == pytest.approx({"plain": 0.807, "oracle": 0.415, "plug": 0.323}, abs=0.03)
    assert r[100]["coverage"] == pytest.approx({"plain": 0.912, "oracle": 0.647, "plug": 0.652}, abs=0.03)
    assert r[300]["coverage"] == pytest.approx({"plain": 0.94, "oracle": 0.915, "plug": 0.928}, abs=0.03)
    assert r[1000]["coverage"] == pytest.approx({"plain": 0.952, "oracle": 0.938, "plug": 0.938}, abs=0.03)
    assert r[30]["ratio_oracle"] == pytest.approx(0.233, abs=0.03) and r[30]["ratio_plug"] == pytest.approx(0.438, abs=0.04)


# --- Experiment 4: seltene Ausfälle --------------------------------------------------------------------------------------------------------------

def test_experiment_4_seltene_ausfaelle():
    netz, tree, table = _setup()
    exp = ev.rare_event(STANDARD, netz, tree, table)
    b95, b99 = exp["blocks"]
    assert (b95["p"], b99["p"]) == (0.95, 0.99) and b95["q"] == pytest.approx(0.01207, abs=5e-6) and b99["q"] == pytest.approx(0.00042, abs=5e-6)
    assert b95["factor_exact"] == pytest.approx(0.1247, abs=5e-5) and b99["factor_exact"] == pytest.approx(0.0269, abs=5e-5)
    r95 = {r["N"]: r for r in b95["rows"]}
    assert r95[1000]["N_q"] == pytest.approx(12.07, abs=0.01) and r95[1000]["ratio_plug"] == pytest.approx(0.141, abs=0.03)
    assert r95[1000]["coverage"] == pytest.approx({"plain": 0.913, "oracle": 0.753, "plug": 0.747}, abs=0.04)
    assert r95[10000]["coverage"] == pytest.approx({"plain": 0.927, "oracle": 0.953, "plug": 0.947}, abs=0.04)
    r99 = {r["N"]: r for r in b99["rows"]}
    assert r99[1000]["nonzero"] == pytest.approx(0.33, abs=0.05) and r99[10000]["nonzero"] == pytest.approx(0.97, abs=0.03)
    assert r99[10000]["N_q"] == pytest.approx(4.17, abs=0.01)
    assert r99[10000]["ratio_oracle"] == pytest.approx(0.033, abs=0.02) and r99[10000]["ratio_plug"] == pytest.approx(0.147, abs=0.04)
    assert r99[10000]["coverage"] == pytest.approx({"plain": 0.94, "oracle": 0.16, "plug": 0.16}, abs=0.06)
    assert r99[100]["coverage"]["plain"] == pytest.approx(0.027, abs=0.03)                                  # fast nie ein Ausfall
    assert ev.verdict_rare(exp)[0] == "warning"


# --- Experiment 5: CUPED ----------------------------------------------------------------------------------------------------------------------------

def test_experiment_5_cuped_varianzverhaeltnis():
    exp = ev.cuped_tab()
    assert [r["rho"] for r in exp["rows"]] == [0.2, 0.5, 0.8]
    assert [r["exact"] for r in exp["rows"]] == pytest.approx([0.96, 0.75, 0.36])
    assert [r["ratio_fixed"] for r in exp["rows"]] == pytest.approx([0.964, 0.745, 0.359], abs=0.03)
    assert [r["ratio_hat"] for r in exp["rows"]] == pytest.approx([0.955, 0.738, 0.354], abs=0.03)
    assert all(abs(r["mean_raw"]) < 0.01 and abs(r["mean_fixed"]) < 0.01 for r in exp["rows"])
    assert ev.verdict_cuped(exp)[0] == "info"


# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------------

def test_spielwiese_standardnetz():
    res = ev.custom_result(STANDARD, parse_edges("0-1\n1-2\n2-3\n3-0\n0-2\n2-4\n4-5\n5-2"))
    a = res["analysis"]
    assert a.netz.m == 8 and a.netz.n == 6 and a.truth == pytest.approx(0.94951, abs=5e-6)
    assert res["factors"] == pytest.approx({"K": 0.7654, "iso": 0.2606, "tree": 0.9233, "alle": 0.2431}, abs=5e-4)
    assert a.ratio_plug == pytest.approx(0.258, abs=0.03)


def test_gitter_kantenzahlen():
    assert [named_netz(k).m for k in C.GRIDS] == [7, 12, 13, 17] and C.LIVE_EDGE_LIMIT == 17
