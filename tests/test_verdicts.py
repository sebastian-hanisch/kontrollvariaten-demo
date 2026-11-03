"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt gesetzte Werte, keine Seeds. Damit prüft der Test die Grenzen und jede genannte Zahl direkt."""

import kvr_evaluation as ev
from kvr_scenario import named_netz


def _analysis(factor, ratio_oracle, ratio_plug, names=("iso",), nonzero=1.0):
    return ev.Analysis(named_netz("2x3"), tuple(names), 0.9, factor, {nm: -0.8 for nm in names}, {nm: -0.4 for nm in names}, 0.02, 0.01, 0.011,
                       ratio_oracle, ratio_plug, -0.001, (0.9,), (0.9,), (0.9,), {"plain": 0.9, "oracle": 0.9, "plug": 0.9}, nonzero)


def test_zahlenformat_ist_deutsch():
    assert ev.de(1234567.891, 2) == "1.234.567,89" and ev.de(0.05, 3) == "0,050" and ev.de(-0.5, 1) == "−0,5" and ev.de_e(4.17e-10) == "4,2e-10"


def test_hauptverdikt_gewinn_nennt_alle_zahlen():
    kind, text = ev.verdict_main(_analysis(0.232, 0.21, 0.20), 1000)
    assert kind == "success"
    for teil in ("Mit 1.000 Ziehungen", "„Zahl der isolierten Knoten“", "exakte Varianzfaktor 0,232", "R² = 0,768", "0,21 mit dem exakten β", "0,20 mit dem aus der Stichprobe", "auf rund 23 %"):
        assert teil in text, teil


def test_hauptverdikt_schwelle_bei_0_95():
    assert ev.verdict_main(_analysis(0.95, 0.95, 0.95), 100)[0] == "success"            # genau 0,95: noch ein (kleiner) Gewinn
    kind, text = ev.verdict_main(_analysis(0.951, 0.95, 0.95), 100)
    assert kind == "info" and "bringt hier fast nichts" in text


def test_hauptverdikt_warnt_wenn_das_schaetzen_den_gewinn_frisst():
    kind, text = ev.verdict_main(_analysis(0.483, 0.49, 0.794, names=("K", "iso", "tree"), nonzero=0.48), 30)
    assert kind == "warning" and "kostet einen großen Teil des Gewinns" in text and "0,48" in text
    assert "Zahl der haltenden Kanten, Zahl der isolierten Knoten, ein fester Spannbaum hält ganz" in text


def test_hauptverdikt_warnschwellen_sind_anderthalbfach_und_ein_zehntel():
    assert ev.verdict_main(_analysis(0.2, 0.2, 0.30), 100)[0] == "success"              # Abstand 0,1: nicht größer als 0,1
    assert ev.verdict_main(_analysis(0.2, 0.2, 0.301), 100)[0] == "warning"
    assert ev.verdict_main(_analysis(0.8, 0.8, 1.15), 100)[0] == "success"              # 1,15 ist nicht mehr als 1,5 * 0,8 = 1,2
    assert ev.verdict_main(_analysis(0.8, 0.8, 1.21), 100)[0] == "warning"


def test_hauptverdikt_ohne_streuung():
    kind, text = ev.verdict_main(_analysis(float("nan"), float("nan"), float("nan")), 100)
    assert kind == "info" and "nie oder immer" in text


def test_n_sweep_verdikt_nennt_bereiche_und_den_schlechtesten_fall():
    rows = [{"N": 30, "ratio_oracle": 0.49, "ratio_plug": 0.79}, {"N": 100, "ratio_oracle": 0.46, "ratio_plug": 0.56}, {"N": 1000, "ratio_oracle": 0.445, "ratio_plug": 0.448}]
    kind, text = ev.verdict_n_sweep({"rows": rows, "factor_exact": 0.483})
    assert kind == "info"
    for teil in ("Exakter Faktor 0,483", "zwischen 0,45 und 0,49", "zwischen 0,45 und 0,79", "am schlechtesten bei N = 30 (0,79 gegen 0,49)", "rund 8 %"):
        assert teil in text, teil


def test_wahl_verdikt():
    rows = [{"p": 0.5, "K": 0.80, "iso": 0.90, "tree": 0.99, "alle": 0.80}, {"p": 0.7, "K": 0.572, "iso": 0.599, "tree": 0.976, "alle": 0.483},
            {"p": 0.99, "K": 0.990, "iso": 0.027, "tree": 0.996, "alle": 0.027}]
    kind, text = ev.verdict_choice({"rows": rows})
    assert kind == "info"
    for teil in ("Bei p = 0,70 gewinnt die Kantenzahl am meisten (Faktor 0,572)", "bei p = 0,99 (seltener Ausfall) hilft sie praktisch nicht (0,990)", "auf 0,027, der feste Spannbaum auf 0,996", "alle drei zusammen 0,027"):
        assert teil in text, teil


def test_beta_verdikt_nennt_verzerrung_und_niedrigste_ueberdeckung():
    rows = [{"N": 30, "bias": -0.012, "bias_se": -5.2, "bias_times_n": -0.36, "coverage": {"plain": 0.933, "oracle": 0.931, "plug": 0.918}},
            {"N": 1000, "bias": -0.0005, "bias_se": -1.4, "bias_times_n": -0.52, "coverage": {"plain": 0.948, "oracle": 0.958, "plug": 0.955}}]
    kind, text = ev.verdict_beta({"rows": rows})
    assert kind == "info"
    for teil in ("bei N = 30 −0,0120 (−5,2 Standardfehler)", "bei N = 1.000 −0,0005 (−1,4)", "−0,36 gegen −0,52", "niedrigste Überdeckung des naiven 95-%-Intervalls in der Tabelle ist 0,918"):
        assert teil in text, teil


def test_seltenheits_verdikt_nennt_faktor_ueberdeckungen_und_anteil():
    block = {"p": 0.99, "q": 0.00042, "factor_exact": 0.027,
             "rows": [{"N": 1000, "N_q": 0.4, "ratio_oracle": 0.03, "ratio_plug": 0.127, "coverage": {"plain": 0.35, "oracle": 0.01, "plug": 0.01}, "nonzero": 0.34},
                      {"N": 10000, "N_q": 4.2, "ratio_oracle": 0.028, "ratio_plug": 0.124, "coverage": {"plain": 0.93, "oracle": 0.13, "plug": 0.12}, "nonzero": 0.97}]}
    kind, text = ev.verdict_rare({"blocks": [block]})
    assert kind == "warning"
    for teil in ("p = 0,99 (q = 0,00042)", "wäre der Faktor 0,027", "bei N = 10.000 (N·q = 4,2)", "holt der Plug-in-Schätzer 0,124", "das Orakel 0,028", "einfach 0,93, Orakel 0,13, Plug-in 0,12", "bei 0,97"):
        assert teil in text, teil


def test_cuped_verdikt():
    rows = [{"rho": 0.2, "ratio_fixed": 0.963, "ratio_hat": 0.953, "exact": 0.96}, {"rho": 0.5, "ratio_fixed": 0.752, "ratio_hat": 0.745, "exact": 0.75}]
    kind, text = ev.verdict_cuped({"rows": rows})
    assert kind == "info" and "ρ = 0,2: 0,963 (Soll 0,960); ρ = 0,5: 0,752 (Soll 0,750)" in text and "Mit geschätztem θ: 0,953, 0,745" in text
