"""Randfälle: Grenzen der Regler, entartete Netze, Eingabefehler der Spielwiese."""

import math

import numpy as np
import pytest

import kvr_algorithm as A
import kvr_constants as C
import kvr_evaluation as ev
from kvr_scenario import Netz, named_netz, parse_edges


def test_kleinste_und_groesste_ziehungszahl_laufen_durch():
    for n in (C.N_MIN, C.N_MAX):
        a = ev.analyse(ev.Settings(n=n))
        assert 0.0 <= min(a.plain) and max(a.plain) <= 1.0 and np.isfinite(a.sd_plain)


def test_hoechstes_und_niedrigstes_p_laufen_durch():
    for p in (C.P_MIN, C.P_MAX):
        for kontrolle in C.CONTROLS:
            a = ev.analyse(ev.Settings(p=p, kontrolle=kontrolle, n=100))
            assert 0.0 < a.truth < 1.0


@pytest.mark.parametrize("kontrolle", list(C.CONTROLS))
def test_jedes_gitter_und_jede_kontrollvariate_laeuft_durch(kontrolle):
    for key in C.GRIDS:
        a = ev.analyse(ev.Settings(gitter=key, kontrolle=kontrolle, n=100))
        assert 0.0 < a.truth < 1.0 and 0.0 <= a.factor_exact <= 1.0


def test_nicht_zusammenhaengendes_netz_hat_keine_streuung_und_ein_ehrliches_verdikt():
    netz = Netz("zwei Paare", 4, ((0, 1), (2, 3)), tuple((float(i), 0.0) for i in range(4)))
    a = ev.analyse(ev.Settings(n=100), netz=netz, tree=A.spanning_tree(4, netz.edges), table=A.enumerate_table(4, netz.edges, A.spanning_tree(4, netz.edges)))
    assert a.truth == 0.0 and math.isnan(a.factor_exact) and a.sd_plain == 0.0 and a.bias_plug == 0.0
    kind, text = ev.verdict_main(a, 100)
    assert kind == "info" and "nie oder immer" in text


def test_netz_aus_einer_kante():
    netz = Netz("Kante", 2, ((0, 1),), ((0.0, 0.0), (1.0, 0.0)))
    tree = A.spanning_tree(2, netz.edges)
    table = A.enumerate_table(2, netz.edges, tree)
    mo = A.exact_moments(table, 1, 0.3, ["K"])
    assert mo["EX"] == pytest.approx(0.3) and mo["factor"] == pytest.approx(0.0, abs=1e-12)        # K = X: die Kontrollvariate erklärt alles


def test_netz_oberhalb_der_aufzaehlungsgrenze_wird_abgelehnt():
    edges = tuple((i, j) for i in range(6) for j in range(i + 1, 7))[: C.LIVE_EDGE_LIMIT + 1]       # Teil des K7: 18 Kanten
    with pytest.raises(ValueError):
        ev.netz_table(Netz("zu gross", 7, edges, tuple((0.0, 0.0) for _ in range(7))))


def test_eigenes_netz_laeuft_durch_und_nennt_alle_faktoren():
    res = ev.custom_result(ev.Settings(p=0.9, n=100), parse_edges("0-1\n1-2\n2-3\n3-0\n0-2"))
    assert set(res["factors"]) == set(C.CONTROLS) and all(0.0 <= v <= 1.0 for v in res["factors"].values())
    assert res["analysis"].truth > 0.9


@pytest.mark.parametrize("text,fragment", [
    ("", "Keine Kanten"),
    ("# nur Kommentar\n\n", "Keine Kanten"),
    ("0-0", "Schleife"),
    ("0-1\n1-0", "Doppelte Kante"),
    ("0-1\nabc", "Zeile 2"),
    ("0-1-2", "Zeile 1"),
    ("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)), "Kanten: höchstens"),
    ("\n".join(f"0-{i}" for i in range(1, C.CUSTOM_NODE_LIMIT + 1)), "Knoten: höchstens"),
])
def test_eingabefehler_der_spielwiese_haben_eine_klare_meldung(text, fragment):
    with pytest.raises(ValueError) as err:
        parse_edges(text)
    assert fragment in str(err.value)


def test_knoten_werden_von_null_an_umnummeriert():
    netz = parse_edges("10-20\n20-30")
    assert netz.n == 3 and netz.edges == ((0, 1), (1, 2))


def test_worst_case_am_regler_maximum_laeuft_durch():
    """Hänger-Netz am Slider-Maximum (kein Zeittest, nur ein Netz): 3x4, 3000 Ziehungen, alle drei Kontrollvariaten."""
    a = ev.analyse(ev.Settings(n=C.N_MAX, kontrolle="alle"))
    assert len(a.plain) == C.REPS and np.isfinite(a.ratio_plug)


def test_named_netz_kennt_alle_gitter():
    assert [named_netz(k).m for k in C.GRIDS] == [7, 12, 13, 17]
