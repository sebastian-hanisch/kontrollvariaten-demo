"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kvr_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_sweep", "Exakter Faktor 0,232"),
    ("run_beta", "Die Verzerrung des Plug-in-Schätzers ist bei N = 30"),
    ("run_rare", "Bei p = 0,99 (q = 0,00042) wäre der Faktor 0,027"),
    ("run_cuped", "CUPED ist die Kontrollvariaten-Formel"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_wahl_experiment_steht_ohne_klick_da():
    assert "Bei p = 0,70 gewinnt die Kantenzahl am meisten (Faktor 0,572)" in _texte(_app())


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_cuped").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)
    assert not at.exception
    assert "CUPED ist die Kontrollvariaten-Formel" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Die Varianz sinkt auf rund 23 %" in _texte(at)
    at.sidebar.button(key="preset_Die naheliegende Wahl").click().run(timeout=TIMEOUT)
    assert "bringt hier fast nichts" in _texte(at)
    at.sidebar.button(key="preset_Kleine Stichprobe, drei Kontrollvariaten").click().run(timeout=TIMEOUT)
    assert "kostet einen großen Teil des Gewinns" in _texte(at)


def test_jede_kontrollvariate_laeuft_durch():
    at = _app()
    for kontrolle in C.CONTROLS:
        at.sidebar.selectbox(key="kontrolle_select").set_value(kontrolle).run(timeout=TIMEOUT)
        assert not at.exception, (kontrolle, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    assert "Exakte Faktoren bei diesem p" in " ".join(m.value for m in at.markdown)
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_nicht_zusammenhaengendes_netz_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n2-3").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "nie oder immer" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="n_select").set_value(300).run(timeout=TIMEOUT)     # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["p"] = "5"
    at.query_params["n"] = "2400"
    at.query_params["g"] = "9x9"            # unbekanntes Gitter: verworfen
    at.query_params["k"] = "quatsch"        # unbekannte Kontrollvariate: verworfen
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="p_slider").value == C.P_MAX
    assert at.sidebar.select_slider(key="n_select").value == 3000                    # 2400 liegt näher an 3000 als an 1000
    assert at.sidebar.selectbox(key="gitter_select").value == C.DEFAULT_GRID
    assert at.sidebar.selectbox(key="kontrolle_select").value == C.DEFAULT_CONTROL
    assert at.sidebar.number_input(key="seed_input").value == 0
