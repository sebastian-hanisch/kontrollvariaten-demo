# Kontrollvariaten: Wissen nutzen, das man schon hat – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kontrollvariaten-demo.streamlit.app/)** -->

Drittes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Stück 2, feste Kopplung durch Spiegelung) und [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo) (Stück 1): Hier senkt eine **lernende** Korrektur die Varianz. Eine **Kontrollvariate** ist eine Größe C mit **bekanntem** Erwartungswert, die mit dem Ziel X mitschwankt; der Schätzer **X − β (C − E C)** bleibt erwartungstreu, und seine Varianz sinkt auf den Faktor **1 − R²**.

Vehikel ist wie in Stück 1 und 2 die **Zuverlässigkeit eines Gitternetzes**: Jede Kante hält unabhängig mit Wahrscheinlichkeit p; Ziel ist, dass das Netz zusammenhängt. Bis 17 Kanten ist die Wahrheit **exakt** aufgezählt, und mit ihr der optimale Koeffizient β* und der Varianzfaktor jeder Kontrollvariate.

## Warum dieses Problem

Das Rezept „ziehe β (C − E C) ab“ ist schnell erklärt, und es verschweigt zwei Dinge. Erstens: **Welche** Größe man nimmt, entscheidet mehr als wie viele. Die naheliegende Wahl (Zahl der haltenden Kanten) senkt die Varianz in der Mitte auf gut die Hälfte und bei seltenen Ausfällen praktisch gar nicht; die Zahl der **isolierten Knoten**, die den Mechanismus des seltenen Ausfalls trifft, senkt sie dort auf ein Siebenunddreißigstel. Zweitens: **β muss geschätzt werden**, und das kostet eine kleine Verzerrung, Gewinn bei kleinem N und, bei seltenen Ereignissen, **Intervalle, die weit unterdecken** – schlechter als ohne Kontrollvariate. CUPED aus der Experimente-Linie ist dieselbe Formel mit der Differenz einer Vorab-Messung zwischen zwei Armen als Kontrollvariate.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **Die Gliederung nannte „Varianzfaktor 1/(1−ρ²)“.** Richtig ist: Die **Varianz wird mit 1 − ρ² multipliziert**; 1/(1−ρ²) ist der Gewinn in „mal weniger“.
- **Fehler in der Gliederung berichtigt:** Dort stand `cuped-demo` nur als „dort Experimentdesign, hier Schätzer“. Die Demo misst aber schon 1 − ρ², die Sättigung mit mehr Vorperioden-Fenstern, die leicht liberale Fehlalarmrate durch das geschätzte θ und die Falle „Kovariate nach der Zuteilung“; „Standardfehler mit Schätzunsicherheit von θ“ und „Stratifizierung“ stehen dort als nicht gebaut. Dieses Stück wiederholt nichts davon, sondern misst die **Wahl** der Kontrollvariate bei bekanntem Erwartungswert, **Verzerrung und Überdeckung mit geschätztem β** und die Identität CUPED = Kontrollvariate.
- **Die Verzerrung des Plug-in hat kein festes Vorzeichen.** Die Vormessung bei p = 0,7 zeigte −0,36/N; bei p = 0,9 (dem Standardfall) ist sie **positiv**: +0,010 bei N = 30, das sind 8,5 Standardfehler. Der Betrag liegt bei etwa 0,3 bis 0,4 durch N, das Vorzeichen hängt vom Netz und von p ab.
- **Der Gewinn auf dem Papier ist nicht der Gewinn in der Stichprobe.** Bei seltenen Ausfällen holt das geschätzte β nur einen Teil des exakten Gewinns heraus, und das naive Intervall unterdeckt auch mit dem **exakten** β (siehe Befunde).
- **Die CUPED-Prüfung ist eine Identität, kein Orakel:** Die Gleichheit von CUPED und Kontrollvariaten-Formel ist derselbe Ausdruck; belegt wird das Varianzverhältnis gegen 1 − ρ².
- **Beim Bau gefunden:** Mit drei Kontrollvariaten und N = 30 kann die Kovarianzmatrix der Stichprobe singulär oder X konstant sein; dann ist β̂ = 0 und der Schätzer fällt auf den einfachen zurück (Anteil der Stichproben mit β̂ ≠ 0 im Preset „Kleine Stichprobe“: 0,49).

## Modell

- **Netz.** Gitter 2×3, 3×3, 2×5, 3×4 (7, 12, 13, 17 Kanten); jede Kante hält unabhängig mit Wahrscheinlichkeit p. Ziel X: das Netz hängt zusammen; gesucht ist R = E X.
- **Kontrollvariaten** (alle mit exakt bekanntem Erwartungswert): **K** = Zahl der haltenden Kanten, E K = m·p; **iso** = Zahl der isolierten Knoten (alle Kanten am Knoten fallen aus), E iso = Σ_v (1−p)^Grad(v); **tree** = ein fester Spannbaum hält ganz, E tree = p^(Kanten des Baums). Dazu die Kombination aller drei.
- **Schätzer.** Aus N Ziehungen: Mittel(X) − β (Mittel(C) − E C). Drei Schätzer nebeneinander, mit **denselben** Ziehungen: *einfach* (β = 0), *exaktes β* (β* aus der Aufzählung, Orakel) und *geschätztes β* (Plug-in: β aus derselben Stichprobe).
- **Wahrheit.** Aufzählung aller 2^m Kantenmengen mit den Gewichten p^k(1−p)^(m−k) liefert E X, die Kovarianzen und damit β* = Σ_CC⁻¹ Σ_CX und R² exakt; der **exakte Varianzfaktor** ist 1 − R².
- **Intervall.** Naives 95-%-Intervall: Schätzung ± 1,96·√(s²/N), s² = Residualvarianz der Anpassung mit N − 1 − k Freiheitsgraden.
- **Zufallsstrom.** SplitMix64, zählerbasiert (Ausgabe i hängt nur von Strom und i ab), je (Einstellung, Wiederholung) ein fester Strom.
- **CUPED.** Zwei Arme zu je 50 Einheiten, Y = ρ·X + √(1−ρ²)·e mit X = Vorab-Messung, keine Wirkung gesetzt; Kontrollvariate C = Differenz der Mittel von X (Erwartungswert 0).
- Alles ist **erfunden**; Netze und Messwerte stehen nicht für echte Daten. Alle Formeln im Expander „📐 Mathematische Formulierung“.

## Methodik

Fünf Experimente, jedes mit live berechnetem Verdikt: (1) Gewinn gegen Stichprobengröße, (2) welche Kontrollvariate (exakte Kurven aus der Aufzählung), (3) was das Schätzen von β kostet (Verzerrung und Überdeckung), (4) seltene Ausfälle, (5) CUPED als Kontrollvariate. Dazu eine Spielwiese für ein eigenes Netz (bis 17 Kanten). Die Experimente 1, 3, 4 und 5 laufen auf Knopfdruck (einige Sekunden) und werden zwischengespeichert; Experiment 2 ist exakt und steht sofort da.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`; Ströme sind fest (nie ein einzelner Seed). Das Rauschen der gemessenen Varianzverhältnisse beträgt bei 200 Wiederholungen rund 10 %, bei 300 rund 8 %: Abweichungen dieser Größe sind kein Befund; deshalb steht daneben der exakte Faktor aus der Aufzählung.

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (3×4, p = 0,90, N = 1000, „isolierte Knoten“, Strom 35) | R = 0,94451; exakter Faktor **0,232** (R² = 0,768, Korrelation −0,877, β* = −0,923); Streuung 0,00714 (einfach), 0,00325 (exaktes β), 0,00322 (geschätztes β); gemessene Varianzverhältnisse 0,21 und 0,20; Überdeckung 0,965 / 0,945 / 0,945 | `test_preset_standardfall` |
| **Seltener Ausfall** (3×4, p = 0,99, N = 3000, „isolierte Knoten“) | R = 0,99958; exakter Faktor **0,027** (Korrelation −0,986); gemessen 0,025 mit exaktem β, aber **0,28** mit geschätztem; in nur 65 % der Stichproben ist β̂ ≠ 0; Überdeckung einfach 0,66, **Orakel und Plug-in 0,035** | `test_preset_seltener_ausfall` |
| **Die naheliegende Wahl** (dasselbe, „Kantenzahl K“) | exakter Faktor **0,990** (Korrelation +0,101, β* = 0,005): die Kontrollvariate bringt praktisch nichts | `test_preset_die_naheliegende_wahl` |
| **Kleine Stichprobe, drei Kontrollvariaten** (3×4, p = 0,70, N = 30) | exakter Faktor 0,483; gemessen 0,44 mit exaktem, **0,71** mit geschätztem β; Verzerrung −0,0116; β̂ ≠ 0 in 49 % der Stichproben | `test_preset_kleine_stichprobe_drei_kontrollvariaten` |
| Gewinn gegen N (3×4, p = 0,9, „isolierte Knoten“, 300 Wdh.), N = 30 / 100 / 300 / 1000 | Varianzverhältnis mit exaktem β 0,18 / 0,27 / 0,24 / 0,21, mit geschätztem **0,37** / 0,25 / 0,24 / 0,21 (exakt 0,232); bei N = 30 ist β̂ nur in 74 % der Stichproben von 0 verschieden | `test_experiment_1_gewinn_gegen_n_bei_p_0_9` |
| **Welche Kontrollvariate?** Exakter Faktor 3×4, p = 0,5 / 0,6 / 0,7 / 0,8 / 0,9 / 0,95 / 0,99, **Kantenzahl** | 0,803 / 0,644 / **0,572** / 0,655 / 0,845 / 0,936 / **0,990** | `test_experiment_2_exakte_faktoren_am_3x4_gitter` |
| dasselbe, **isolierte Knoten** | 0,898 / 0,766 / 0,599 / 0,421 / 0,232 / 0,125 / **0,027** | dasselbe |
| dasselbe, **Spannbaum hält** | 0,992 / 0,986 / 0,976 / 0,968 / 0,973 / 0,984 / 0,996 | dasselbe |
| dasselbe, **alle drei** | 0,798 / 0,633 / **0,483** / 0,358 / 0,219 / 0,123 / 0,027 | dasselbe |
| Andere Gitter, p = 0,99, isolierte Knoten | 3×3: **0,029**; 2×3: 0,342 | `test_experiment_2_faktoren_anderer_gitter_bei_p_0_99` |
| **β schätzen**, p = 0,7, „isolierte Knoten“, N = 30 / 100 / 300 / 1000 (600 Wdh.) | Verzerrung −0,0112 / −0,0033 / −0,0001 / −0,0007 (−3,8 / −2,0 / −0,1 / −1,5 Standardfehler); Verzerrung mal N −0,34 / −0,33 / −0,04 / −0,72; Überdeckung des naiven Intervalls (einfach / exaktes β / geschätztes β) bei N = 30: 0,948 / 0,945 / **0,928** | `test_experiment_3_verzerrung_und_ueberdeckung_bei_p_0_7` |
| dasselbe bei **p = 0,9** (Standardfall) | N = 30: Verzerrung **+0,0101** (+8,5 Standardfehler), Überdeckung 0,807 / **0,415** / **0,323**; N = 100: 0,912 / 0,647 / 0,652; N = 300: 0,940 / 0,915 / 0,928; N = 1000: 0,952 / 0,938 / 0,938 | `test_experiment_3_ueberdeckung_bei_p_0_9_unterdeckt_bei_kleinem_n` |
| **Seltene Ausfälle**, p = 0,95 (q = 0,01207), N = 100 / 1000 / 10 000 (N·q = 1,2 / 12,1 / 121) | Überdeckung einfach 0,747 / 0,913 / 0,927; exaktes β **0,120 / 0,753** / 0,953; geschätztes β 0,113 / 0,747 / 0,947; Varianzverhältnis mit exaktem β 0,13 / 0,14 / 0,13 (exakt 0,125) | `test_experiment_4_seltene_ausfaelle` |
| p = 0,99 (q = 0,00042), N = 1000 / 10 000 (N·q = 0,4 / 4,2) | Varianzverhältnis exaktes β 0,018 / **0,033**, geschätztes β 0,111 / **0,147** (exakt 0,027); Anteil β̂ ≠ 0: 0,33 / 0,97; Überdeckung einfach 0,333 / **0,940**, Orakel 0,007 / **0,160**, Plug-in 0,007 / 0,160 | dasselbe |
| **CUPED** als Kontrollvariate (20 000 Experimente, 50 Einheiten je Arm), ρ = 0,2 / 0,5 / 0,8 | Varianzverhältnis CUPED / roh **0,964 / 0,745 / 0,359** (1 − ρ² = 0,960 / 0,750 / 0,360); mit geschätztem θ 0,955 / 0,738 / 0,354; die Mittel liegen bei 0 | `test_experiment_5_cuped_varianzverhaeltnis` |
| Spielwiese, Standardnetz (0-1, 1-2, 2-3, 3-0, 0-2, 2-4, 4-5, 5-2) | R = 0,94951; exakte Faktoren Kantenzahl 0,765, isolierte Knoten **0,261**, Spannbaum 0,923, alle drei 0,243; gemessen (geschätztes β) 0,26 | `test_spielwiese_standardnetz` |

## Ehrliche Grenzen

- **Nur an erfundenen Gittern geprüft.** Die Wahrheit ist exakt; ob die Wahl der Kontrollvariate bei realen Netzen genauso ausfällt, zeigt die Demo nicht. Die Zahl der isolierten Knoten wirkt an diesen Gittern (Ecken vom Grad 2) stark, am 2×3-Gitter bei p = 0,99 mit 0,342 deutlich weniger als am 3×3- und 3×4-Gitter (0,029 und 0,027).
- **Nur lineare Kontrollvariaten.** Der Gewinn ist 1 − R² der **linearen** Anpassung; nichtlineare oder gelernte Kontrollvariaten (Stein, neuronal) sind nicht gebaut.
- **Der Standardfehler ist naiv.** Er benutzt die Residualvarianz und berücksichtigt nicht, dass β geschätzt wurde; die schlechte Überdeckung bei kleinem N·q ist zum Teil eine Folge davon und zum Teil eine der schiefen Verteilung (sie tritt auch mit dem exakten β auf). Ein Intervall nach der Delta-Methode ist nicht gebaut.
- **Bei N = 100 und p = 0,99** enthalten fast alle Wiederholungen keinen Ausfall; die gemessenen Varianzverhältnisse (0,000 / 0,002) sind dort nicht belastbar und stehen nur als Hinweis.
- **Das Vorzeichen der Verzerrung hängt von Netz und p ab** (negativ bei p = 0,7, positiv bei p = 0,9); die Größenordnung ist etwa 0,3 bis 0,4 durch N.
- **Zeiten werden nicht gemessen**, nur die Zahl der Ziehungen; Kontrollvariaten kosten je Ziehung etwas Rechenzeit.
- **Der CUPED-Reiter prüft eine Identität** der Formeln (derselbe Ausdruck), keinen unabhängigen Rechenweg.
- **Literaturangaben im Mathematik-Expander** (Lavenberg und Welch 1981; Glasserman 2003; Deng, Xu, Kohavi und Walker 2013) stammen aus dem Gedächtnis und sind nicht eingesehen. Die Websuche zum Stand der Technik im Plan stützt sich auf Suchauszüge.
- **Mit echtem Mausklick** sind nur die Knöpfe der Experimente im Browser angesehen; AppTest deckt Regler, Presets, Permalink, PDF und Spielwiese ab.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom (Startwert 0xE220A8397B1DCDAF, vektorisiert gleich Ganzzahl-Referenz), Zusammenhang, Spannbaum und Spannwald, die drei Kontrollvariaten und ihre bekannten Erwartungswerte von Hand (Dreieck: E iso = 3/4), exakte Momente von Hand (Dreieck bei p = ½: Kantenzahl β* = ½ und Faktor ¼; isolierte Knoten β* = −2/5 und Faktor 2/5; Pfad mit zwei Kanten Faktor ⅓), Schätzer und Plug-in-Koeffizient an Handbeispielen und Randfällen (konstantes X, singuläre Kovarianz, zu wenige Ziehungen, kollineare Kontrollvariaten), unabhängige Neuberechnung einer Wiederholung, CUPED-Varianzverhältnis.
- **Orakel** (`tests/test_oracle_kvr.py`): Zusammenhang, isolierte Knoten und Spannwald gegen `networkx` auf zufälligen Kantenmengen; **optimales β und Faktor gegen eine gewichtete Kleinste-Quadrate-Anpassung** (`numpy.linalg.lstsq` mit √w-Skalierung, andere Formel als Kovarianzmatrix mal Kovarianzvektor; 2×3 und 3×3, alle vier Kontrollvariaten-Mengen, drei p) und gegen Bruchrechnung (`fractions`) über alle Kantenmengen; bekannte Erwartungswerte gegen die Aufzählung, auch für Netze mit Wald und Mehrfachkanten; das Orakel-Schätzverfahren ist in der Simulation erwartungstreu und seine Varianz entspricht dem Faktor.
- **Eigenschaften** (`tests/test_properties.py`): mit exaktem β ist der Schätzer im Erwartungswert exakt (gewichtete Summen über alle Kantenmengen) und seine Varianz gleich Faktor mal Var X; der Faktor liegt in [0, 1], mehr Kontrollvariaten machen ihn nie schlechter; bei einer Kontrollvariate ist er 1 − Korrelation²; invariant gegen Umskalieren und Verschieben von C und gegen die Kantenreihenfolge; β = 0 liefert den einfachen Schätzer; Richtung der Plug-in-Verzerrung bei p = 0,7 und N = 30.
- **Randfälle** (`tests/test_edge_cases.py`): Regler an den Grenzen, jedes Gitter mit jeder Kontrollvariate, nicht zusammenhängendes Netz (Faktor NaN, ehrliches Verdikt), Netz aus einer Kante (Kantenzahl erklärt alles), Eingabefehler der Spielwiese, Worst-Case am Regler-Maximum.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle (0,95; anderthalbfach und ein Zehntel) und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig, Werte innerhalb der Grenzen und Optionen; Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, Regler an Minimum und Maximum, die Experimente, jede Kontrollvariate, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests` (Stand beim Bau: 155 Tests).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `kvr_algorithm.py` | Kern: Zufallsstrom, Zusammenhang, Kontrollvariaten, Aufzählung, exakte Momente, Schätzer, CUPED |
| `kvr_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kvr_evaluation.py` | Analyse, fünf Experimente, Spielwiese, Verdikte, Zahlenformat |
| `kvr_layout.py` | Fester Kopf (Stylesheet) |
| `kvr_pdf_export.py` | PDF der aktuellen Schätzung (fpdf2) |
| `kvr_presets.py` | Permalink (Begrenzen, Einrasten), Presets, Zufallsstrom-Knopf |
| `kvr_scenario.py` | Netze: Gitter, eigene Kantenliste mit Eingabeprüfung |
| `kvr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Mini-Instanzen, Kern, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Mehrere Modelle als Kontrollvariaten (Multifidelity):** Stück 4 der Linie.
- **Importance Sampling und Kreuzentropie für seltene Ausfälle:** Stücke 7 und 8; dort wird auch das Problem der Schätzung bei kaum beobachteten Ereignissen behandelt.
- **Bedingtes Monte Carlo:** Stück 6.
- **Vorab-Kovariaten im Experiment (CUPED im Detail, CUPAC, Falle „Kovariate nach der Zuteilung“):** `cuped-demo` (Experimente-Linie, Stück 6).
- **Standardfehler mit Schätzunsicherheit von β (Delta-Methode), nichtlineare und gelernte Kontrollvariaten:** nicht gebaut.
- **Feste Kopplung durch Spiegelung und gemeinsame Zufallszahlen:** `gekoppelte-ziehungen-demo` (Stück 2).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
