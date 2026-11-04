"""Kontrollvariaten: Wissen nutzen, das man schon hat - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Eine Kontrollvariate ist eine Größe mit bekanntem Erwartungswert, die mit dem Ziel
mitschwankt; der Schätzer X - beta (C - E C) bleibt für festes beta erwartungstreu, seine Varianz sinkt bei bestem beta auf den Faktor 1 - R^2. Vehikel ist die Zuverlässigkeit eines
Gitternetzes, die Wahrheit ist exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kvr_constants as C
import kvr_evaluation as ev
from kvr_evaluation import Settings, de
from kvr_layout import dunkles_design, kopf_css
from kvr_pdf_export import build_pdf
from kvr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kvr_scenario import named_netz, parse_edges
from kvr_visualization import build_beta, build_choice, build_cuped, build_n_sweep, build_netz, build_rare, build_spread

st.set_page_config(page_title="Kontrollvariaten: Wissen nutzen, das man schon hat – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(gitter):
    return ev.netz_table(named_netz(gitter))


@st.cache_data(show_spinner="Schätzungen wiederholen …")
def _analysis(settings, tree, table):
    return ev.analyse(settings, tree=tree, table=table)


@st.cache_data(show_spinner="Streuung gegen N messen …")
def _n_sweep(settings, tree, table):
    return ev.n_sweep(settings, named_netz(settings.gitter), tree, table)


@st.cache_data(show_spinner=False)
def _choice(gitter, tree, table):
    return ev.choice_curve(named_netz(gitter), tree, table)


@st.cache_data(show_spinner="Verzerrung und Überdeckung messen …")
def _beta(settings, tree, table):
    return ev.beta_estimation(settings, named_netz(settings.gitter), tree, table)


@st.cache_data(show_spinner="Seltene Ausfälle messen …")
def _rare(settings, tree, table):
    return ev.rare_event(settings, named_netz(settings.gitter), tree, table)


@st.cache_data(show_spinner=False)
def _cuped():
    return ev.cuped_tab()


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _gitter_label(key):
    rows, cols = C.GRIDS[key]
    return f"{rows}×{cols} ({rows * (cols - 1) + cols * (rows - 1)} Kanten)"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kvr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Kontrollvariaten: Wissen nutzen, das man schon hat")
st.markdown(
    """
Eine **Kontrollvariate** ist eine Größe C mit **bekanntem** Erwartungswert, die mit dem Ziel X mitschwankt. Der Schätzer **X − β (C − E C)** bleibt für festes β erwartungstreu, und seine Varianz sinkt bei bestem β auf den Faktor **1 − R²**.
Entscheidend ist, **welche** Größe man nimmt: Die naheliegende (Zahl der haltenden Kanten) hilft in der Mitte und fast gar nicht bei **seltenen Ausfällen**; die, die den **Mechanismus** des seltenen Ereignisses trifft (Zahl der isolierten Knoten), senkt die Varianz dort drastisch.
Der Preis: **β muss geschätzt werden** (kleine Verzerrung, Gewinnverlust bei kleinem N). Zusätzlich unterdeckt das naive Intervall bei seltenen Ereignissen, auch mit dem exakten β. Hier ist die Wahrheit bekannt (Zuverlässigkeit eines Gitternetzes, alle Kantenmengen aufgezählt).

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Kontrollvariate", expanded=True):
    st.markdown(
        """
1. **Ziehen:** Jede Kante hält unabhängig mit Wahrscheinlichkeit p. Eine Ziehung liefert das **Ziel** X (hängt das Netz zusammen?) und die **Kontrollvariaten**: K (Zahl haltender Kanten), iso (Zahl isolierter Knoten), tree (ein fester Spannbaum hält ganz).
2. **Bekannter Erwartungswert:** E K = m·p, E iso = Σ (1−p)^Grad, E tree = p^(Kanten des Baums) sind exakt bekannt; deshalb ist X − β (C − E C) für jedes feste β erwartungstreu.
3. **Optimales β:** β* = Cov(X, C) / Var(C). Dann sinkt die Varianz auf den Faktor **1 − R²** (R = Korrelation von X und C; bei mehreren Kontrollvariaten das R² der Anpassung).
4. **β schätzen:** Meist kennt man β* nicht und schätzt es aus **derselben** Stichprobe (Plug-in). Das ist fast so gut wie das exakte β, solange genug Ziehungen da sind und das Ziel nicht selten ist.
5. **Warum das funktioniert:** Was man über C schon weiß, muss die Stichprobe nicht mehr lernen; es bleibt nur der Teil von X, den C nicht erklärt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Kontrollvariate**")
    gitter = st.selectbox("Gitter", options("gitter_select"), format_func=_gitter_label, key="gitter_select", help="Knoten in Zeilen und Spalten, Kanten waagerecht und senkrecht. Die Wahrheit wird live aufgezählt.")
    kontrolle = st.selectbox("Kontrollvariate", options("kontrolle_select"), format_func=C.CONTROL_LABELS.get, key="kontrolle_select",
                             help="Welche Größe mit bekanntem Erwartungswert abgezogen wird: Zahl haltender Kanten, Zahl isolierter Knoten, ein fester Spannbaum hält ganz, oder alle drei gemeinsam.")
    p = st.slider("Haltewahrscheinlichkeit p je Kante", *bounds("p_slider"), key="p_slider", step=0.01, help="Jede Kante hält unabhängig mit dieser Wahrscheinlichkeit; bei p nahe 1 sind Ausfälle selten.")
    st.markdown("**Schätzung**")
    n = st.select_slider("Ziehungen N", options("n_select"), key="n_select", help="Ziehungen je Schätzung. Kontrollvariaten-Schätzer und einfacher Schätzer benutzen dieselben Ziehungen.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"gitter_select": gitter, "kontrolle_select": kontrolle, "p_slider": float(p), "n_select": int(n), "seed_input": int(seed)})

settings = Settings(str(gitter), float(p), int(n), str(kontrolle), int(seed))
tree, table = _table(settings.gitter)
a = _analysis(settings, tree, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wahrheit R", de(a.truth, 4), help=f"Exakte Zuverlässigkeit des {a.netz.name}s bei p = {de(settings.p, 2)} (Aufzählung aller Kantenmengen).")
m2.metric("Exakter Faktor", de(a.factor_exact, 3) if a.factor_exact == a.factor_exact else "–", help="Exakter Varianzfaktor 1 − R² der gewählten Kontrollvariate (aus der Aufzählung): so viel der Varianz bleibt mit dem besten β.")
m3.metric("Gemessen (β exakt)", de(a.ratio_oracle, 2) if a.ratio_oracle == a.ratio_oracle else "–", help=f"Varianz gegen den einfachen Schätzer, aus {C.REPS} Wiederholungen, mit dem exakten β*. Rauschen etwa 10 bis 15 %.")
m4.metric("Gemessen (β geschätzt)", de(a.ratio_plug, 2) if a.ratio_plug == a.ratio_plug else "–", help="Dasselbe mit β aus der Stichprobe.")
st.caption("Optimaler Koeffizient β* und Korrelation mit dem Ziel: " + "; ".join(f"{C.CONTROL_NAMES[nm]}: β* = {de(a.beta_star[nm], 3)}, Korrelation {de(a.corr[nm], 3)}" for nm in a.names) + ".")
st.plotly_chart(build_spread(a), width="stretch", key="main_chart")
st.caption("Kästen: Verteilung der Schätzungen über die Wiederholungen (links einfach, Mitte mit exaktem β, rechts mit geschätztem β; Linie: Median, Raute und Balken: Mittel und Streuung). Gestrichelt: die Wahrheit.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kontrollvariaten.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welche Kontrollvariate lohnt sich, und was kostet es, β zu schätzen?")
kind, text = ev.verdict_main(a, settings.n)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Gewinn gegen Stichprobengröße")
st.caption(f"Varianzverhältnis (gegen den einfachen Schätzer) bei N = {', '.join(str(x) for x in C.SWEEP_N)}, je {C.SWEEP_REPS} Wiederholungen mit festen Strömen; gewähltes Netz, Kontrollvariate und p. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_sweep"):
    st.session_state["sweep_done"] = True
if st.session_state.get("sweep_done"):
    sweep = _n_sweep(settings, tree, table)
    st.plotly_chart(build_n_sweep(sweep), width="stretch", key="sweep_chart")
    kind, text = ev.verdict_n_sweep(sweep)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Welche Kontrollvariate?")
st.caption("Der **exakte** Varianzfaktor 1 − R² jeder Kontrollvariate allein und aller drei zusammen gegen p, aus der Aufzählung aller Kantenmengen (kein Zufall).")
choice = _choice(settings.gitter, tree, table)
st.plotly_chart(build_choice(choice, settings.p), width="stretch", key="choice_chart")
kind, text = ev.verdict_choice(choice)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Was kostet es, β zu schätzen?")
st.caption(f"Verzerrung des Plug-in-Schätzers und Überdeckung des naiven 95-%-Intervalls bei N = {', '.join(str(x) for x in C.BETA_N)}, je {C.BETA_REPS} Wiederholungen; gewähltes Netz, Kontrollvariate und p. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_beta"):
    st.session_state["beta_done"] = True
if st.session_state.get("beta_done"):
    beta = _beta(settings, tree, table)
    st.plotly_chart(build_beta(beta), width="stretch", key="beta_chart")
    st.dataframe(
        {"N": [r["N"] for r in beta["rows"]], "Verzerrung": [de(r["bias"], 4) for r in beta["rows"]], "in Standardfehlern": [de(r["bias_se"], 1) for r in beta["rows"]],
         "Verzerrung mal N": [de(r["bias_times_n"], 2) for r in beta["rows"]], "Varianz (Orakel-β)": [de(r["ratio_oracle"], 2) for r in beta["rows"]], "Varianz (geschätztes β)": [de(r["ratio_plug"], 2) for r in beta["rows"]]},
        hide_index=True, width="stretch", key="beta_table",
    )
    kind, text = ev.verdict_beta(beta)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Seltene Ausfälle")
st.caption(f"Bei p = {', '.join(str(x).replace('.', ',') for x in C.RARE_PS)} und N = {', '.join(str(x) for x in C.RARE_N)}: Orakel-β gegen Plug-in-β, je {C.RARE_REPS} Wiederholungen. Kostet etwa eine halbe Minute.")
if st.button("▶ Messung starten", key="run_rare"):
    st.session_state["rare_done"] = True
if st.session_state.get("rare_done"):
    rare = _rare(settings, tree, table)
    cols = st.columns(len(rare["blocks"]))
    for col, block in zip(cols, rare["blocks"]):
        with col:
            st.plotly_chart(build_rare(block), width="stretch", key=f"rare_chart_{block['p']}")
    st.dataframe(
        {"p": [str(b["p"]).replace(".", ",") for b in rare["blocks"] for _ in b["rows"]], "N": [r["N"] for b in rare["blocks"] for r in b["rows"]],
         "N·q": [de(r["N_q"], 1) for b in rare["blocks"] for r in b["rows"]], "β̂ ≠ 0": [de(r["nonzero"], 2) for b in rare["blocks"] for r in b["rows"]],
         "Überdeckung einfach": [de(r["coverage"]["plain"], 2) for b in rare["blocks"] for r in b["rows"]], "Überdeckung Orakel": [de(r["coverage"]["oracle"], 2) for b in rare["blocks"] for r in b["rows"]],
         "Überdeckung Plug-in": [de(r["coverage"]["plug"], 2) for b in rare["blocks"] for r in b["rows"]]},
        hide_index=True, width="stretch", key="rare_table",
    )
    kind, text = ev.verdict_rare(rare)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: CUPED ist eine Kontrollvariate")
st.caption(
    f"Zwei Arme zu je {C.CUPED_ARM} Einheiten, {de(C.CUPED_EXPERIMENTS, 0)} Experimente je Zeile; Y = ρ·X + √(1−ρ²)·e mit X = Vorab-Messung, keine Wirkung gesetzt. Die Differenz der Vorab-Messung zwischen den Armen hat den Erwartungswert 0 "
    "(zufällige Zuteilung): sie ist eine Kontrollvariate mit bekanntem Erwartungswert. Mehr zu CUPED in der Experimente-Linie."
)
if st.button("▶ Messung starten", key="run_cuped"):
    st.session_state["cuped_done"] = True
if st.session_state.get("cuped_done"):
    cu = _cuped()
    st.plotly_chart(build_cuped(cu), width="stretch", key="cuped_chart")
    kind, text = ev.verdict_cuped(cu)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; Kontrollvariate, p, N und Strom stammen aus der Seitenleiste. "
        "Es zeigt die exakten Faktoren aller Kontrollvariaten und eine Schätzung."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-3\n3-0\n0-2\n2-4\n4-5\n5-2", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Wahrheit R", de(ca.truth, 4))
        s2.metric("Exakter Faktor (gewählt)", de(ca.factor_exact, 3) if ca.factor_exact == ca.factor_exact else "–")
        s3.metric("Gemessen (β geschätzt)", de(ca.ratio_plug, 2) if ca.ratio_plug == ca.ratio_plug else "–")
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        st.markdown("**Exakte Faktoren bei diesem p:** " + "; ".join(f"{C.CONTROL_LABELS[k]}: {de(v, 3) if v == v else '–'}" for k, v in cres["factors"].items()) + ".")
        kind, text = ev.verdict_main(ca, settings.n)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Erwartungswert der Kontrollvariate ist exakt bekannt** | Ist er nur geschätzt, wandert dieser Fehler in den Schätzer; bei einer falschen Zahl ist der Schätzer verzerrt. | – |
| **Die Kontrollvariate hängt linear am Ziel** | Der Gewinn ist 1 − R² der **linearen** Anpassung; eine nichtlineare Beziehung wird nur teilweise genutzt (Experiment 2: Spannbaum). | – |
| **β ist bekannt** | Mit geschätztem β entsteht eine Verzerrung der Größenordnung 1/N, bei kleinem N und mehreren Kontrollvariaten geht der Gewinn verloren (Experimente 1 und 3). | – |
| **Das Ereignis ist nicht selten** | Bei seltenen Ausfällen holt das geschätzte β nur einen Teil des Gewinns heraus und die Intervalle unterdecken (Experiment 4). | Stück 7 (Importance Sampling) |
| **Eine Größe mit bekanntem Erwartungswert ist verfügbar** | Sonst hilft ein billiges Modell als Kontrollvariate. | Stück 4 (Mehrere Modelle) |
| **Vorab-Kovariate vor der Zuteilung** | Bei CUPED gilt der Erwartungswert 0 nur, wenn die Kovariate vor der Zuteilung feststeht. | `cuped-demo` (Experimente 6) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Ausfälle sind **erfunden**; die Wahrheit ist die exakte Aufzählung. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten. Es gibt drei Kontrollvariaten und ihre Kombination, nur lineare Anpassung. Die Überdeckung wird für das **naive** Intervall gemessen (Schätzung ± 1,96 · Standardfehler aus der Residualvarianz), nicht für eine Delta-Methode, die die Schätzung von β berücksichtigt. Zeiten werden nicht gemessen.

**Bekannte Probleme und Unsicherheiten.** Varianzverhältnisse aus 200 oder 300 Wiederholungen rauschen um rund 10 bis 15 %. Der einfache Schätzer benutzt für sein Intervall dieselben Freiheitsgrade N − 1 − k wie die Anpassung; bei N = 30 und drei Kontrollvariaten ist sein Standardfehler dadurch um 5,6 % zu groß. Bei N = 100 und p = 0,99 enthalten die meisten Wiederholungen keinen Ausfall; das gemessene Varianzverhältnis ist dort nicht belastbar. Experiment 5 prüft nur eine **Identität** der Formeln (derselbe Ausdruck), kein unabhängiger Rechenweg; belegt ist das Varianzverhältnis. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $R = \mathbb E\,X$ für das Ziel $X \in \{0,1\}$ („das Netz hängt zusammen“). Gegeben sind Größen $C_1, \dots, C_k$ aus denselben Ziehungen mit **bekanntem** Erwartungswert $\mu = \mathbb E\,C$.

**Schätzer.** Aus $N$ Ziehungen: $\hat R_\beta = \bar X - \beta^\top (\bar C - \mu)$. Für jedes feste $\beta$ ist $\mathbb E \hat R_\beta = R$.

**Optimales $\beta$ und Varianz.** $\beta^\ast = \Sigma_{CC}^{-1}\Sigma_{CX}$, und $\operatorname{Var}\hat R_{\beta^\ast} = \tfrac1N \operatorname{Var} X\,(1 - R^2)$ mit $R^2 = \Sigma_{XC}\Sigma_{CC}^{-1}\Sigma_{CX} / \operatorname{Var} X$. Bei einer Kontrollvariate ist $R^2 = \rho^2$.

**Plug-in.** Mit $\hat\beta$ aus derselben Stichprobe ist $\hat R_{\hat\beta}$ im Allgemeinen verzerrt (Größenordnung $1/N$), weil $\hat\beta$ und $\bar C$ zusammenhängen. Der naive Standardfehler benutzt die Residualvarianz $s^2/(N - 1 - k)$.

**Die drei Kontrollvariaten.** $K = \sum_e \mathbf 1[e \text{ hält}]$ mit $\mathbb E K = mp$; $\mathrm{iso} = \#\{v : \text{alle Kanten an } v \text{ fallen aus}\}$ mit $\mathbb E\,\mathrm{iso} = \sum_v (1-p)^{\deg v}$; $\mathrm{tree} = \mathbf 1[\text{ein fester Spannbaum hält ganz}]$ mit $\mathbb E\,\mathrm{tree} = p^{n-1}$ (bei zusammenhängendem Netz).

**Wahrheit.** Die Aufzählung aller $2^m$ Kantenmengen liefert $\mathbb E X$, die Kovarianzen und damit $\beta^\ast$ und $R^2$ exakt, mit den Gewichten $p^k(1-p)^{m-k}$.

**CUPED.** Zwei Arme, zufällig zugeteilt, $D = \bar Y_B - \bar Y_A$ und $C = \bar X_B - \bar X_A$ (Vorab-Messung $X$); $\mathbb E C = 0$. Der CUPED-Schätzer $D - \theta C$ ist die Formel $\hat R_\beta$ mit $\mu = 0$ und $\beta = \theta$; mit $Y = \rho X + \sqrt{1-\rho^2}\,e$ ist die Varianz $\operatorname{Var} D\,(1-\rho^2)$.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Lavenberg und Welch 1981 (Kontrollvariaten in der Simulation); Glasserman, *Monte Carlo Methods in Financial Engineering* 2003 (Kapitel zu Kontrollvariaten); Deng, Xu, Kohavi und Walker 2013, „Improving the sensitivity of online controlled experiments by utilizing pre-experiment data“ (CUPED).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html)."
)
