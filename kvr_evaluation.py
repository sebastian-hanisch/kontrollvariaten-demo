"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Ziel X = 1, wenn das Netz zusammenhängt; gesucht ist R = E X. Der Kontrollvariaten-Schätzer ist Mittel(X) - beta (Mittel(C) - E C); seine Varianz
sinkt auf den Faktor 1 - R^2 (R^2 des besten linearen Zusammenhangs zwischen X und den Kontrollvariaten). Drei Schätzer stehen nebeneinander:
einfach (beta = 0), Orakel (exaktes beta* aus der Aufzählung) und Plug-in (beta aus derselben Stichprobe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import kvr_algorithm as A
import kvr_constants as C
from kvr_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    gitter: str = C.DEFAULT_GRID
    p: float = C.DEFAULT_P
    n: int = C.DEFAULT_N
    kontrolle: str = C.DEFAULT_CONTROL
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    names: tuple[str, ...]
    truth: float                  # exaktes R (Aufzählung)
    factor_exact: float           # exakter Varianzfaktor 1 - R^2 (NaN, wenn X keine Varianz hat)
    corr: dict                    # exakte Korrelation je Kontrollvariate mit X
    beta_star: dict               # exakter optimaler Koeffizient
    sd_plain: float
    sd_oracle: float
    sd_plug: float
    ratio_oracle: float           # gemessen: Varianz Orakel / Varianz einfach
    ratio_plug: float
    bias_plug: float              # Mittel der Plug-in-Schätzungen minus Wahrheit
    plain: tuple[float, ...]
    oracle: tuple[float, ...]
    plug: tuple[float, ...]
    coverage: dict
    nonzero_share: float


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz) -> tuple[list[int], dict]:
    """Spannbaum und Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten)."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    tree = A.spanning_tree(netz.n, netz.edges)
    return tree, A.enumerate_table(netz.n, netz.edges, tree)


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def _var_ratio(a: np.ndarray, b: np.ndarray) -> float:
    vb = float(b.var(ddof=1))
    return float(a.var(ddof=1)) / vb if vb > 0 else float("nan")


def analyse(settings: Settings, netz: Netz | None = None, tree=None, table=None) -> Analysis:
    """REPS Schätzungen aus je N Ziehungen mit drei Schätzern (einfach, Orakel-beta, Plug-in-beta), gleiche Ziehungen für alle drei."""
    netz = netz if netz is not None else named_netz(settings.gitter)
    if tree is None or table is None:
        tree, table = netz_table(netz)
    names = C.CONTROLS[settings.kontrolle]
    mo = A.exact_moments(table, netz.m, settings.p, names)
    mu = A.known_means(netz.n, netz.edges, tree, settings.p)
    sp = A.spread(settings.seed, C.REPS, settings.n, settings.p, netz.n, netz.edges, tree, names, mu, mo["beta"], mo["EX"])
    est = sp["est"]
    return Analysis(
        netz, tuple(names), mo["EX"], mo["factor"], mo["corr"], mo["beta"], float(est["plain"].std(ddof=1)), float(est["oracle"].std(ddof=1)), float(est["plug"].std(ddof=1)),
        _var_ratio(est["oracle"], est["plain"]), _var_ratio(est["plug"], est["plain"]), float(est["plug"].mean() - mo["EX"]),
        tuple(float(x) for x in est["plain"]), tuple(float(x) for x in est["oracle"]), tuple(float(x) for x in est["plug"]), sp["coverage"], sp["nonzero_share"],
    )


def verdict_main(a: Analysis, n: int) -> tuple[str, str]:
    """Senkt die Kontrollvariate die Streuung, und wie viel davon bleibt, wenn der Koeffizient geschätzt werden muss?"""
    if not np.isfinite(a.factor_exact):
        return "info", "Das Netz hängt bei diesen Einstellungen nie oder immer zusammen: das Ziel hat keine Streuung, es gibt nichts zu senken."
    kopf = (f"Mit {de(n, 0)} Ziehungen und der Kontrollvariate „{', '.join(C.CONTROL_NAMES[x] for x in a.names)}“ ist der exakte Varianzfaktor {de(a.factor_exact, 3)} "
            f"(R² = {de(1 - a.factor_exact, 3)}); gemessen wurde {de(a.ratio_oracle, 2)} mit dem exakten β und {de(a.ratio_plug, 2)} mit dem aus der Stichprobe geschätzten.")
    if a.factor_exact > 0.95:
        return "info", kopf + " Diese Größe sagt über das Ziel praktisch nichts: die Kontrollvariate bringt hier fast nichts."
    if a.ratio_plug > a.ratio_oracle * 1.5 and a.ratio_plug - a.ratio_oracle > 0.1:
        return "warning", kopf + f" Das Schätzen von β kostet einen großen Teil des Gewinns (Anteil der Stichproben mit β̂ ≠ 0: {de(a.nonzero_share, 2)})."
    return "success", kopf + f" Die Varianz sinkt auf rund {de(a.factor_exact * 100, 0)} %."


# --- Experiment 1: Streuung gegen N -----------------------------------------------------------------------------------------------------------

def n_sweep(settings: Settings, netz: Netz, tree, table) -> dict:
    names = C.CONTROLS[settings.kontrolle]
    mo = A.exact_moments(table, netz.m, settings.p, names)
    mu = A.known_means(netz.n, netz.edges, tree, settings.p)
    rows = []
    for N in C.SWEEP_N:
        sp = A.spread(settings.seed + 1, C.SWEEP_REPS, N, settings.p, netz.n, netz.edges, tree, names, mu, mo["beta"], mo["EX"])
        e = sp["est"]
        rows.append({"N": N, "ratio_oracle": _var_ratio(e["oracle"], e["plain"]), "ratio_plug": _var_ratio(e["plug"], e["plain"]),
                     "bias_plug": float(e["plug"].mean() - mo["EX"]), "nonzero": sp["nonzero_share"]})
    return {"rows": rows, "factor_exact": mo["factor"], "names": list(names)}


def verdict_n_sweep(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    worst = max(rows, key=lambda r: r["ratio_plug"] - r["ratio_oracle"])
    noise = 200 / np.sqrt(C.SWEEP_REPS - 1)
    text = (f"Exakter Faktor {de(exp['factor_exact'], 3)}. Mit dem exakten β liegt das gemessene Verhältnis zwischen {de(min(r['ratio_oracle'] for r in rows), 2)} und {de(max(r['ratio_oracle'] for r in rows), 2)}; "
            f"mit dem geschätzten zwischen {de(min(r['ratio_plug'] for r in rows), 2)} und {de(max(r['ratio_plug'] for r in rows), 2)}, am schlechtesten bei N = {de(worst['N'], 0)} "
            f"({de(worst['ratio_plug'], 2)} gegen {de(worst['ratio_oracle'], 2)}). Das Rauschen der Verhältnisse beträgt bei {C.SWEEP_REPS} Wiederholungen rund {de(noise, 0)} %.")
    return "info", text


# --- Experiment 2: Wahl der Kontrollvariate ----------------------------------------------------------------------------------------------------

def choice_curve(netz: Netz, tree, table) -> dict:
    """Exakter Varianzfaktor 1 - R^2 gegen p für jede Kontrollvariate allein und alle drei zusammen (Aufzählung)."""
    rows = []
    for p in C.CHOICE_PS:
        row = {"p": p}
        for key, names in C.CONTROLS.items():
            mo = A.exact_moments(table, netz.m, p, names)
            row[key] = mo["factor"]
            row["R"] = mo["EX"]
        rows.append(row)
    return {"rows": rows}


def verdict_choice(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    last = rows[-1]
    mid = min(rows, key=lambda r: r["K"])
    text = (f"Bei p = {de(mid['p'], 2)} gewinnt die Kantenzahl am meisten (Faktor {de(mid['K'], 3)}); bei p = {de(last['p'], 2)} (seltener Ausfall) hilft sie praktisch nicht "
            f"({de(last['K'], 3)}), die Zahl der isolierten Knoten dagegen senkt die Varianz auf {de(last['iso'], 3)}, der feste Spannbaum auf {de(last['tree'], 3)}; alle drei zusammen {de(last['alle'], 3)}.")
    return "info", text


# --- Experiment 3: Verzerrung und Überdeckung bei geschätztem beta -----------------------------------------------------------------------------

def beta_estimation(settings: Settings, netz: Netz, tree, table) -> dict:
    names = C.CONTROLS[settings.kontrolle]
    mo = A.exact_moments(table, netz.m, settings.p, names)
    mu = A.known_means(netz.n, netz.edges, tree, settings.p)
    rows = []
    for N in C.BETA_N:
        sp = A.spread(settings.seed + 2, C.BETA_REPS, N, settings.p, netz.n, netz.edges, tree, names, mu, mo["beta"], mo["EX"])
        e = sp["est"]["plug"]
        se = float(e.std(ddof=1) / np.sqrt(C.BETA_REPS))
        bias = float(e.mean() - mo["EX"])
        rows.append({"N": N, "bias": bias, "bias_se": bias / se if se > 0 else float("nan"), "bias_times_n": bias * N, "coverage": sp["coverage"],
                     "ratio_plug": _var_ratio(e, sp["est"]["plain"]), "ratio_oracle": _var_ratio(sp["est"]["oracle"], sp["est"]["plain"])})
    return {"rows": rows, "factor_exact": mo["factor"], "q": 1.0 - mo["EX"]}


def verdict_beta(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    first, last = rows[0], rows[-1]
    lo = min(min(r["coverage"].values()) for r in rows)
    text = (f"Die Verzerrung des Plug-in-Schätzers ist bei N = {de(first['N'], 0)} {de(first['bias'], 4)} ({de(first['bias_se'], 1)} Standardfehler) und bei N = {de(last['N'], 0)} {de(last['bias'], 4)} "
            f"({de(last['bias_se'], 1)}); Verzerrung mal N: {de(first['bias_times_n'], 2)} gegen {de(last['bias_times_n'], 2)}. "
            f"Die niedrigste Überdeckung des naiven 95-%-Intervalls in der Tabelle ist {de(lo, 3)}.")
    return "info", text


# --- Experiment 4: seltenes Ereignis ----------------------------------------------------------------------------------------------------------

def rare_event(settings: Settings, netz: Netz, tree, table) -> dict:
    """Orakel-β gegen Plug-in-β bei seltenen Ausfällen (p = RARE_PS), Kontrollvariate des Reglers."""
    names = C.CONTROLS[settings.kontrolle]
    blocks = []
    for p in C.RARE_PS:
        mo = A.exact_moments(table, netz.m, p, names)
        mu = A.known_means(netz.n, netz.edges, tree, p)
        rows = []
        for N in C.RARE_N:
            sp = A.spread(settings.seed + 3, C.RARE_REPS, N, p, netz.n, netz.edges, tree, names, mu, mo["beta"], mo["EX"])
            e = sp["est"]
            rows.append({"N": N, "N_q": N * (1 - mo["EX"]), "ratio_oracle": _var_ratio(e["oracle"], e["plain"]), "ratio_plug": _var_ratio(e["plug"], e["plain"]),
                         "coverage": sp["coverage"], "nonzero": sp["nonzero_share"]})
        blocks.append({"p": p, "q": 1.0 - mo["EX"], "factor_exact": mo["factor"], "rows": rows})
    return {"blocks": blocks}


def verdict_rare(exp: dict) -> tuple[str, str]:
    b = exp["blocks"][-1]
    last = b["rows"][-1]
    text = (f"Bei p = {de(b['p'], 2)} (q = {de(b['q'], 5)}) wäre der Faktor {de(b['factor_exact'], 3)}; bei N = {de(last['N'], 0)} (N·q = {de(last['N_q'], 1)}) holt der Plug-in-Schätzer {de(last['ratio_plug'], 3)} heraus, "
            f"das Orakel {de(last['ratio_oracle'], 3)}. Überdeckung des naiven Intervalls: einfach {de(last['coverage']['plain'], 2)}, Orakel {de(last['coverage']['oracle'], 2)}, Plug-in {de(last['coverage']['plug'], 2)}. "
            f"Der Anteil der Stichproben, in denen β̂ von 0 verschieden ist, liegt bei {de(last['nonzero'], 2)}.")
    return "warning", text


# --- Experiment 5: CUPED ----------------------------------------------------------------------------------------------------------------------

def cuped_tab() -> dict:
    rows = []
    for rho in C.CUPED_RHOS:
        r = A.cuped_experiment(rho, C.CUPED_EXPERIMENTS, C.CUPED_ARM, int(rho * 10) + 700)
        rows.append({"rho": rho, "ratio_fixed": r["var_fixed"] / r["var_raw"], "ratio_hat": r["var_hat"] / r["var_raw"], "exact": 1.0 - rho**2,
                     "mean_raw": r["mean_raw"], "mean_fixed": r["mean_fixed"], "mean_hat": r["mean_hat"]})
    return {"rows": rows}


def verdict_cuped(exp: dict) -> tuple[str, str]:
    rows = exp["rows"]
    teile = "; ".join(f"ρ = {de(r['rho'], 1)}: {de(r['ratio_fixed'], 3)} (Soll {de(r['exact'], 3)})" for r in rows)
    return "info", (f"CUPED ist die Kontrollvariaten-Formel D − θ(C − 0) mit C = Differenz der Vorab-Messung zwischen den Armen. Varianzverhältnis gegen die rohe Differenz: {teile}. "
                    f"Mit geschätztem θ: {', '.join(de(r['ratio_hat'], 3) for r in rows)}. Der Mittelwert bleibt bei null (keine Wirkung gesetzt).")


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz) -> dict:
    """Ein eigenes Netz: Analyse mit dem gewählten Koeffizienten-Verfahren und die exakten Faktoren aller Kontrollvariaten bei p."""
    tree, table = netz_table(netz)
    a = analyse(settings, netz=netz, tree=tree, table=table)
    factors = {key: A.exact_moments(table, netz.m, settings.p, names)["factor"] for key, names in C.CONTROLS.items()}
    return {"analysis": a, "factors": factors}
