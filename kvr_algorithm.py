"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom       stream_seed, uniforms, edge_states         SplitMix64, zählerbasiert, ein fester Strom je (Einstellung, Wiederholung)
  Ziel und Größen    connected, control_values, known_means     X = Netz hängt zusammen; Kontrollvariaten K, iso, tree mit exakt bekanntem Erwartungswert
  Wahrheit           enumerate_table, exact_moments             Aufzählung aller 2^m Kantenmengen: exakte Erwartungswerte, optimales beta, Faktor 1 - R^2
  Schätzer           cv_estimate, plugin_beta, cv_stderr         Kontrollvariaten-Schätzer mit festem und mit geschätztem Koeffizienten
  Wiederholungen     spread                                      einfach / Orakel-beta / Plug-in-beta über feste Ströme, mit Überdeckung
  CUPED              normals, cuped_experiment                   dieselbe Formel mit der Differenz einer Vorab-Messung zwischen zwei Armen
Zufall kommt nur über einen übergebenen Strom (seed), nie aus einem globalen Generator.
"""

from __future__ import annotations

import numpy as np

import kvr_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def edge_states(seed: int, N: int, p: float, m: int) -> np.ndarray:
    """(N, m) bool: Kante e der Ziehung i hält genau dann, wenn ihre Zufallszahl (Nr. i*m + e des Stroms) unter p liegt."""
    return uniforms(seed, N * m).reshape(N, m) < p


# --- Ziel und Kontrollvariaten ---------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def spanning_tree(n: int, edges) -> list[int]:
    """Indizes der Kanten eines Spannbaums (Kruskal in Listenreihenfolge); bei nicht zusammenhängendem Netz ein Spannwald."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []
    for e, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            tree.append(e)
    return tree


def degrees(n: int, edges) -> np.ndarray:
    deg = np.zeros(n)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def control_values(present: np.ndarray, n: int, edges, tree_idx) -> dict:
    """X und die Kontrollvariaten je Zeile (alles float): K = Zahl haltender Kanten, iso = Zahl isolierter Knoten, tree = fester Spannbaum hält ganz."""
    inc = np.zeros((len(edges), n), dtype=np.int32)
    for e, (u, v) in enumerate(edges):
        inc[e, u] = inc[e, v] = 1
    up_deg = present.astype(np.int32) @ inc
    return {
        "X": connected(present, n, edges).astype(np.float64),
        "K": present.sum(axis=1).astype(np.float64),
        "iso": (up_deg == 0).sum(axis=1).astype(np.float64),
        "tree": present[:, list(tree_idx)].all(axis=1).astype(np.float64),
    }


def known_means(n: int, edges, tree_idx, p: float) -> dict:
    """Exakt bekannte Erwartungswerte der Kontrollvariaten: E K = m p, E iso = Summe_v (1-p)^Grad(v), E tree = p^(Kanten des Spannbaums)."""
    return {"K": len(edges) * p, "iso": float(np.sum((1.0 - p) ** degrees(n, edges))), "tree": p ** len(tree_idx)}


# --- Wahrheit durch Aufzählung ---------------------------------------------------------------------------------------------------------

def enumerate_table(n: int, edges, tree_idx, chunk: int = 1 << 17) -> dict:
    """Alle 2^m Kantenmengen: Zahl haltender Kanten k und die Werte von X, K, iso, tree (Grundlage der exakten gewichteten Momente)."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    cols = {"k": [], "X": [], "K": [], "iso": [], "tree": []}
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        v = control_values(present, n, edges, tree_idx)
        cols["k"].append(present.sum(axis=1))
        for key in ("X", "K", "iso", "tree"):
            cols[key].append(v[key])
    return {k: np.concatenate(v) for k, v in cols.items()}


def exact_moments(table: dict, m: int, p: float, names) -> dict:
    """Exakte Momente aus der Aufzählung: E X, Var X, Erwartungswerte der Kontrollvariaten, optimaler Koeffizient beta*, R^2 und Varianzfaktor 1 - R^2.

    Hat X keine Varianz (das Netz hängt nie oder immer zusammen), sind R^2 und der Faktor NaN."""
    k = table["k"]
    w = p**k * (1.0 - p) ** (m - k)
    X = table["X"]
    mean_x = float(np.sum(w * X))
    var_x = float(np.sum(w * X * X) - mean_x**2)
    names = list(names)
    out = {"EX": mean_x, "VarX": var_x, "names": names}
    if not names:
        return out
    Cm = np.stack([table[nm] for nm in names], axis=1)
    mean_c = (w[:, None] * Cm).sum(axis=0)
    cc = (w[:, None, None] * Cm[:, :, None] * Cm[:, None, :]).sum(axis=0) - np.outer(mean_c, mean_c)
    cx = (w[:, None] * Cm * X[:, None]).sum(axis=0) - mean_c * mean_x
    beta = np.linalg.lstsq(cc, cx, rcond=None)[0]
    r2 = float(cx @ beta / var_x) if var_x > 1e-15 else float("nan")
    out.update({
        "EC": dict(zip(names, mean_c.tolist())), "beta": dict(zip(names, beta.tolist())), "R2": r2, "factor": 1.0 - r2,
        "corr": {nm: (float(cx[i] / np.sqrt(var_x * cc[i, i])) if var_x > 1e-15 and cc[i, i] > 1e-15 else float("nan")) for i, nm in enumerate(names)},
    })
    return out


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

def cv_estimate(X: np.ndarray, Cm: np.ndarray, mu: np.ndarray, beta: np.ndarray) -> float:
    """Kontrollvariaten-Schätzer: Mittel von X minus beta * (Mittel von C - mu). Bei beta = 0 der einfache Mittelwert."""
    return float(X.mean() - (Cm.mean(axis=0) - mu) @ beta)


def plugin_beta(X: np.ndarray, Cm: np.ndarray) -> np.ndarray:
    """Koeffizient aus der Stichprobe: Kovarianzmatrix der Kontrollvariaten mal Kovarianzvektor mit X. Null, wenn X oder C in der Stichprobe konstant sind."""
    k = Cm.shape[1]
    N = len(X)
    if N < 3 or X.std() == 0:
        return np.zeros(k)
    Cc = Cm - Cm.mean(axis=0)
    cov_cc = Cc.T @ Cc / (N - 1)
    cov_cx = Cc.T @ (X - X.mean()) / (N - 1)
    if np.linalg.matrix_rank(cov_cc) < k:
        return np.zeros(k)
    return np.linalg.solve(cov_cc, cov_cx)


def cv_stderr(X: np.ndarray, Cm: np.ndarray, mu: np.ndarray, beta: np.ndarray) -> float:
    """Naiver Standardfehler: Residualvarianz der Anpassung geteilt durch N (Freiheitsgrade N - 1 - Zahl der Kontrollvariaten, mindestens 1)."""
    N, k = Cm.shape
    resid = X - (Cm - mu) @ beta
    dof = max(N - 1 - k, 1)
    return float(np.sqrt(((resid - resid.mean()) ** 2).sum() / dof / N))


def spread(seed: int, reps: int, N: int, p: float, n: int, edges, tree_idx, names, mu: dict, beta_star: dict, truth: float) -> dict:
    """Je Wiederholung drei Schätzungen aus denselben N Ziehungen: einfach, mit dem exakten beta* (Orakel) und mit beta aus der Stichprobe (Plug-in).

    Rückgabe: Schätzreihen, Überdeckung des naiven 95-%-Intervalls je Schätzer und der Anteil der Wiederholungen, in denen beta aus der Stichprobe von 0 verschieden ist."""
    names = list(names)
    mu_v = np.array([mu[nm] for nm in names])
    b_star = np.array([beta_star[nm] for nm in names])
    est = {"plain": np.empty(reps), "oracle": np.empty(reps), "plug": np.empty(reps)}
    cover = {"plain": 0, "oracle": 0, "plug": 0}
    nonzero = 0
    zeros = np.zeros(len(names))
    for rep in range(reps):
        v = control_values(edge_states(stream_seed(1, int(p * 1000), N, rep, seed), N, p, len(edges)), n, edges, tree_idx)
        X = v["X"]
        Cm = np.stack([v[nm] for nm in names], axis=1)
        bh = plugin_beta(X, Cm)
        nonzero += bool(np.any(bh != 0))
        for key, b in (("plain", zeros), ("oracle", b_star), ("plug", bh)):
            e = cv_estimate(X, Cm, mu_v, b)
            est[key][rep] = e
            cover[key] += abs(e - truth) <= C.Z95 * cv_stderr(X, Cm, mu_v, b)
    return {"est": est, "coverage": {k: v / reps for k, v in cover.items()}, "nonzero_share": nonzero / reps}


# --- CUPED: dieselbe Formel ----------------------------------------------------------------------------------------------------------------

def normals(seed: int, shape) -> np.ndarray:
    """Standardnormalzahlen aus dem Strom (Box-Muller aus je zwei Gleichverteilten)."""
    count = int(np.prod(shape))
    u = uniforms(seed, 2 * count)
    u1, u2 = 1.0 - u[:count], u[count:]                 # u1 in (0, 1]
    return (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).reshape(shape)


def cuped_experiment(rho: float, M: int, arm: int, seed: int) -> dict:
    """M Experimente mit je zwei Armen zu `arm` Einheiten; Y = rho X + sqrt(1 - rho^2) e mit X = Vorab-Messung, keine Wirkung gesetzt.

    Rohe Differenz D der Armmittel; Kontrollvariate C = Differenz der Mittel von X (Erwartungswert 0, weil zufällig zugeteilt).
    CUPED mit festem theta = rho ist die Kontrollvariaten-Formel D - theta (C - 0); mit geschätztem theta (gepoolte Steigung) ebenfalls."""
    x = normals(stream_seed(seed, 1), (M, 2, arm))
    e = normals(stream_seed(seed, 2), (M, 2, arm))
    y = rho * x + np.sqrt(1.0 - rho**2) * e
    d = y[:, 1].mean(axis=1) - y[:, 0].mean(axis=1)
    c = x[:, 1].mean(axis=1) - x[:, 0].mean(axis=1)
    xs, ys = x.reshape(M, -1), y.reshape(M, -1)
    xc, yc = xs - xs.mean(axis=1, keepdims=True), ys - ys.mean(axis=1, keepdims=True)
    theta_hat = (xc * yc).sum(axis=1) / (xc**2).sum(axis=1)
    cuped_fixed, cuped_hat = d - rho * c, d - theta_hat * c
    return {
        "var_raw": float(d.var(ddof=1)), "var_fixed": float(cuped_fixed.var(ddof=1)), "var_hat": float(cuped_hat.var(ddof=1)),
        "mean_raw": float(d.mean()), "mean_fixed": float(cuped_fixed.mean()), "mean_hat": float(cuped_hat.mean()),
    }
