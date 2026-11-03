"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import kvr_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_spread(a):
    """Verteilung der Schätzungen aus REPS Wiederholungen: einfach, mit exaktem β (Orakel) und mit geschätztem β (Plug-in); gestrichelt die Wahrheit."""
    fig = go.Figure()
    for name, series, color in (("einfach", a.plain, C.COLOR_PLAIN), ("exaktes β", a.oracle, C.COLOR_ORACLE), ("geschätztes β", a.plug, C.COLOR_CV)):
        fig.add_trace(go.Box(y=list(series), name=name, marker_color=color, boxmean="sd", boxpoints="outliers"))
    fig.add_hline(y=a.truth, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Wahrheit", annotation_position="top right")
    fig.update_yaxes(title_text="Schätzung von R")
    fig = _base(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def build_n_sweep(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [r["N"] for r in rows]
    fig.add_trace(go.Scatter(x=ns, y=[r["ratio_oracle"] for r in rows], mode="markers+lines", name="exaktes β", line=dict(color=C.COLOR_ORACLE)))
    fig.add_trace(go.Scatter(x=ns, y=[r["ratio_plug"] for r in rows], mode="markers+lines", name="geschätztes β", line=dict(color=C.COLOR_CV)))
    fig.add_hline(y=exp["factor_exact"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="exakter Faktor 1 − R²", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="einfacher Schätzer", annotation_position="top left")
    fig.update_xaxes(title_text="Ziehungen N", type="log")
    fig.update_yaxes(title_text="Varianz mit Kontrollvariate / einfach")
    return _base(fig, 330)


def build_choice(exp, p_now):
    rows = exp["rows"]
    fig = go.Figure()
    ps = [r["p"] for r in rows]
    for key, color, dash in (("K", C.COLOR_PLAIN, "solid"), ("iso", C.COLOR_CV, "solid"), ("tree", C.COLOR_OTHER, "solid"), ("alle", C.COLOR_ORACLE, "dash")):
        fig.add_trace(go.Scatter(x=ps, y=[r[key] for r in rows], mode="markers+lines", name=C.CONTROL_LABELS[key], line=dict(color=color, dash=dash)))
    fig.add_vline(x=p_now, line=dict(color=C.COLOR_OTHER, dash="dot"), annotation_text="Regler p", annotation_position="bottom right")
    fig.update_xaxes(title_text="Haltewahrscheinlichkeit p")
    fig.update_yaxes(title_text="exakter Varianzfaktor 1 − R²", type="log")
    return _base(fig, 340)


def build_beta(exp):
    rows = exp["rows"]
    fig = go.Figure()
    ns = [str(r["N"]) for r in rows]
    for key, label, color in (("plain", "einfach", C.COLOR_PLAIN), ("oracle", "exaktes β", C.COLOR_ORACLE), ("plug", "geschätztes β", C.COLOR_CV)):
        fig.add_trace(go.Scatter(x=ns, y=[r["coverage"][key] for r in rows], mode="markers+lines", name=label, line=dict(color=color)))
    fig.add_hline(y=0.95, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="Soll 0,95", annotation_position="bottom right")
    fig.update_xaxes(title_text="Ziehungen N", type="category")
    fig.update_yaxes(title_text="Überdeckung des naiven 95-%-Intervalls", range=[0, 1.02])
    return _base(fig, 320)


def build_rare(block):
    rows = block["rows"]
    fig = go.Figure()
    ns = [str(r["N"]) for r in rows]
    fig.add_trace(go.Bar(x=ns, y=[r["ratio_oracle"] for r in rows], name="exaktes β", marker_color=C.COLOR_ORACLE))
    fig.add_trace(go.Bar(x=ns, y=[r["ratio_plug"] for r in rows], name="geschätztes β", marker_color=C.COLOR_CV))
    fig.add_hline(y=block["factor_exact"], line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text="exakter Faktor", annotation_position="top right")
    fig.update_layout(barmode="group", title=dict(text=f"p = {block['p']:g}".replace(".", ","), font=dict(size=13)))
    fig.update_xaxes(title_text="Ziehungen N", type="category")
    fig.update_yaxes(title_text="Varianz mit Kontrollvariate / einfach", range=[0, 1.05])
    return _base(fig, 300)


def build_cuped(exp):
    rows = exp["rows"]
    fig = go.Figure()
    xs = [f"ρ = {r['rho']:g}".replace(".", ",") for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["ratio_fixed"] for r in rows], name="CUPED, θ = ρ", marker_color=C.COLOR_CV))
    fig.add_trace(go.Bar(x=xs, y=[r["ratio_hat"] for r in rows], name="CUPED, θ geschätzt", marker_color=C.COLOR_ORACLE))
    fig.add_trace(go.Scatter(x=xs, y=[r["exact"] for r in rows], mode="markers", name="1 − ρ²", marker=dict(symbol="line-ew", size=26, line=dict(color=C.COLOR_TRUTH, width=3))))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Varianz CUPED / roh", range=[0, 1.05])
    return _base(fig, 300)


def build_netz(netz):
    fig = go.Figure()
    xs, ys = [c[0] for c in netz.coords], [c[1] for c in netz.coords]
    ex, ey = [], []
    for u, v in netz.edges:
        ex += [xs[u], xs[v], None]
        ey += [ys[u], ys[v], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#666", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=22, color="white", line=dict(color="#222", width=2)), showlegend=False, hoverinfo="skip"))
    for i, (x, y) in enumerate(zip(xs, ys)):
        fig.add_annotation(x=x, y=y, text=str(i), showarrow=False, font=dict(size=10, color="#222"))
    pad = 0.5
    fig.update_xaxes(visible=False, range=[min(xs) - pad, max(xs) + pad])
    fig.update_yaxes(visible=False, range=[min(ys) - pad, max(ys) + pad])
    fig = _base(fig, 280)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    return fig
