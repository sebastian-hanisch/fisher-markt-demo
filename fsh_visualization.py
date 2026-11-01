"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import fsh_constants as C

FLOOR = 1e-16       # Untergrenze der logarithmischen Fehlerachse (Fehler 0 wird hier gezeichnet)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.28), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def buyer_labels(n: int) -> list[str]:
    return [f"Spediteur {i + 1}" for i in range(n)]


def slot_labels(m: int) -> list[str]:
    return [f"Slot {j + 1}" for j in range(m)]


def build_spend(spend, prices, title: str = "") -> go.Figure:
    """Ausgaben b_ij = Preis mal Menge je Käufer und Gut (Heatmap, Zahlen in den Zellen); darüber die Preise als Beschriftung der Spalten."""
    n, m = spend.shape
    cols = [f"{s}<br>p = {p:.3f}" for s, p in zip(slot_labels(m), prices)]
    fig = go.Figure(go.Heatmap(
        z=spend, x=cols, y=buyer_labels(n), colorscale=[[0, "#f4f1ea"], [1, C.COLOR_PR]], zmin=0, showscale=True, colorbar=dict(title="Ausgabe"),
        texttemplate="%{z:.2f}" if n * m <= 80 else None, hovertemplate="%{y}, %{x}<br>Ausgabe %{z:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(260, 34 * n + 120), margin=dict(l=10, r=10, t=10 if not title else 40, b=10), plot_bgcolor="rgba(0,0,0,0)", title=dict(text=title, font=dict(size=13)) if title else None)
    return lock_axes(fig)


def build_convergence(errs: dict, work: dict) -> go.Figure:
    """Relativer Preisfehler (logarithmisch) über die Arbeit: Schritte (Proportional Response, Tâtonnement) bzw. Funktionsauswertungen (Gradientenverfahren)."""
    fig = go.Figure()
    for key in ("pr", "tat", "pg"):
        y = np.maximum(np.asarray(errs[key], dtype=float), FLOOR)
        fig.add_trace(go.Scatter(x=work[key], y=y, mode="lines", name=C.METHOD_LABEL[key], line=dict(color=C.METHOD_COLOR[key], width=2),
                                 hovertemplate="%{x:.0f}<br>Fehler %{y:.2e}<extra>" + C.METHOD_LABEL[key] + "</extra>"))
    for t in C.TARGETS:
        fig.add_hline(y=t, line=dict(color="rgba(128,128,128,0.45)", width=1, dash="dot"))
    fig.update_yaxes(type="log", title="relativer Preisfehler", exponentformat="power")
    fig.update_xaxes(title="Schritte (Gradientenverfahren: Funktionsauswertungen)")
    return _base(fig, 380)


def build_tie_allocations(x1, x2, labels=("Referenz", "Proportional Response aus Zufallsstart")) -> go.Figure:
    """Zwei Zuteilungen zu denselben Preisen nebeneinander (gleiche Farbskala)."""
    n, m = x1.shape
    fig = make_subplots(rows=1, cols=2, subplot_titles=labels, horizontal_spacing=0.08)
    for col, x in enumerate((x1, x2), start=1):
        fig.add_trace(go.Heatmap(
            z=x, x=slot_labels(m), y=buyer_labels(n), colorscale=[[0, "#f4f1ea"], [1, C.COLOR_TRUE]], zmin=0, zmax=1, showscale=(col == 2), colorbar=dict(title="Anteil"),
            texttemplate="%{z:.2f}" if n * m <= 40 else None, hovertemplate="%{y}, %{x}<br>Anteil %{z:.4f}<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(260, 32 * n + 120), margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_lie_curve(ratios, utilities, honest: float, true_ratio: float) -> go.Figure:
    """Wahrer Nutzen des Käufers über das gemeldete Nutzenverhältnis Gut 2 : Gut 1 (logarithmisch); ehrlicher Nutzen und ehrliches Verhältnis markiert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratios, y=utilities, mode="lines", name="wahrer Nutzen bei Meldung", line=dict(color=C.COLOR_GAIN, width=2)))
    fig.add_hline(y=honest, line=dict(color=C.COLOR_TRUE, dash="dash", width=1.5), annotation_text="ehrlich", annotation_position="bottom right")
    fig.add_vline(x=true_ratio, line=dict(color=C.COLOR_TRUE, dash="dot", width=1.5))
    best = int(np.argmax(utilities))
    fig.add_trace(go.Scatter(x=[ratios[best]], y=[utilities[best]], mode="markers", name="beste Meldung", marker=dict(size=11, color=C.COLOR_FAIL, symbol="diamond")))
    fig.update_xaxes(type="log", title="gemeldetes Verhältnis Slot 2 : Slot 1 (ehrlich gepunktet)")
    fig.update_yaxes(title="wahrer Nutzen")
    return _base(fig, 340)


def build_table_chart(rows, key_n, key_m) -> go.Figure:
    """Median der Arbeit bis 10^-3 / 10^-6 / 10^-9 je Verfahren für eine Marktgröße (gruppierte Balken, logarithmisch; fehlende Balken = Ziel nie erreicht)."""
    sel = [r for r in rows if r["n"] == key_n and r["m"] == key_m]
    fig = go.Figure()
    for key in ("pr", "tat", "pg"):
        ys = [r[key]["median"] if r[key]["median"] is not None else None for r in sel]
        fig.add_trace(go.Bar(x=[f"≤ {r['target']:.0e}" for r in sel], y=ys, name=C.METHOD_LABEL[key], marker_color=C.METHOD_COLOR[key]))
    fig.update_yaxes(type="log", title="Median der Arbeit")
    fig.update_xaxes(title="Preisfehler")
    fig.update_layout(barmode="group")
    return _base(fig, 320)
