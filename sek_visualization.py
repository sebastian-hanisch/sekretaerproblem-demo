"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import sek_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_example(values: list[float], k: int, chosen: int, best: int):
    """Eine gezogene Folge: angesehene Angebote grau, übrige blass, gewähltes grün, bestes blau (fällt beides zusammen, ist es grün)."""
    n = len(values)
    colors = [C.COLOR_SEEN if i < k else C.COLOR_OTHER for i in range(n)]
    colors[best] = C.COLOR_BEST
    colors[chosen] = C.COLOR_PICK
    fig = go.Figure(go.Bar(x=list(range(1, n + 1)), y=values, marker_color=colors, hovertemplate="Angebot %{x}: %{y:.3f}<extra></extra>"))
    if k > 0:
        fig.add_vrect(x0=0.5, x1=k + 0.5, fillcolor="rgba(150,150,150,0.12)", line_width=0, annotation_text="nur ansehen", annotation_position="top left")
    fig.update_xaxes(title_text="Reihenfolge der Angebote")
    fig.update_yaxes(title_text="Qualität (größer ist besser)", range=[0, 1.05])
    return _base(fig, height=300)


def build_success_curve(ks, exact, est, k_chosen: int, n: int, sort: int):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=exact, mode="lines", name="exakt, zufällige Reihenfolge", line=dict(color=C.COLOR_EXACT, width=3), hovertemplate="k = %{x}: %{y:.3f}<extra>exakt</extra>"))
    label = "Schätzung, gewählte Reihenfolge" + ("" if sort == 0 else f" ({sort} % am Anfang aufsteigend)")
    fig.add_trace(go.Scatter(x=ks, y=est, mode="lines+markers", name=label, line=dict(color=C.COLOR_MC, width=2), hovertemplate="k = %{x}: %{y:.3f}<extra>Schätzung</extra>"))
    fig.add_vline(x=k_chosen, line=dict(color=C.COLOR_PICK, dash="dot"), annotation_text=f"gewählt k = {k_chosen}", annotation_position="top")
    fig.add_hline(y=0.3679, line=dict(color=C.COLOR_GUARANTEE, dash="dash"), annotation_text="1/e", annotation_position="bottom right")
    fig.update_xaxes(title_text=f"nur angesehene Angebote k (von {n})")
    top = 1.05 if max(est) > 0.62 else 0.62                       # eine sortierte Folge erreicht bei k = n − 1 den Wert 1: sonst würde die Linie abgeschnitten
    fig.update_yaxes(title_text="Erfolgswahrscheinlichkeit", range=[0, top])
    return _base(fig)


def build_order(rows):
    xs = [r["Reihenfolge"] for r in rows]
    fig = go.Figure(go.Bar(x=xs, y=[r["Erfolg"] for r in rows], marker_color=C.COLOR_PICK, text=[f"{r['Erfolg'] * 100:.1f} %" for r in rows], textposition="outside",
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Erfolg (der Beste)", range=[0, 0.5])
    return _base(fig, height=320)


def build_value(rows):
    ks = [str(r["k"]) for r in rows]
    fig = go.Figure()
    for key, color in (("Erfolg", C.COLOR_BEST), ("Wert/Maximum", C.COLOR_PICK), ("unter den besten 5", C.COLOR_GUARANTEE)):
        fig.add_trace(go.Scatter(x=ks, y=[r[key] for r in rows], mode="lines+markers", name=key, line=dict(color=color, width=3), hovertemplate="k = %{x}: %{y:.3f}<extra>" + key + "</extra>"))
    fig.update_xaxes(title_text="nur angesehene Angebote k (von 100)", type="category")
    fig.update_yaxes(title_text="Anteil", range=[0, 1.0])
    return _base(fig)


def build_growth(rows):
    xs = [str(r["n"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["Erfolg"] for r in rows], mode="lines+markers", name="Erfolg beim besten k", line=dict(color=C.COLOR_BEST, width=3), hovertemplate="n = %{x}: %{y:.4f}<extra>Erfolg</extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[r["k/n"] for r in rows], mode="lines+markers", name="bestes k / n", line=dict(color=C.COLOR_PICK, width=3), hovertemplate="n = %{x}: %{y:.4f}<extra>k/n</extra>"))
    fig.add_hline(y=0.3679, line=dict(color=C.COLOR_GUARANTEE, dash="dash"), annotation_text="1/e = 0,3679", annotation_position="top left")
    fig.update_xaxes(title_text="Zahl der Angebote n", type="category")
    fig.update_yaxes(title_text="Anteil", range=[0.25, 0.55])
    return _base(fig)


def build_prophet(table):
    fig = go.Figure()
    for r in table:
        fig.add_trace(go.Bar(x=list(C.ORDNUNGEN_B), y=[r[o] for o in C.ORDNUNGEN_B], name=r["Regel"], marker_color=C.COLORS_B[r["Regel"]],
                             hovertemplate="%{x}: %{y:.3f}<extra>" + r["Regel"] + "</extra>"))
    fig.update_layout(barmode="group")
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_GUARANTEE, dash="dash"), annotation_text="Garantie 0,5", annotation_position="top left")
    fig.update_yaxes(title_text="Wert geteilt durch E[max]", range=[0, 1.1])
    return _base(fig)


def build_instances(res):
    names = list(res)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[res[k]["Mittel"] for k in names], name="Mittel", marker_color=C.COLOR_PICK, hovertemplate="%{x}: %{y:.3f}<extra>Mittel</extra>"))
    fig.add_trace(go.Bar(x=names, y=[res[k]["Minimum"] for k in names], name="Minimum", marker_color=C.COLOR_BEST, hovertemplate="%{x}: %{y:.3f}<extra>Minimum</extra>"))
    fig.update_layout(barmode="group")
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_GUARANTEE, dash="dash"), annotation_text="Garantie 0,5", annotation_position="top left")
    fig.update_yaxes(title_text="Wert geteilt durch E[max]", range=[0, 1.05])
    return _base(fig, height=320)


def build_iid(rows):
    xs = [str(r["n"]) for r in rows]
    fig = go.Figure()
    for key in ("optimal", "Schwelle E[max]/2", "Schwelle Median des Maximums"):
        color = C.COLORS_B["optimal (Rückwärtsrechnung)"] if key == "optimal" else C.COLORS_B[key]
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=key, line=dict(color=color, width=3), hovertemplate="n = %{x}: %{y:.4f}<extra>" + key + "</extra>"))
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_GUARANTEE, dash="dash"), annotation_text="Garantie 0,5", annotation_position="bottom left")
    fig.update_xaxes(title_text="Zahl der Angebote n", type="category")
    fig.update_yaxes(title_text="Wert geteilt durch E[max]", range=[0.4, 1.02])
    return _base(fig)
