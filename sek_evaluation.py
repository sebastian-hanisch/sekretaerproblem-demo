"""Kennzahlen, Verdikt-Logik und Experimente. Verdikte sind reine Funktionen (Art, Text); tests/test_verdicts.py prüft jede Zahl darin."""

import math
from dataclasses import dataclass

import numpy as np

import sek_algorithm as A
import sek_constants as C
from sek_scenario import draw_batch, draw_one, wait_count

OK_SUCCESS, WARN_SUCCESS = 0.3, 0.1              # Verdikt: grün ab 30 % Erfolg, gelb ab 10 %, sonst Warnung


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    kpct: int = C.DEFAULT_KPCT
    sort: int = C.DEFAULT_SORT
    eps: float = C.DEFAULT_EPS
    ordnung_b: str = C.DEFAULT_ORDNUNG_B
    regel_b: str = C.DEFAULT_REGEL_B
    seed: int = C.DEFAULT_SEED

    @property
    def k(self) -> int:
        return wait_count(self.n, self.kpct)


@dataclass(frozen=True)
class AnalysisA:
    settings: Settings
    k: int
    exact: float                  # exakte Erfolgswahrscheinlichkeit bei zufälliger Reihenfolge
    stats: dict                   # Schätzung für die gewählte Reihenfolge (Erfolg, Wert/Maximum, unter den besten 5)
    example: dict                 # eine gezogene Folge: Werte, gewählter Index, Index des Besten
    curve: tuple                  # (k-Werte, exakt, geschätzt für den gewählten Anteil)


@dataclass(frozen=True)
class AnalysisB:
    settings: Settings
    emax: float
    value: float                  # erwarteter Wert der gewählten Regel in der gewählten Reihenfolge
    ratio: float                  # value / E[max]
    threshold: object             # Schwelle (None bei „immer das erste“ und „optimal“)
    table: list                   # alle Regeln × alle Reihenfolgen: Quotient zu E[max]


def ratio(value: float, emax: float) -> float:
    """Verhältnis zum erwarteten Maximum; ohne Maximum (kann nicht vorkommen) ist 1 der neutrale Wert."""
    return 1.0 if emax <= 0 else value / emax


def analyse_a(s: Settings) -> AnalysisA:
    k = s.k
    V = draw_batch(s.n, s.sort, C.TRIALS, s.seed)
    st = A.stats(V, k)
    one = draw_one(s.n, s.sort, s.seed)
    ks = sorted({int(x) for x in np.linspace(0, s.n - 1, min(s.n, C.CURVE_POINTS))})
    est = [A.success_rate(V, kk) for kk in ks]
    return AnalysisA(s, k, A.p_success(s.n, k), st, {"values": [float(x) for x in one], "chosen": A.pick(list(one), k), "best": int(np.argmax(one))},
                     (ks, [A.p_success(s.n, kk) for kk in ks], est))


def analyse_b(s: Settings) -> AnalysisB:
    emax = A.expected_max(s.eps)
    value = A.rule_value(s.eps, s.ordnung_b, s.regel_b)
    table = [{"Regel": r, **{o: ratio(A.rule_value(s.eps, o, r), emax) for o in C.ORDNUNGEN_B}} for r in C.REGELN_B]
    return AnalysisB(s, emax, value, ratio(value, emax), A.rule_threshold(s.eps, s.regel_b), table)


# ---------------------------------------------------------------------------------------------------------------------------------
# Experimente mit festen Folgen und Seeds


def order_rows(n: int = C.ORDER_N, k: int = C.ORDER_K, trials: int = C.ORDER_TRIALS) -> list[dict]:
    """Erfolg der Regel bei teilweise aufsteigend sortierter Folge, dazu absteigend (die Folge umgekehrt aufsteigend sortiert)."""
    rows = []
    for share in C.ORDER_SHARES:
        V = draw_batch(n, share, trials, 11)
        rows.append({"Reihenfolge": "zufällig" if share == 0 else "aufsteigend" if share == 100 else f"die ersten {share} % aufsteigend", "Anteil": share, **A.stats(V, k)})
    desc = np.sort(draw_batch(n, 0, trials, 12), axis=1)[:, ::-1]
    rows.append({"Reihenfolge": "absteigend", "Anteil": -1, **A.stats(desc, k)})
    return rows


def value_rows(n: int = C.VALUE_N, ks=C.VALUE_KS, trials: int = C.VALUE_TRIALS) -> list[dict]:
    V = draw_batch(n, 0, trials, 13)
    return [{"k": k, **A.stats(V, k)} for k in ks]


def growth_rows(ns=C.GROWTH_NS) -> list[dict]:
    rows = []
    for n in ns:
        k = A.best_k(n)
        rows.append({"n": n, "bestes k": k, "k/n": k / n, "Erfolg": A.p_success(n, k)})
    return rows


def instance_rows(count: int = C.INSTANCES, nvars: int = C.INSTANCE_VARS, samples: int = C.INSTANCE_SAMPLES) -> dict:
    """Zufällige Instanzen unabhängiger, verschiedener Verteilungen (Stichprobe, fester Seed): Schwelle E[max]/2 in drei Reihenfolgen, Quotient zu E[max]."""
    rng = np.random.default_rng(7)
    out = {"zufällig": [], "aufsteigend nach E[X]": [], "absteigend nach E[X]": []}
    for _ in range(count):
        S = np.array([_sample(_random_dist(rng), rng, samples) for _ in range(nvars)])
        emax = float(S.max(axis=0).mean())
        T = emax / 2
        means = S.mean(axis=1)
        out["zufällig"].append(np.mean([_threshold_value(S, list(rng.permutation(nvars)), T) for _ in range(6)]) / emax)
        out["aufsteigend nach E[X]"].append(_threshold_value(S, list(np.argsort(means)), T) / emax)
        out["absteigend nach E[X]"].append(_threshold_value(S, list(np.argsort(-means)), T) / emax)
    return {k: {"Mittel": float(np.mean(v)), "Minimum": float(np.min(v))} for k, v in out.items()}


def _random_dist(rng):
    t = int(rng.integers(3))
    if t == 0:
        return ("exp", float(rng.uniform(0.2, 5)))
    if t == 1:
        return ("zwei", float(rng.uniform(0.02, 0.5)), float(rng.uniform(2, 30)))
    return ("gleich", float(rng.uniform(0.5, 6)))


def _sample(d, rng, m):
    if d[0] == "exp":
        return rng.exponential(d[1], m)
    if d[0] == "zwei":
        return np.where(rng.random(m) < d[1], d[2], 0.0)
    return rng.uniform(0, d[1], m)


def _threshold_value(S, order, T) -> float:
    val = np.zeros(S.shape[1])
    done = np.zeros(S.shape[1], dtype=bool)
    for i in order:
        take = (~done) & (S[i] >= T)
        val[take] = S[i][take]
        done |= take
    last = S[order[-1]]
    val[~done] = last[~done]                    # kommt keines über die Schwelle: das letzte Angebot
    return float(val.mean())


def iid_rows(ns=C.IID_NS) -> list[dict]:
    rows = []
    for n in ns:
        emax = A.iid_uniform_expected_max(n)
        rows.append({
            "n": n, "E[max]": emax, "optimal": ratio(A.iid_uniform_optimal(n), emax),
            "Schwelle E[max]/2": ratio(A.iid_uniform_threshold_value(n, emax / 2), emax),
            "Schwelle Median des Maximums": ratio(A.iid_uniform_threshold_value(n, A.iid_uniform_median_threshold(n)), emax),
        })
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------
# Verdikte (rein: Art, Text)


def delta_text(value: float, reference: float, label: str) -> tuple[str, str]:
    """Text und Farbregel des Deltas: „mein Wert minus Referenz“, in Prozentpunkten. Ein auf eine Stelle gleicher Wert bekommt keinen Pfeil und keine Farbe."""
    d = 100.0 * (value - reference)
    if abs(d) < 0.05:
        return f"gleich wie {label}", "off"
    return f"{d:+.1f} Punkte gegen {label}", "normal"


def verdict_a(a: AnalysisA) -> tuple[str, str]:
    """Kernfrage Teil A: Wie oft findet die Regel den Besten, und was kostet die Reihenfolge?"""
    s = a.settings
    st = a.stats
    text = (f"Bei {s.n} Angeboten und {a.k} nur angesehenen findet die Regel den Besten in **{st['Erfolg'] * 100:.1f} %** der Fälle "
            f"(bei zufälliger Reihenfolge exakt {a.exact * 100:.1f} %, Grenzwert 1/e = {100 / math.e:.1f} %). Das gewählte Angebot ist im Mittel das {st['Wert/Maximum']:.2f}-Fache des besten "
            f"und liegt in {st['unter den besten 5'] * 100:.0f} % der Fälle unter den besten 5.")
    if s.sort > 0:
        text += f" Die ersten {s.sort} % der Folge sind aufsteigend sortiert."
    kind = "success" if st["Erfolg"] >= OK_SUCCESS else "info" if st["Erfolg"] >= WARN_SUCCESS else "warning"
    return kind, text


def verdict_growth(rows: list[dict]) -> tuple[str, str]:
    """Experiment A4: bestes k und Erfolg bei wachsendem n."""
    a, b = rows[0], rows[-1]
    return "info", (f"Bei n = {a['n']} ist das beste k = {a['bestes k']} (Erfolg {a['Erfolg']:.4f}), bei n = {b['n']} ist es k = {b['bestes k']} (Anteil {b['k/n']:.4f}, Erfolg {b['Erfolg']:.4f}); "
                    "beides nähert sich 1/e = 0,3679.")


def verdict_order(rows: list[dict]) -> tuple[str, str]:
    """Experiment A2: Reihenfolge."""
    by = {r["Reihenfolge"]: r for r in rows}
    half = next(r for r in rows if r["Anteil"] == 50)
    return "warning", (f"Bei zufälliger Reihenfolge findet die Regel den Besten in {by['zufällig']['Erfolg'] * 100:.1f} % der Fälle, bei aufsteigender und bei absteigender in "
                       f"{by['aufsteigend']['Erfolg'] * 100:.1f} % und {by['absteigend']['Erfolg'] * 100:.1f} %. Ist nur die erste Hälfte aufsteigend sortiert, sind es **{half['Erfolg'] * 100:.1f} %**.")


def verdict_value(rows: list[dict]) -> tuple[str, str]:
    """Experiment A3: das Ziel bestimmt die Wartezeit."""
    best_success = max(rows, key=lambda r: r["Erfolg"])
    best_value = max(rows, key=lambda r: r["Wert/Maximum"])
    return "info", (f"Wer den Besten will, wartet k = {best_success['k']} (Erfolg {best_success['Erfolg'] * 100:.1f} %, Wert/Maximum {best_success['Wert/Maximum']:.3f}); "
                    f"wer einen hohen Wert will, wartet k = {best_value['k']} (Wert/Maximum {best_value['Wert/Maximum']:.3f}, Erfolg {best_value['Erfolg'] * 100:.1f} %).")


def verdict_b(b: AnalysisB) -> tuple[str, str]:
    """Kernfrage Teil B: Wie viel vom erwarteten Maximum erreicht die Regel in dieser Reihenfolge?"""
    s = b.settings
    schwelle = f" Die Schwelle ist {b.threshold:.3f}." if b.threshold is not None else ""
    text = (f"**{s.regel_b}** erreicht bei „{s.ordnung_b}“ im Mittel **{b.value:.3f}**, das ist das **{b.ratio:.3f}-Fache** des erwarteten Maximums {b.emax:.3f} "
            f"(Garantie der Schwellenregel: 0,5 in jeder Reihenfolge).{schwelle}")
    kind = "success" if b.ratio >= 0.9 else "info" if b.ratio >= 0.6 else "warning"
    return kind, text


def verdict_instances(res: dict) -> tuple[str, str]:
    """Experiment B2: zufällige Instanzen."""
    worst = min(v["Minimum"] for v in res.values())
    return "info", (f"Schwelle E[max]/2 bei zufälliger Reihenfolge im Mittel {res['zufällig']['Mittel']:.3f} des erwarteten Maximums (Minimum {res['zufällig']['Minimum']:.3f}), "
                    f"aufsteigend nach E[X] {res['aufsteigend nach E[X]']['Mittel']:.3f}, absteigend {res['absteigend nach E[X]']['Mittel']:.3f}; der kleinste Wert aller Instanzen und Reihenfolgen ist {worst:.3f}, die Garantie 0,5.")


def verdict_iid(rows: list[dict]) -> tuple[str, str]:
    """Experiment B3: unabhängig und gleich verteilt (0, 1]."""
    last = rows[-1]
    gap = (last["optimal"] - last["Schwelle E[max]/2"]) * 100
    return "info", (f"Bei n = {last['n']} gleichverteilten Angeboten erreicht die optimale Rückwärtsrechnung {last['optimal']:.4f} des erwarteten Maximums, die Schwelle E[max]/2 {last['Schwelle E[max]/2']:.4f}, "
                    f"die Schwelle am Median des Maximums {last['Schwelle Median des Maximums']:.4f}: {gap:.1f} Punkte Unterschied.")
