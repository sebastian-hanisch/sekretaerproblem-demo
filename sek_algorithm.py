"""Kern: Sekretärproblem (nur die Rangfolge der bisherigen Angebote bekannt) und Prophet-Ungleichung (Verteilungen bekannt).

Sekretärregel: die ersten k Angebote nur ansehen, danach das erste nehmen, das besser ist als alle bisher gesehenen; kommt keines, das letzte.
Größer ist besser."""

import numpy as np

import sek_constants as C


# ---------------------------------------------------------------------------------------------------------------------------------
# Teil A


def p_success(n: int, k: int) -> float:
    """Exakte Wahrscheinlichkeit, den Besten zu wählen, bei zufälliger Reihenfolge: (k/n) · Σ_{i=k}^{n−1} 1/i für k ≥ 1, 1/n für k = 0."""
    if k == 0:
        return 1.0 / n
    return k / n * sum(1.0 / i for i in range(k, n))


def best_k(n: int) -> int:
    """Wartezeit mit der größten Erfolgswahrscheinlichkeit (bei Gleichstand die kleinste)."""
    ps = [p_success(n, k) for k in range(n)]
    return int(np.argmax(ps))


def pick(values, k: int) -> int:
    """Index des gewählten Angebots."""
    n = len(values)
    best_seen = max(values[:k]) if k > 0 else -np.inf
    for i in range(k, n):
        if values[i] > best_seen:
            return i
    return n - 1


def pick_batch(V: np.ndarray, k: int) -> np.ndarray:
    """Index des gewählten Angebots je Zeile (vektorisiert, dieselbe Regel wie pick)."""
    t, n = V.shape
    if k == 0:
        return np.zeros(t, dtype=int)
    seen = V[:, :k].max(axis=1)
    better = V[:, k:] > seen[:, None]
    has = better.any(axis=1)
    return np.where(has, better.argmax(axis=1) + k, n - 1)


def success_rate(V: np.ndarray, k: int) -> float:
    """Anteil der Zeilen, in denen die Regel den Besten wählt (ohne die Sortierung, die stats für „unter den besten 5“ braucht)."""
    idx = pick_batch(V, k)
    return float(np.mean(V[np.arange(V.shape[0]), idx] == V.max(axis=1)))


def stats(V: np.ndarray, k: int) -> dict:
    """Erfolg (der Beste), Wert/Maximum und Anteil unter den besten TOP, je als Mittel über die Zeilen von V."""
    idx = pick_batch(V, k)
    rows = np.arange(V.shape[0])
    chosen = V[rows, idx]
    best = V.max(axis=1)
    top = min(C.TOP, V.shape[1])
    threshold = np.partition(V, V.shape[1] - top, axis=1)[:, V.shape[1] - top]
    return {"Erfolg": float(np.mean(chosen == best)), "Wert/Maximum": float(np.mean(chosen / best)), "unter den besten 5": float(np.mean(chosen >= threshold))}


# ---------------------------------------------------------------------------------------------------------------------------------
# Teil B: das scharfe Beispiel X1 = 1 sicher, X2 = 1/eps mit Wahrscheinlichkeit eps, sonst 0


def expected_max(eps: float) -> float:
    return (1 - eps) * 1.0 + eps * (1.0 / eps)


def median_max_threshold(eps: float) -> float:
    """Kleinster Wert t mit P(max <= t) >= 1/2: max = 1/eps mit Wahrscheinlichkeit eps, sonst 1; für eps <= 1/2 ist es 1."""
    return 1.0 if (1 - eps) >= 0.5 else 1.0 / eps


def rule_value(eps: float, order: str, rule: str) -> float:
    """Erwarteter Wert der Regel im scharfen Beispiel für 'kleine zuerst', 'große zuerst' (exakt) oder 'zufällig' (Mittel beider)."""
    if order == "zufällig":
        return 0.5 * (rule_value(eps, "kleine zuerst", rule) + rule_value(eps, "große zuerst", rule))
    big = 1.0 / eps
    first_small = order == "kleine zuerst"
    if rule == "optimal (Rückwärtsrechnung)":
        if first_small:
            return 1.0                                  # nach der 1 bleibt im Mittel E[X2] = 1: Gleichstand, der Wert ist 1
        return eps * big + (1 - eps) * 1.0              # X2 zuerst: nimm es, wenn es mindestens E[X1] = 1 ist (also bei 1/eps), sonst die 1
    if rule == "immer das erste":
        return 1.0                                      # das erste Angebot: 1 (kleine zuerst) beziehungsweise E[X2] = 1 (große zuerst)
    T = expected_max(eps) / 2.0 if rule == "Schwelle E[max]/2" else median_max_threshold(eps)
    if first_small:
        return 1.0                                      # die sichere 1 erreicht die Schwelle (T <= 1); sonst bliebe das letzte Angebot mit E[X2] = 1: in beiden Fällen 1
    # große zuerst: X2 = big (Wahrscheinlichkeit eps) wird genommen, wenn big >= T; sonst X2 = 0 < T, dann die letzte, die sichere 1
    take_big = big >= T
    return eps * (big if take_big else 1.0) + (1 - eps) * 1.0


def rule_threshold(eps: float, rule: str):
    if rule == "Schwelle E[max]/2":
        return expected_max(eps) / 2.0
    if rule == "Schwelle Median des Maximums":
        return median_max_threshold(eps)
    return None


# Unabhängig und gleich verteilt (0, 1]


def iid_uniform_optimal(n: int) -> float:
    """Wert der optimalen Rückwärtsrechnung bei n unabhängigen gleichverteilten Angeboten: v_1 = 1/2, v_{k+1} = (1 + v_k²)/2."""
    v = 0.5
    for _ in range(n - 1):
        v = (1 + v * v) / 2
    return v


def iid_uniform_expected_max(n: int) -> float:
    return n / (n + 1.0)


def iid_uniform_threshold_value(n: int, T: float) -> float:
    """Erwarteter Wert der Regel „das erste Angebot über T, sonst das letzte“ bei n gleichverteilten Angeboten."""
    p = 1.0 - T                                        # Wahrscheinlichkeit, dass ein Angebot über T liegt
    return (1 - (1 - p) ** n) * (1 + T) / 2 + (1 - p) ** n * (T / 2)


def iid_uniform_median_threshold(n: int) -> float:
    """Median des Maximums: P(max <= t) = t^n = 1/2."""
    return 0.5 ** (1.0 / n)
