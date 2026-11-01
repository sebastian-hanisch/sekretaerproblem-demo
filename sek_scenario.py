"""Folgen von Angeboten für Teil A: Werte gleichverteilt in (0, 1), die ersten m (Anteil in Prozent) aufsteigend sortiert."""

import numpy as np

import sek_constants as C


def wait_count(n: int, kpct: int) -> int:
    """Zahl der nur angesehenen Angebote: Anteil kpct von n, gerundet, höchstens n − 1."""
    return min(n - 1, int(round(n * kpct / 100.0)))


def sorted_prefix_length(n: int, sort_pct: int) -> int:
    return int(round(n * sort_pct / 100.0))


def draw_batch(n: int, sort_pct: int, trials: int, seed: int) -> np.ndarray:
    """trials Folgen der Länge n: zufällige Reihenfolge verschiedener Werte, danach die ersten m aufsteigend sortiert."""
    rng = np.random.default_rng([seed, n, sort_pct])
    V = rng.random((trials, n))
    m = sorted_prefix_length(n, sort_pct)
    if m > 1:
        V[:, :m] = np.sort(V[:, :m], axis=1)
    return V


def draw_one(n: int, sort_pct: int, seed: int) -> np.ndarray:
    return draw_batch(n, sort_pct, 1, seed + 1)[0]
