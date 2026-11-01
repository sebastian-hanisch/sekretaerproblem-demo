"""Orakel: Aufzählung aller Reihenfolgen gegen die Formel; zweite, getrennt geschriebene Fassungen der Regeln; Stichproben gegen Formeln.

Jede Prüfung rechnet auf einem anderen Weg als der Kern: Aufzählung aller Permutationen (Teil A), Aufzählung der Ausgänge des
riskanten Angebots mit eigener Regelschleife und eigener Rückwärtsrechnung (Teil B), Monte-Carlo mit schleifenweiser Regel (gleichverteilt)."""

import itertools
import math

import numpy as np
import pytest

import sek_algorithm as A
import sek_constants as C
from sek_scenario import draw_batch


def _gewinnt(perm, k) -> bool:
    """Schleifenweise Fassung der Sekretärregel auf einer Rangfolge (größer ist besser)."""
    n = len(perm)
    bester = max(perm[:k]) if k > 0 else -1
    for i in range(k, n):
        if perm[i] > bester:
            return perm[i] == n - 1                                           # gewählt; Erfolg, wenn es der Beste ist
    return perm[n - 1] == n - 1                                               # nichts war besser: das letzte


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_formel_gleicht_der_aufzaehlung_aller_reihenfolgen(n):
    perms = list(itertools.permutations(range(n)))
    for k in range(n):
        anteil = sum(_gewinnt(p, k) for p in perms) / len(perms)
        assert A.p_success(n, k) == pytest.approx(anteil, abs=1e-12), (n, k)


def test_n3_k1_drei_von_sechs():
    gewinner = [p for p in itertools.permutations(range(3)) if _gewinnt(p, 1)]
    assert len(gewinner) == 3
    assert sorted(gewinner) == [(0, 2, 1), (1, 0, 2), (1, 2, 0)]


@pytest.mark.parametrize("n,k", [(10, 3), (50, 18), (100, 37), (100, 10)])
def test_stichprobe_streut_um_die_formel(n, k):
    trials = 40000
    V = draw_batch(n, 0, trials, 101)
    p = A.p_success(n, k)
    sd = math.sqrt(p * (1 - p) / trials)
    assert abs(A.success_rate(V, k) - p) < 4.5 * sd


def _simuliere(eps, ordnung, regel):
    """Regel im scharfen Beispiel durch Aufzählung der beiden Ausgänge des riskanten Angebots, mit eigener Schleife und Rückwärtsrechnung."""
    ausgaenge = [(1.0 / eps, eps), (0.0, 1 - eps)]
    emax = sum(w * max(1.0, x) for x, w in ausgaenge)
    schwelle = {"Schwelle E[max]/2": emax / 2, "Schwelle Median des Maximums": None}.get(regel)
    if regel == "Schwelle Median des Maximums":
        verteilung = sorted([(max(1.0, x), w) for x, w in ausgaenge])
        summe = 0.0
        for wert, w in verteilung:
            summe += w
            if summe >= 0.5:
                schwelle = wert
                break
    gesamt = 0.0
    for x, w in ausgaenge:
        folge = [1.0, x] if ordnung == "kleine zuerst" else [x, 1.0]            # in dieser Reihenfolge kommen die Angebote
        if regel == "immer das erste":
            wert = folge[0]
        elif regel == "optimal (Rückwärtsrechnung)":
            # E des zweiten Angebots, solange es noch nicht gesehen ist: 1 (sicher) beziehungsweise eps · 1/eps = 1 (riskant)
            wert = folge[0] if folge[0] >= 1.0 else 1.0                          # nimm das erste, wenn es mindestens den Erwartungswert 1 des zweiten hat
        else:
            wert = folge[0] if folge[0] >= schwelle else folge[1]
        gesamt += w * wert
    return gesamt


@pytest.mark.parametrize("eps", C.EPS_OPTIONS)
@pytest.mark.parametrize("ordnung", ["kleine zuerst", "große zuerst"])
@pytest.mark.parametrize("regel", C.REGELN_B)
def test_scharfes_beispiel_gleicht_der_aufzaehlung(eps, ordnung, regel):
    if regel == "optimal (Rückwärtsrechnung)" and ordnung == "große zuerst":
        # Optimal hier: das riskante Angebot nehmen, wenn es mindestens 1 ist (Ausgang 1/eps), sonst die sichere 1 (Erwartungswert, bleibt 1)
        erwartet = eps * (1.0 / eps) + (1 - eps) * 1.0
    else:
        erwartet = _simuliere(eps, ordnung, regel)
    assert A.rule_value(eps, ordnung, regel) == pytest.approx(erwartet, abs=1e-12)


def _wert_optimal_gleichverteilt(n: int, rng, trials: int) -> float:
    """Schleifenweise Simulation der optimalen Regel (Schwelle v_m bei m ausstehenden Angeboten, Rückwärtsrechnung unabhängig neu geschrieben)."""
    v = [0.0, 0.5]
    for _ in range(n):
        v.append((1 + v[-1] ** 2) / 2)
    summe = 0.0
    for _ in range(trials):
        x = rng.random(n)
        for i in range(n):
            ausstehend = n - i - 1
            if ausstehend == 0 or x[i] >= v[ausstehend]:
                summe += x[i]
                break
    return summe / trials


def test_gleichverteilt_optimal_gleicht_der_simulation():
    rng = np.random.default_rng(5)
    trials = 20000
    for n in (2, 5):
        sim = _wert_optimal_gleichverteilt(n, rng, trials)
        assert abs(sim - A.iid_uniform_optimal(n)) < 4.5 * 0.29 / math.sqrt(trials), n


@pytest.mark.parametrize("n,T", [(3, 0.5), (10, 0.3), (25, 0.8)])
def test_gleichverteilt_schwelle_gleicht_der_simulation(n, T):
    rng = np.random.default_rng(6)
    X = rng.random((200000, n))
    ueber = X >= T
    erste = ueber.argmax(axis=1)
    wert = np.where(ueber.any(axis=1), X[np.arange(len(X)), erste], X[:, -1])
    assert abs(wert.mean() - A.iid_uniform_threshold_value(n, T)) < 4.5 * wert.std() / math.sqrt(len(X))
