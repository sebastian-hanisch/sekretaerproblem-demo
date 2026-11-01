"""Kern von Hand gerechnet: Erfolgsformel, Wahlregel (einzeln und vektorisiert), Kennzahlen, Folgen, scharfes Beispiel (Prophet)."""

import numpy as np
import pytest

import sek_algorithm as A
import sek_constants as C
from sek_scenario import draw_batch, draw_one, sorted_prefix_length, wait_count


def test_erfolgsformel_von_hand():
    assert A.p_success(3, 1) == pytest.approx(0.5)                 # von sechs Reihenfolgen gewinnen drei (siehe Aufzählung im Orakeltest)
    assert A.p_success(4, 1) == pytest.approx(11 / 24)             # (1/4)(1 + 1/2 + 1/3)
    assert A.p_success(4, 2) == pytest.approx(10 / 24)             # (2/4)(1/2 + 1/3)
    assert A.p_success(5, 0) == pytest.approx(0.2)                 # ohne Wartezeit: das erste Angebot, Erfolg 1/n
    assert A.p_success(1, 0) == 1.0


def test_bestes_k_von_hand():
    assert A.best_k(3) == 1
    assert A.best_k(5) == 2
    assert A.best_k(10) == 3
    assert A.best_k(100) == 37


def test_wahl_von_hand():
    v = [3, 1, 4, 2, 5]
    assert A.pick(v, 0) == 0                    # ohne Wartezeit: das erste
    assert A.pick(v, 2) == 2                    # das beste Gesehene ist 3, die 4 an Index 2 ist die erste bessere
    assert A.pick(v, 4) == 4                    # das beste Gesehene ist 4, die 5 am Ende ist besser
    assert A.pick([5, 1, 2], 1) == 2            # nichts übertrifft die 5: das letzte Angebot
    assert A.pick([1, 2, 3, 4], 3) == 3         # die Wartezeit n − 1 lässt nur das letzte übrig


def test_vektorisierte_wahl_gleicht_der_einzelnen():
    V = draw_batch(12, 0, 300, 3)
    for k in (0, 1, 4, 11):
        idx = A.pick_batch(V, k)
        assert [int(i) for i in idx] == [A.pick(list(row), k) for row in V]


def test_kennzahlen_von_hand():
    V = np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [3.0, 1.0, 2.0]])
    # k = 1: Zeile 1 wählt 3 (der Beste), Zeile 2 wählt 3 (der Beste), Zeile 3 sieht 3 und nimmt das letzte, 2
    st = A.stats(V, 1)
    assert st["Erfolg"] == pytest.approx(2 / 3)
    assert st["Wert/Maximum"] == pytest.approx((1 + 1 + 2 / 3) / 3)
    assert st["unter den besten 5"] == 1.0                                # nur drei Angebote: alle gehören zu den „besten 5“
    assert A.success_rate(V, 1) == pytest.approx(st["Erfolg"])


def test_unter_den_besten_zaehlt_den_fuenften_platz_mit():
    V = np.arange(1.0, 11.0)[None, :]                                     # aufsteigend 1 … 10
    assert A.stats(V, 4)["unter den besten 5"] == 0.0                     # wählt die 5, der sechstbeste Wert
    assert A.stats(V, 5)["unter den besten 5"] == 1.0                     # wählt die 6, der fünftbeste Wert


def test_wartezeit_aus_prozent():
    assert wait_count(100, 37) == 37
    assert wait_count(10, 30) == 3
    assert wait_count(10, 95) == 9                                        # höchstens n − 1
    assert wait_count(1000, 0) == 0
    assert wait_count(10, 100) == 9


def test_folgen_sind_reproduzierbar_verschieden_und_wie_verlangt_sortiert():
    V = draw_batch(20, 50, 50, 5)
    assert V.shape == (50, 20)
    assert np.array_equal(V, draw_batch(20, 50, 50, 5))
    assert not np.array_equal(V, draw_batch(20, 50, 50, 6))
    m = sorted_prefix_length(20, 50)
    assert m == 10
    assert np.all(np.diff(V[:, :m], axis=1) > 0)                          # der Anfang ist streng aufsteigend
    assert not np.all(np.diff(V[:, m:], axis=1) > 0)                      # der Rest nicht
    assert len(set(V[0])) == 20                                           # die Werte sind verschieden
    assert draw_one(20, 0, 3).shape == (20,)


def test_scharfes_beispiel_von_hand():
    eps = 0.1                                                             # X1 = 1, X2 = 10 mit Wahrscheinlichkeit 0,1
    assert A.expected_max(eps) == pytest.approx(1.9)
    # kleine zuerst: die sichere 1 erreicht die Schwelle 0,95, der Wert ist 1
    assert A.rule_value(eps, "kleine zuerst", "Schwelle E[max]/2") == pytest.approx(1.0)
    # große zuerst: 0,1 · 10 + 0,9 · 1 = 1,9, das ganze Maximum
    assert A.rule_value(eps, "große zuerst", "Schwelle E[max]/2") == pytest.approx(1.9)
    assert A.rule_value(eps, "zufällig", "Schwelle E[max]/2") == pytest.approx(1.45)
    assert A.rule_threshold(eps, "Schwelle E[max]/2") == pytest.approx(0.95)
    assert A.rule_threshold(eps, "immer das erste") is None


def test_gleichverteilt_von_hand():
    assert A.iid_uniform_optimal(1) == pytest.approx(0.5)
    assert A.iid_uniform_optimal(2) == pytest.approx(5 / 8)               # (1 + 1/4) / 2
    assert A.iid_uniform_optimal(3) == pytest.approx(89 / 128)            # (1 + 25/64) / 2
    assert A.iid_uniform_expected_max(2) == pytest.approx(2 / 3)
    assert A.iid_uniform_median_threshold(2) == pytest.approx(0.5 ** 0.5)
    assert A.iid_uniform_threshold_value(3, 0.0) == pytest.approx(0.5)    # Schwelle 0: das erste Angebot, Mittel 1/2
    assert A.iid_uniform_threshold_value(1, 0.5) == pytest.approx(0.5)    # ein Angebot: es ist das letzte, Mittel 1/2
    # zwei Angebote, Schwelle 0,5: mindestens eines darüber (Wahrscheinlichkeit 3/4, Mittel 0,75), sonst das letzte unter 0,5 (Wahrscheinlichkeit 1/4, Mittel 0,25)
    assert A.iid_uniform_threshold_value(2, 0.5) == pytest.approx(0.75 * 0.75 + 0.25 * 0.25)


def test_konstanten_sind_in_sich_stimmig():
    assert C.TOP == 5
    assert set(C.REGELN_B) == set(C.REGEL_B_HILFE)
    assert set(C.ORDNUNGEN_B) == set(C.ORDNUNG_B_HILFE)
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert C.DEFAULT_KPCT == 37
