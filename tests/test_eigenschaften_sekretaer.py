"""Eigenschaften, die für jede Eingabe gelten müssen (nicht nur für die Beispielzahlen): Gestalt der Erfolgskurve, Rangtreue, Online-Eigenschaft,
Verhalten bei sortierten Folgen, Garantie der Schwelle, Ordnung der Regeln, Wachstum bei unabhängig gleich verteilten Angeboten."""

import math

import numpy as np
import pytest

import sek_algorithm as A
import sek_constants as C
import sek_evaluation as E
from sek_scenario import draw_batch


@pytest.mark.parametrize("n", [5, 10, 37, 100, 500])
def test_erfolgskurve_steigt_bis_zum_besten_k_und_faellt_danach(n):
    ps = [A.p_success(n, k) for k in range(n)]
    kb = A.best_k(n)
    assert all(ps[i] <= ps[i + 1] + 1e-12 for i in range(1, kb))
    assert all(ps[i] >= ps[i + 1] - 1e-12 for i in range(kb, n - 1))
    assert ps[kb] == max(ps)


def test_bestes_k_naehert_sich_n_durch_e_und_der_erfolg_faellt_von_oben_gegen_1_durch_e():
    rows = E.growth_rows()
    assert all(r["Erfolg"] > 1 / math.e for r in rows)                                    # nie darunter
    assert all(a["Erfolg"] > b["Erfolg"] for a, b in zip(rows, rows[1:]))                 # monoton fallend
    assert abs(rows[-1]["k/n"] - 1 / math.e) < 1e-3
    assert abs(rows[-1]["Erfolg"] - 1 / math.e) < 1e-4


def test_jede_wartezeit_liefert_mindestens_1_durch_n():
    for n in (5, 20, 100):
        assert min(A.p_success(n, k) for k in range(n)) >= 1.0 / n - 1e-12


@pytest.mark.parametrize("transform", [lambda v: v ** 3, lambda v: np.exp(5 * v), lambda v: 10 * v + 7])
def test_wahl_haengt_nur_von_der_rangfolge_ab(transform):
    V = draw_batch(30, 0, 400, 21)
    for k in (0, 5, 11, 29):
        assert np.array_equal(A.pick_batch(V, k), A.pick_batch(transform(V), k))


def test_online_eigenschaft_spaetere_angebote_aendern_die_wahl_nicht():
    rng = np.random.default_rng(3)
    V = draw_batch(25, 0, 300, 22)
    for k in (0, 4, 9, 24):
        idx = A.pick_batch(V, k)
        W = V.copy()
        for r in range(len(V)):
            W[r, idx[r] + 1:] = rng.random(25 - idx[r] - 1)                                  # alles nach der Wahl neu würfeln
        assert np.array_equal(idx, A.pick_batch(W, k))


def test_gewaehlt_wird_ein_rekord_oder_das_letzte():
    V = draw_batch(20, 0, 500, 23)
    for k in (0, 3, 7):
        for row, i in zip(V, A.pick_batch(V, k)):
            assert i == len(row) - 1 or row[i] > row[:i].max(initial=-1.0)


@pytest.mark.parametrize("n", [10, 20])
def test_aufsteigende_folge_verfehlt_den_besten_ausser_bei_voller_wartezeit(n):
    V = draw_batch(n, 100, 200, 24)
    assert A.success_rate(V, n - 1) == 1.0                                                     # alles angesehen bis auf das letzte: das ist der Beste
    for k in range(n - 1):
        assert A.success_rate(V, k) == 0.0                                                     # das Angebot an Index k ist nie der Beste


@pytest.mark.parametrize("n", [10, 20])
def test_absteigende_folge_findet_den_besten_nur_ohne_wartezeit(n):
    V = np.sort(draw_batch(n, 0, 200, 25), axis=1)[:, ::-1]
    assert A.success_rate(V, 0) == 1.0
    for k in range(1, n):
        assert A.success_rate(V, k) == 0.0


def test_je_laenger_sortiert_desto_schlechter_bei_fester_wartezeit():
    rows = E.order_rows()
    by = {r["Anteil"]: r["Erfolg"] for r in rows}
    assert by[0] > 0.3 and by[25] > 0.3                                                         # die Sortierung im angesehenen Teil schadet nicht
    assert by[50] == by[75] == by[100] == by[-1] == 0.0                                         # sobald die Sortierung über die Wartezeit hinausreicht, bricht alles ein


@pytest.mark.parametrize("eps", C.EPS_OPTIONS)
def test_schwelle_garantiert_die_haelfte_in_jeder_reihenfolge(eps):
    b = E.analyse_b(E.Settings(eps=eps))
    for row in b.table:
        if row["Regel"] in ("Schwelle E[max]/2", "Schwelle Median des Maximums"):
            assert all(row[o] >= 0.5 - 1e-12 for o in C.ORDNUNGEN_B), (eps, row)


def test_scharf_der_quotient_der_schwelle_ist_1_durch_2_minus_eps():
    for eps in C.EPS_OPTIONS:
        b = E.analyse_b(E.Settings(eps=eps))
        row = next(r for r in b.table if r["Regel"] == "Schwelle E[max]/2")
        assert row["kleine zuerst"] == pytest.approx(1 / (2 - eps), abs=1e-12)
    assert E.analyse_b(E.Settings(eps=0.01)).table[0]["kleine zuerst"] < 0.5026


@pytest.mark.parametrize("eps", C.EPS_OPTIONS)
def test_optimal_ist_nie_schlechter_und_reihenfolge_ordnet_die_schwelle(eps):
    t = {r["Regel"]: r for r in E.analyse_b(E.Settings(eps=eps)).table}
    for o in C.ORDNUNGEN_B:
        assert all(t["optimal (Rückwärtsrechnung)"][o] >= r[o] - 1e-12 for r in t.values())
    s = t["Schwelle E[max]/2"]
    assert s["große zuerst"] >= s["zufällig"] >= s["kleine zuerst"]
    assert t["Schwelle E[max]/2"] == t["Schwelle Median des Maximums"] | {"Regel": "Schwelle E[max]/2"}   # in diesem Beispiel stimmen beide Schwellen überein


def test_unabhaengig_gleichverteilt_optimal_schlaegt_schwellen_und_waechst():
    rows = E.iid_rows()
    assert all(r["optimal"] >= r["Schwelle E[max]/2"] and r["optimal"] >= r["Schwelle Median des Maximums"] for r in rows)
    assert all(r["optimal"] <= 1.0 for r in rows)
    assert all(r["Schwelle E[max]/2"] >= 0.5 for r in rows)
    v = [A.iid_uniform_optimal(n) for n in range(1, 30)]
    assert all(a < b for a, b in zip(v, v[1:]))                                                 # mehr Angebote, mehr Wert
    assert A.iid_uniform_optimal(1) == A.iid_uniform_expected_max(1)                            # ein Angebot: kein Unterschied zum Propheten
    assert all(A.iid_uniform_optimal(n) < A.iid_uniform_expected_max(n) for n in range(2, 30))  # sonst echt weniger als der Prophet


def test_zufaellige_instanzen_erfuellen_die_garantie_in_jeder_reihenfolge():
    res = E.instance_rows()
    assert min(v["Minimum"] for v in res.values()) > 0.5
    assert res["aufsteigend nach E[X]"]["Mittel"] < res["absteigend nach E[X]"]["Mittel"]       # kleine zuerst schadet


def test_schwellenregel_vektorisiert_gleicht_einer_schleife_von_hand():
    S = np.array([[1.0, 5.0], [4.0, 0.0], [2.0, 2.0]])           # drei Angebote, zwei Stichproben
    # Reihenfolge 0, 1, 2 und Schwelle 3: Stichprobe 0 → Angebot 1 (Wert 4 ist in Reihe 1, aber Reihe 0 hat nur 1; zuerst ≥ 3 ist Reihe 1 mit 4);
    # Stichprobe 1 → Reihe 0 hat 5 ≥ 3 und wird gewählt.
    wert = E._threshold_value(S, [0, 1, 2], 3.0)
    assert wert == pytest.approx((4.0 + 5.0) / 2)
    # Schwelle 10: nichts erreicht sie, das letzte Angebot (Reihe 2): 2 und 2
    assert E._threshold_value(S, [0, 1, 2], 10.0) == pytest.approx(2.0)
