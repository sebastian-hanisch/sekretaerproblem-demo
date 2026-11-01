"""Claims: jede Zahl der README, über dieselben Auswertungsfunktionen wie in der App nachgerechnet.

Gezogene Folgen laufen mit festem Seed und sind auf jedem Rechner dieselben; die Zufallsinstanzen in Experiment 4 sind versionsabhängig
(Zufallsgenerator-Verteilungen), dort gilt eine Toleranz von 0,01."""

import math

import pytest

import sek_algorithm as A
import sek_constants as C
import sek_evaluation as ev


def pct(x: float) -> float:
    return round(100 * x, 1)


@pytest.fixture(scope="module")
def order():
    return {r["Anteil"]: r for r in ev.order_rows()}


@pytest.fixture(scope="module")
def value():
    return {r["k"]: r for r in ev.value_rows()}


def _preset(name):
    return ev.analyse_a(ev.Settings(**C.PRESETS[name]))


def test_klassisch():
    a = _preset("Klassisch (n = 100, 37 %)")
    assert pct(a.exact) == 37.1 and pct(a.stats["Erfolg"]) == 37.1
    assert round(a.stats["Wert/Maximum"], 2) == 0.81 and round(100 * a.stats["unter den besten 5"]) == 63


def test_bestes_k_bei_wachsendem_n():
    rows = {r["n"]: r for r in ev.growth_rows()}
    assert [rows[n]["bestes k"] for n in (3, 10, 100, 1000, 10000)] == [1, 3, 37, 368, 3679]
    assert [pct(rows[n]["Erfolg"]) for n in (3, 10, 100, 1000, 10000)] == [50.0, 39.9, 37.1, 36.8, 36.8]
    assert round(1 / math.e * 100, 2) == 36.79 and round(rows[10000]["k/n"], 4) == 0.3679


def test_aufsteigend_sortiert():
    a = _preset("Aufsteigend sortiert")
    assert a.stats["Erfolg"] == 0.0 and round(a.stats["Wert/Maximum"], 3) == 0.379
    assert a.example["chosen"] == a.k == 37 and a.example["chosen"] + 1 == 38           # die Regel nimmt das 38. Angebot (Index 37)


def test_nur_der_anfang_sortiert(order):
    assert pct(order[25]["Erfolg"]) == 36.8
    assert all(order[x]["Erfolg"] == 0.0 for x in (50, 75, 100, -1))
    assert round(order[50]["Wert/Maximum"], 3) == 0.752 and round(order[75]["Wert/Maximum"], 3) == 0.505
    assert round(order[100]["Wert/Maximum"], 3) == 0.380 and round(order[-1]["Wert/Maximum"], 3) == 0.010
    assert pct(order[0]["Erfolg"]) == 37.6
    assert abs(order[0]["Erfolg"] - A.p_success(100, 37)) < 0.01                       # Stichprobe nahe dem exakten Wert


def test_frueh_zuschlagen():
    a = _preset("Früh zuschlagen (k = 10 %)")
    assert a.k == 10 and pct(a.exact) == 23.5 and pct(a.stats["Erfolg"]) == 23.1
    assert round(a.stats["Wert/Maximum"], 2) == 0.92


def test_kleine_folge():
    a = _preset("Kleine Folge (n = 10)")
    assert a.k == 3 and pct(a.exact) == 39.9


def test_ziel_bestimmt_die_wartezeit(value):
    ks = (0, 5, 10, 20, 37, 50, 70)
    assert [pct(value[k]["Erfolg"]) for k in ks] == [1.0, 15.2, 23.3, 32.6, 37.2, 35.0, 25.2]
    assert [round(value[k]["Wert/Maximum"], 3) for k in ks] == [0.504, 0.899, 0.913, 0.885, 0.810, 0.748, 0.650]
    assert [pct(value[k]["unter den besten 5"]) for k in ks] == [4.8, 47.0, 62.1, 70.5, 63.1, 51.9, 33.0]
    assert max(value.values(), key=lambda r: r["Erfolg"])["k"] == 37
    assert max(value.values(), key=lambda r: r["Wert/Maximum"])["k"] == 10
    assert max(value.values(), key=lambda r: r["unter den besten 5"])["k"] == 20


def test_scharfes_beispiel():
    b = ev.analyse_b(ev.Settings(eps=0.01))
    t = {r["Regel"]: r for r in b.table}
    assert b.emax == pytest.approx(1.99)
    s = t["Schwelle E[max]/2"]
    assert [round(s[o], 4) for o in ("kleine zuerst", "große zuerst", "zufällig")] == [0.5025, 1.0, 0.7513]
    assert [round(t["immer das erste"][o], 4) for o in C.ORDNUNGEN_B] == [0.5025] * 3
    assert s["kleine zuerst"] == pytest.approx(1 / (2 - 0.01))
    half = {r["Regel"]: r for r in ev.analyse_b(ev.Settings(eps=0.5)).table}["Schwelle E[max]/2"]
    assert [round(half[o], 3) for o in ("kleine zuerst", "große zuerst", "zufällig")] == [0.667, 1.0, 0.833]


def test_optimal_und_beide_schwellen_gleichen_sich_im_scharfen_beispiel():
    t = {r["Regel"]: r for r in ev.analyse_b(ev.Settings(eps=0.01)).table}
    for o in C.ORDNUNGEN_B:
        assert round(t["optimal (Rückwärtsrechnung)"][o], 4) == round(t["Schwelle E[max]/2"][o], 4) == round(t["Schwelle Median des Maximums"][o], 4)


def test_zufaellige_instanzen():
    res = ev.instance_rows()
    for name, mean, minimum in (("zufällig", 0.854, 0.756), ("aufsteigend nach E[X]", 0.800, 0.680), ("absteigend nach E[X]", 0.866, 0.759)):
        assert res[name]["Mittel"] == pytest.approx(mean, abs=0.01), name
        assert res[name]["Minimum"] == pytest.approx(minimum, abs=0.01), name
    assert min(v["Minimum"] for v in res.values()) > 0.5


def test_unabhaengig_gleichverteilt():
    rows = {r["n"]: r for r in ev.iid_rows()}
    assert [round(rows[n]["optimal"], 4) for n in (2, 10, 100)] == [0.9375, 0.9472, 0.9910]
    assert [round(rows[n]["Schwelle E[max]/2"], 4) for n in (2, 10, 100)] == [0.9167, 0.7998, 0.7550]
    assert [round(rows[n]["Schwelle Median des Maximums"], 4) for n in (2, 10, 100)] == [0.9053, 0.7882, 0.7540]
    assert round(100 * (rows[100]["optimal"] - rows[100]["Schwelle E[max]/2"]), 1) == 23.6


def test_verdikte_der_standardeinstellung_nennen_die_readme_zahlen():
    kind, text = ev.verdict_a(_preset("Klassisch (n = 100, 37 %)"))
    assert kind == "success" and "**37.1 %**" in text and "0.81-Fache" in text and "63 %" in text
