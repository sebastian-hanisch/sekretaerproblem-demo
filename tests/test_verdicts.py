"""Verdikte: jede Zahl und jede Verzweigung der Texte. Die Verdikte sind reine Funktionen (Art, Text); die Zahlen im Text werden aus den Eingaben nachgerechnet."""

import pytest

import sek_evaluation as ev


def _a(erfolg, sort=0, wert=0.8123, top=0.627, exact=0.3710, n=100, k=37):
    s = ev.Settings(n=n, kpct=k, sort=sort)
    return ev.AnalysisA(s, k, exact, {"Erfolg": erfolg, "Wert/Maximum": wert, "unter den besten 5": top}, {}, ())


def test_verdict_a_zahlen_und_art():
    kind, text = ev.verdict_a(_a(0.3708))
    assert kind == "success"
    assert "Bei 100 Angeboten und 37 nur angesehenen" in text
    assert "**37.1 %**" in text and "exakt 37.1 %" in text and "36.8 %" in text
    assert "das 0.81-Fache" in text and "in 63 % der Fälle unter den besten 5" in text
    assert "sortiert" not in text


@pytest.mark.parametrize("erfolg,art", [(0.30, "success"), (0.2999, "info"), (0.10, "info"), (0.0999, "warning"), (0.0, "warning")])
def test_verdict_a_schwellen(erfolg, art):
    assert ev.verdict_a(_a(erfolg))[0] == art


def test_verdict_a_nennt_die_sortierung_nur_wenn_es_eine_gibt():
    assert "Die ersten 50 % der Folge sind aufsteigend sortiert." in ev.verdict_a(_a(0.0, sort=50))[1]
    assert "sortiert" not in ev.verdict_a(_a(0.37))[1]


def test_verdict_growth():
    rows = [{"n": 3, "bestes k": 1, "k/n": 1 / 3, "Erfolg": 0.5}, {"n": 10000, "bestes k": 3679, "k/n": 0.3679, "Erfolg": 0.367911}]
    kind, text = ev.verdict_growth(rows)
    assert kind == "info"
    assert "Bei n = 3 ist das beste k = 1 (Erfolg 0.5000)" in text
    assert "bei n = 10000 ist es k = 3679 (Anteil 0.3679, Erfolg 0.3679)" in text


def test_verdict_order():
    rows = [{"Reihenfolge": "zufällig", "Anteil": 0, "Erfolg": 0.376},
            {"Reihenfolge": "die ersten 50 % aufsteigend", "Anteil": 50, "Erfolg": 0.0},
            {"Reihenfolge": "aufsteigend", "Anteil": 100, "Erfolg": 0.0},
            {"Reihenfolge": "absteigend", "Anteil": -1, "Erfolg": 0.0}]
    kind, text = ev.verdict_order(rows)
    assert kind == "warning"
    assert "zufälliger Reihenfolge findet die Regel den Besten in 37.6 % der Fälle" in text
    assert "in 0.0 % und 0.0 %" in text and "**0.0 %**" in text


def test_verdict_value_waehlt_je_ziel_die_beste_zeile():
    rows = [{"k": 0, "Erfolg": 0.01, "Wert/Maximum": 0.50, "unter den besten 5": 0.05},
            {"k": 10, "Erfolg": 0.233, "Wert/Maximum": 0.913, "unter den besten 5": 0.62},
            {"k": 37, "Erfolg": 0.372, "Wert/Maximum": 0.810, "unter den besten 5": 0.63}]
    kind, text = ev.verdict_value(rows)
    assert kind == "info"
    assert "wartet k = 37 (Erfolg 37.2 %, Wert/Maximum 0.810)" in text
    assert "wartet k = 10 (Wert/Maximum 0.913, Erfolg 23.3 %)" in text


def _b(regel, ordnung, value, emax=1.99, threshold=None):
    s = ev.Settings(regel_b=regel, ordnung_b=ordnung)
    return ev.AnalysisB(s, emax, value, ev.ratio(value, emax), threshold, [])


def test_verdict_b_zahlen_und_schwellenzusatz():
    kind, text = ev.verdict_b(_b("Schwelle E[max]/2", "kleine zuerst", 1.0, threshold=0.995))
    assert kind == "warning"
    assert "**Schwelle E[max]/2** erreicht bei „kleine zuerst“ im Mittel **1.000**" in text
    assert "**0.503-Fache** des erwarteten Maximums 1.990" in text and "Garantie der Schwellenregel: 0,5" in text
    assert "Die Schwelle ist 0.995." in text
    assert "Die Schwelle" not in ev.verdict_b(_b("immer das erste", "kleine zuerst", 1.0))[1]


@pytest.mark.parametrize("anteil,art", [(1.0, "success"), (0.9, "success"), (0.8999, "info"), (0.6, "info"), (0.5999, "warning"), (0.5, "warning")])
def test_verdict_b_schwellen(anteil, art):
    assert ev.verdict_b(_b("Schwelle E[max]/2", "zufällig", anteil * 2.0, emax=2.0))[0] == art


def test_verdict_instances():
    res = {"zufällig": {"Mittel": 0.854, "Minimum": 0.756}, "aufsteigend nach E[X]": {"Mittel": 0.800, "Minimum": 0.680}, "absteigend nach E[X]": {"Mittel": 0.866, "Minimum": 0.759}}
    kind, text = ev.verdict_instances(res)
    assert kind == "info"
    assert "im Mittel 0.854 des erwarteten Maximums (Minimum 0.756)" in text
    assert "aufsteigend nach E[X] 0.800, absteigend 0.866" in text and "kleinste Wert aller Instanzen und Reihenfolgen ist 0.680" in text


def test_verdict_iid():
    rows = [{"n": 2, "optimal": 0.9375, "Schwelle E[max]/2": 0.9167, "Schwelle Median des Maximums": 0.9053},
            {"n": 100, "optimal": 0.9910, "Schwelle E[max]/2": 0.7550, "Schwelle Median des Maximums": 0.7540}]
    kind, text = ev.verdict_iid(rows)
    assert kind == "info"
    assert "Bei n = 100" in text and "0.9910" in text and "0.7550" in text and "0.7540" in text and "23.6 Punkte Unterschied" in text


def test_delta_text():
    assert ev.delta_text(0.3708, 0.3710, "die exakte Zahl") == ("gleich wie die exakte Zahl", "off")
    assert ev.delta_text(0.3763, 0.3710, "die exakte Zahl") == ("+0.5 Punkte gegen die exakte Zahl", "normal")
    assert ev.delta_text(0.0, 0.3710, "die exakte Zahl") == ("-37.1 Punkte gegen die exakte Zahl", "normal")
