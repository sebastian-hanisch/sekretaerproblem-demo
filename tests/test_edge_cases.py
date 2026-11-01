"""Randfälle: kleinste und größte Folgen, Wartezeit 0 und n − 1, ganz sortierte Folgen, alle Reglerstellungen laufen durch."""

import numpy as np
import pytest

import sek_algorithm as A
import sek_constants as C
import sek_evaluation as ev
from sek_scenario import draw_batch


def test_wartezeit_null_waehlt_das_erste_angebot():
    a = ev.analyse_a(ev.Settings(n=20, kpct=0))
    assert a.k == 0 and a.example["chosen"] == 0
    assert a.exact == pytest.approx(1 / 20)
    assert a.stats["Erfolg"] == pytest.approx(1 / 20, abs=0.02)


def test_hoechste_wartezeit_laesst_nur_das_letzte_uebrig():
    a = ev.analyse_a(ev.Settings(n=10, kpct=C.KPCT_MAX))
    assert a.k == 9 and a.example["chosen"] == 9
    assert a.exact == pytest.approx(0.1)                                    # (9/10) · (1/9)


@pytest.mark.parametrize("n", C.N_OPTIONS)
def test_jede_folgenlaenge_laeuft_und_haelt_die_kennzahlen_im_bereich(n):
    a = ev.analyse_a(ev.Settings(n=n))
    assert 0 <= a.stats["Erfolg"] <= 1 and 0 < a.stats["Wert/Maximum"] <= 1 and 0 <= a.stats["unter den besten 5"] <= 1
    assert a.stats["Erfolg"] <= a.stats["unter den besten 5"]
    assert len(a.example["values"]) == n and 0 <= a.example["chosen"] < n and 0 <= a.example["best"] < n
    ks, exact, est = a.curve
    assert len(ks) == len(exact) == len(est) and ks[0] == 0 and ks[-1] == n - 1 and ks == sorted(set(ks))


def test_kurve_hat_hoechstens_so_viele_punkte_wie_angebote():
    ks, _, _ = ev.analyse_a(ev.Settings(n=10)).curve
    assert len(ks) == 10
    ks, _, _ = ev.analyse_a(ev.Settings(n=1000)).curve
    assert len(ks) <= C.CURVE_POINTS


def test_ganz_sortierte_folge_und_alle_wartezeiten():
    for kpct in (0, 37, C.KPCT_MAX):
        a = ev.analyse_a(ev.Settings(n=50, kpct=kpct, sort=100))
        assert a.stats["Erfolg"] == (1.0 if a.k == 49 else 0.0)


def test_weniger_als_fuenf_angebote_zaehlen_alle_zu_den_besten():
    V = draw_batch(3, 0, 50, 1)
    assert A.stats(V, 1)["unter den besten 5"] == 1.0


@pytest.mark.parametrize("seed", [0, C.SEED_MAX])
def test_grenzen_des_seeds(seed):
    a = ev.analyse_a(ev.Settings(seed=seed))
    assert 0 < a.stats["Erfolg"] < 1


def test_gleicher_seed_gibt_dasselbe_ergebnis_anderer_seed_ein_anderes():
    a, b, c = ev.analyse_a(ev.Settings(seed=1)), ev.analyse_a(ev.Settings(seed=1)), ev.analyse_a(ev.Settings(seed=2))
    assert a.stats == b.stats and a.example == b.example
    assert a.example != c.example


@pytest.mark.parametrize("eps", C.EPS_OPTIONS)
@pytest.mark.parametrize("ordnung", C.ORDNUNGEN_B)
@pytest.mark.parametrize("regel", C.REGELN_B)
def test_jede_stellung_von_teil_b_laeuft(eps, ordnung, regel):
    b = ev.analyse_b(ev.Settings(eps=eps, ordnung_b=ordnung, regel_b=regel))
    assert b.emax == pytest.approx(2 - eps)
    assert 0.5 - 1e-12 <= b.ratio <= 1 + 1e-12
    assert (b.threshold is None) == (regel in ("immer das erste", "optimal (Rückwärtsrechnung)"))
    assert len(b.table) == len(C.REGELN_B)
    kind, text = ev.verdict_b(b)
    assert kind in ("success", "info", "warning") and text


def test_ohne_maximum_ist_der_quotient_eins():
    assert ev.ratio(0.0, 0.0) == 1.0
    assert ev.ratio(1.0, 2.0) == 0.5
