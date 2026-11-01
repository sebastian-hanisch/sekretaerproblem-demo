"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink; feste Experimente."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import sek_constants as C
import sek_evaluation as ev
import sek_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_ranges():
    for name, preset in C.PRESETS.items():
        assert preset["n"] in C.N_OPTIONS and preset["eps"] in C.EPS_OPTIONS, name
        assert C.KPCT_MIN <= preset["kpct"] <= C.KPCT_MAX and C.SORT_MIN <= preset["sort"] <= C.SORT_MAX, name
        assert (preset["sort"] - C.SORT_MIN) % C.SORT_STEP == 0, name
        assert preset["ordnung_b"] in C.ORDNUNGEN_B and preset["regel_b"] in C.REGELN_B and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**next(iter(C.PRESETS.values()))) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_choice_has_a_help_text_and_defaults_are_valid():
    assert set(C.REGEL_B_HILFE) == set(C.REGELN_B) and set(C.ORDNUNG_B_HILFE) == set(C.ORDNUNGEN_B)
    assert C.DEFAULT_REGEL_B in C.REGELN_B and C.DEFAULT_ORDNUNG_B in C.ORDNUNGEN_B and C.DEFAULT_N in C.N_OPTIONS and C.DEFAULT_EPS in C.EPS_OPTIONS
    assert list(C.N_OPTIONS) == sorted(C.N_OPTIONS) and list(C.EPS_OPTIONS) == sorted(C.EPS_OPTIONS)
    assert set(C.COLORS_B) == set(C.REGELN_B)
    assert C.KPCT_MIN <= C.DEFAULT_KPCT <= C.KPCT_MAX and C.SORT_MIN <= C.DEFAULT_SORT <= C.SORT_MAX


def test_experiment_grids_are_inside_the_controls():
    assert C.ORDER_N in C.N_OPTIONS and C.VALUE_N in C.N_OPTIONS
    assert C.ORDER_K == ev.wait_count(C.ORDER_N, C.DEFAULT_KPCT)
    assert set(C.ORDER_SHARES) <= set(range(C.SORT_MIN, C.SORT_MAX + 1, C.SORT_STEP))
    assert all(0 <= k < C.VALUE_N for k in C.VALUE_KS) and list(C.GROWTH_NS) == sorted(C.GROWTH_NS)


def test_experiment_rows_have_the_documented_shape():
    rows = ev.order_rows(n=20, k=7, trials=200)
    assert [r["Anteil"] for r in rows] == list(C.ORDER_SHARES) + [-1]
    assert [r["k"] for r in ev.value_rows(n=20, ks=(0, 5, 10), trials=200)] == [0, 5, 10]
    assert [r["n"] for r in ev.growth_rows()] == list(C.GROWTH_NS)
    res = ev.instance_rows(count=2, nvars=3, samples=500)
    assert set(res) == {"zufällig", "aufsteigend nach E[X]", "absteigend nach E[X]"} and all(set(v) == {"Mittel", "Minimum"} for v in res.values())
    assert [r["n"] for r in ev.iid_rows()] == list(C.IID_NS)


def _app_with(params: dict) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_permalink_uebernimmt_gueltige_werte():
    at = _app_with({"n": "50", "kpct": "20", "sort": "25", "eps": "0.1", "ordnung_b": "große zuerst", "regel_b": "immer das erste", "seed": "5"})
    state = {k: at.session_state[k] for k in P.SETTING_SPECS}
    assert state == {"n_select": 50, "kpct_slider": 20, "sort_slider": 25, "eps_select": 0.1, "ordnung_b_select": "große zuerst", "regel_b_select": "immer das erste", "seed_input": 5}


def test_permalink_begrenzt_und_ignoriert_ungueltiges():
    at = _app_with({"kpct": "500", "sort": "-4", "n": "77", "eps": "abc", "regel_b": "gibt es nicht", "seed": "99999999"})
    assert at.session_state["kpct_slider"] == C.KPCT_MAX
    assert at.session_state["sort_slider"] == C.SORT_MIN
    assert at.session_state["n_select"] == C.DEFAULT_N                    # 77 ist keine erlaubte Auswahl
    assert at.session_state["eps_select"] == C.DEFAULT_EPS                # keine Zahl
    assert at.session_state["regel_b_select"] == C.DEFAULT_REGEL_B
    assert at.session_state["seed_input"] == C.SEED_MAX
