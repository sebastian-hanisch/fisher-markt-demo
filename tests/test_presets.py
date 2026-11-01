"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen."""

import fsh_constants as C
import fsh_evaluation as ev
import fsh_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["kind"] in C.KINDS and preset["budgets"] in C.BUDGET_MODES, name
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.M_MIN <= preset["m"] <= C.M_MAX, name
        assert C.STEPS_MIN <= preset["steps"] <= C.STEPS_MAX and C.ETA_MIN <= preset["eta"] <= C.ETA_MAX and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_preset_runs_and_is_an_equilibrium():
    for name, preset in C.PRESETS.items():
        a = ev.analyse(ev.Settings(**preset))
        assert a.eq.exact and a.worst_gap < 1e-9, name
