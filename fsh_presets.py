"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import fsh_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: object = None
    hi: object = None
    choices: tuple | None = None   # Auswahlregler (Radio): lo/hi sind dann zwei verschiedene Auswahlwerte, keine Grenzen


SETTING_SPECS = {
    "kind_radio": SettingSpec("kind", str, C.DEFAULT_KIND, C.KINDS[0], C.KINDS[-1], choices=C.KINDS),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "budgets_radio": SettingSpec("budgets", str, C.DEFAULT_BUDGETS, C.BUDGET_MODES[0], C.BUDGET_MODES[-1], choices=C.BUDGET_MODES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "steps_slider": SettingSpec("steps", int, C.DEFAULT_STEPS, C.STEPS_MIN, C.STEPS_MAX),
    "eta_slider": SettingSpec("eta", float, C.DEFAULT_ETA, C.ETA_MIN, C.ETA_MAX),
}
PRESET_KEYS = {"kind": "kind_radio", "n": "n_slider", "m": "m_slider", "budgets": "budgets_radio", "seed": "seed_input", "steps": "steps_slider", "eta": "eta_slider"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp_value(spec: SettingSpec, value):
    """Wert aus der Adresszeile auf die Reglergrenzen begrenzen; ein Auswahlwert muss eine der Auswahlmöglichkeiten sein (sonst None)."""
    if spec.choices is not None:
        return value if value in spec.choices else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return round(value, 2) if spec.caster is float else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = clamp_value(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), abs_tol=1e-9)
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _same(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
