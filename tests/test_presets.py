"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Optionen der Auswahlregler."""

import kvr_constants as C
import kvr_evaluation as ev
import kvr_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for preset in C.PRESETS.values():
        assert preset["gitter"] in C.GRIDS and preset["kontrolle"] in C.CONTROLS
        assert C.P_MIN <= preset["p"] <= C.P_MAX
        assert preset["n"] in C.N_CHOICES
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_options_of_the_specs_match_the_constants():
    assert P.options("gitter_select") == tuple(C.GRIDS)
    assert P.options("kontrolle_select") == tuple(C.CONTROLS)
    assert P.options("n_select") == C.N_CHOICES and list(C.N_CHOICES) == sorted(C.N_CHOICES)
    assert P.options("p_slider") is None


def test_every_grid_can_be_enumerated_live():
    from kvr_scenario import named_netz

    for key in C.GRIDS:
        assert named_netz(key).m <= C.LIVE_EDGE_LIMIT


def test_every_preset_is_a_distinct_situation():
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_control_keys_refer_to_known_quantities():
    assert set(C.CONTROL_LABELS) == set(C.CONTROLS)
    assert {nm for names in C.CONTROLS.values() for nm in names} == set(C.CONTROL_NAMES)
