"""The parameter table is load-bearing: if it breaks, every result is unprovenanced."""
import pytest

from phenei_gym.params import PARAMS, ParamError


def test_table_loads():
    assert len(PARAMS) > 100


def test_scalar_and_range_parsing():
    assert PARAMS.value("SIM-E01") == pytest.approx(1.6242)      # lunar g
    assert PARAMS.range("SIM-P01") == (3000.0, 6000.0)           # Mattia's Tmax range
    assert PARAMS.range("SIM-E06") == (0.0, 10.0)                # slope, "0 to 10 (uniform)"
    assert PARAMS.range("SIM-M01") == (500.0, 500.0)             # scalar as degenerate range


def test_unknown_id_fails_loudly_with_a_hint():
    with pytest.raises(ParamError, match="unknown parameter"):
        PARAMS.value("SIM-E0")


def test_non_scalar_refuses_to_be_a_float():
    with pytest.raises(ParamError, match="not a scalar"):
        PARAMS.value("SIM-P12")          # "27.1 (at 3 kN) to 13.5 (at 6 kN)"


def test_approach_state_is_self_consistent():
    """SIM-A07/A08 must be the components of SIM-A05 at angle SIM-A06."""
    import math

    v = PARAMS.value("SIM-A05")
    fpa = math.radians(PARAMS.value("SIM-A06"))
    assert PARAMS.value("SIM-A07") == pytest.approx(v * math.sin(fpa), rel=1e-3)
    assert PARAMS.value("SIM-A08") == pytest.approx(v * math.cos(fpa), rel=1e-3)


def test_derived_gravity_matches_mu_and_radius():
    mu = PARAMS.value("SIM-E02")
    r = PARAMS.value("SIM-E03") * 1e3
    assert PARAMS.value("SIM-E01") == pytest.approx(mu / r**2, rel=1e-4)


def test_hover_throttle_exceeds_floor_only_at_low_thrust():
    """The TR-SIM-09 finding, as an executable assertion."""
    w = PARAMS.value("SIM-M01") * PARAMS.value("SIM-E01")
    floor = PARAMS.value("SIM-P04") / 100.0
    t_lo, t_hi = PARAMS.range("SIM-P01")
    assert w / t_hi < floor          # 6 kN: four engines cannot hover
    assert 0.5 * floor * t_lo < w    # 3 kN: two engines can (2 of 4 engines = Tmax/2)
