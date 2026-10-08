import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bandgap import (  # noqa: E402
    absorption_coefficient,
    analyze_spectrum,
    clean_spectrum,
    fit_band_gap,
    sigmoid,
    wavelength_to_energy,
)


# --- Unit conversions -------------------------------------------------------
def test_wavelength_to_energy_known_value():
    # 1239.84 nm corresponds to 1 eV by definition of hc.
    assert wavelength_to_energy(1239.84193) == pytest.approx(1.0)


def test_wavelength_to_energy_rejects_non_positive():
    with pytest.raises(ValueError):
        wavelength_to_energy([500, 0])


# --- Beer-Lambert -----------------------------------------------------------
def test_absorption_coefficient_known_value():
    # T = e^-1 through 300 nm  =>  alpha = 1 / 3e-5 cm
    alpha = absorption_coefficient(np.exp(-1), thickness_nm=300)
    assert alpha == pytest.approx(1 / 3e-5)


def test_absorption_coefficient_zero_for_full_transmission():
    assert absorption_coefficient(1.0, thickness_nm=300) == pytest.approx(0.0)


@pytest.mark.parametrize("bad_t", [0.0, -0.1, 1.2])
def test_absorption_coefficient_rejects_unphysical_transmittance(bad_t):
    with pytest.raises(ValueError):
        absorption_coefficient(bad_t, thickness_nm=300)


# --- Cleaning ---------------------------------------------------------------
def test_clean_spectrum_handles_messy_input():
    raw = pd.DataFrame({
        "wavelength_nm": [500, 400, 400, 450, 600, 700],
        "transmittance": [85.0, 60.0, 61.0, "ERR", None, 90.0],  # %T with problems
    })
    clean = clean_spectrum(raw)

    assert list(clean["wavelength_nm"]) == [400, 500, 700]       # sorted, deduplicated
    assert clean["transmittance"].between(0, 1).all()             # converted to fraction
    assert clean.loc[0, "transmittance"] == pytest.approx(0.60)   # first duplicate kept


def test_clean_spectrum_requires_columns():
    with pytest.raises(ValueError):
        clean_spectrum(pd.DataFrame({"wavelength": [400], "T": [0.5]}))


# --- Fitting ----------------------------------------------------------------
def test_fit_recovers_known_band_gap_without_noise():
    energy = np.linspace(1.4, 4.1, 300)
    alpha = sigmoid(energy, a=1.5e5, b=20.0, c=3.46)
    result = fit_band_gap(energy, alpha)
    assert result.band_gap_ev == pytest.approx(3.46, abs=1e-6)


def test_fit_recovers_known_band_gap_with_noise():
    rng = np.random.default_rng(0)
    energy = np.linspace(1.4, 4.1, 300)
    alpha = sigmoid(energy, a=1.5e5, b=20.0, c=3.46) + rng.normal(0, 2e3, energy.size)
    result = fit_band_gap(energy, alpha)
    assert result.band_gap_ev == pytest.approx(3.46, abs=0.01)
    assert result.uncertainty_ev < 0.01


def test_fit_requires_enough_points():
    with pytest.raises(ValueError):
        fit_band_gap([3.0, 3.5, 4.0], [1, 2, 3])


# --- End to end -------------------------------------------------------------
def test_full_pipeline_on_synthetic_spectrum():
    # Build a spectrum from a known band gap and check the pipeline recovers it.
    wavelength = np.arange(300.0, 901.0, 1.0)
    energy = wavelength_to_energy(wavelength)
    alpha = 1.5e5 / (1 + np.exp(-(energy - 3.46) / 0.05))
    transmittance = np.exp(-alpha * 300e-7)
    df = pd.DataFrame({"wavelength_nm": wavelength, "transmittance": transmittance})

    result, data = analyze_spectrum(clean_spectrum(df), thickness_nm=300)

    assert result.band_gap_ev == pytest.approx(3.46, abs=0.005)
    assert {"energy_ev", "alpha_cm"} <= set(data.columns)
