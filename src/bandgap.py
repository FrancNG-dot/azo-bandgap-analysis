"""Optical band gap estimation for thin films from UV-Vis transmittance.

Pipeline:
    1. Load and clean a transmittance spectrum (wavelength vs. T).
    2. Convert wavelength to photon energy.
    3. Compute the absorption coefficient with the Beer-Lambert law.
    4. Fit a sigmoid to alpha(E) and take its inflection point as the band gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# h*c in eV*nm, used to convert wavelength (nm) to photon energy (eV).
HC_EV_NM = 1239.84193


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------
def clean_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean spectrum with columns ``wavelength_nm`` and ``transmittance``.

    Cleaning steps:
      * coerce both columns to numbers and drop rows that are not numeric or missing;
      * drop duplicated wavelengths (keeping the first measurement);
      * convert transmittance from percent to a 0-1 fraction when needed;
      * drop physically impossible values (T <= 0 or T > 1);
      * sort by wavelength.
    """
    required = {"wavelength_nm", "transmittance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[["wavelength_nm", "transmittance"]].apply(pd.to_numeric, errors="coerce")
    out = out.dropna()
    out = out.drop_duplicates(subset="wavelength_nm", keep="first")

    # Spectrophotometers often export %T. Values above 1 mean percent units.
    if out["transmittance"].max() > 1.0:
        out["transmittance"] = out["transmittance"] / 100.0

    valid = (out["transmittance"] > 0) & (out["transmittance"] <= 1)
    out = out[valid & (out["wavelength_nm"] > 0)]

    return out.sort_values("wavelength_nm").reset_index(drop=True)


def load_spectrum(path: str) -> pd.DataFrame:
    """Read a CSV spectrum and return it cleaned (see :func:`clean_spectrum`)."""
    return clean_spectrum(pd.read_csv(path))


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def wavelength_to_energy(wavelength_nm):
    """Convert wavelength in nm to photon energy in eV (E = hc / lambda)."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    if np.any(wavelength_nm <= 0):
        raise ValueError("Wavelengths must be positive.")
    return HC_EV_NM / wavelength_nm


def absorption_coefficient(transmittance, thickness_nm: float):
    """Absorption coefficient in cm^-1 from the Beer-Lambert law.

    I = I0 * exp(-alpha * d)  =>  alpha = -ln(T) / d
    """
    transmittance = np.asarray(transmittance, dtype=float)
    if thickness_nm <= 0:
        raise ValueError("Film thickness must be positive.")
    if np.any((transmittance <= 0) | (transmittance > 1)):
        raise ValueError("Transmittance must be in the interval (0, 1].")
    thickness_cm = thickness_nm * 1e-7
    return -np.log(transmittance) / thickness_cm


def sigmoid(energy, a, b, c):
    """Sigmoid model alpha(E) = a / (1 + exp(-b (E - c)))."""
    return a / (1.0 + np.exp(-b * (energy - c)))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass
class BandGapResult:
    band_gap_ev: float      # inflection point c of the sigmoid
    uncertainty_ev: float   # one standard deviation from the fit covariance
    amplitude: float        # a, in cm^-1
    slope: float            # b, in eV^-1
    n_points: int


def fit_band_gap(energy, alpha, e_min: float | None = None,
                 e_max: float | None = None) -> BandGapResult:
    """Fit the sigmoid model to alpha(E) and return the band gap estimate.

    ``e_min`` and ``e_max`` optionally restrict the fit to an energy window
    around the absorption edge.
    """
    energy = np.asarray(energy, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    mask = np.isfinite(energy) & np.isfinite(alpha)
    if e_min is not None:
        mask &= energy >= e_min
    if e_max is not None:
        mask &= energy <= e_max
    energy, alpha = energy[mask], alpha[mask]

    if energy.size < 4:
        raise ValueError("Not enough points to fit the sigmoid (need at least 4).")

    # Initial guesses: amplitude from the data, edge at the steepest rise.
    order = np.argsort(energy)
    e_sorted, a_sorted = energy[order], alpha[order]
    a0 = np.nanmax(a_sorted)
    c0 = e_sorted[np.argmax(np.gradient(a_sorted, e_sorted))]
    p0 = [a0, 20.0, c0]

    params, cov = curve_fit(sigmoid, energy, alpha, p0=p0, maxfev=20000)
    a, b, c = params
    c_err = float(np.sqrt(cov[2, 2])) if np.all(np.isfinite(cov)) else float("nan")

    return BandGapResult(band_gap_ev=float(c), uncertainty_ev=c_err,
                         amplitude=float(a), slope=float(b), n_points=int(energy.size))


def analyze_spectrum(df: pd.DataFrame, thickness_nm: float, **fit_kwargs):
    """Run the full pipeline on a clean spectrum.

    Returns the fit result and a DataFrame with energy and alpha added.
    """
    data = df.copy()
    data["energy_ev"] = wavelength_to_energy(data["wavelength_nm"])
    data["alpha_cm"] = absorption_coefficient(data["transmittance"], thickness_nm)
    result = fit_band_gap(data["energy_ev"], data["alpha_cm"], **fit_kwargs)
    return result, data
