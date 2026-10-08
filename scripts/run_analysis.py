"""Run the band gap analysis on a transmittance spectrum and save a figure.

Usage:
    python scripts/run_analysis.py [path/to/spectrum.csv] [thickness_nm]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bandgap import analyze_spectrum, load_spectrum, sigmoid  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "synthetic_azo_transmittance.csv"
DEFAULT_THICKNESS_NM = 300.0


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    thickness = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THICKNESS_NM

    spectrum = load_spectrum(path)
    result, data = analyze_spectrum(spectrum, thickness_nm=thickness)

    print(f"Points after cleaning: {len(spectrum)}")
    print(f"Band gap (sigmoid inflection point): "
          f"{result.band_gap_ev:.4f} ± {result.uncertainty_ev:.4f} eV")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(data["wavelength_nm"], data["transmittance"] * 100, lw=1.2)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Transmittance (%)")
    ax1.set_title("UV-Vis transmittance")

    e_fit = np.linspace(data["energy_ev"].min(), data["energy_ev"].max(), 500)
    ax2.plot(data["energy_ev"], data["alpha_cm"], ".", ms=3, label="Data")
    ax2.plot(e_fit, sigmoid(e_fit, result.amplitude, result.slope, result.band_gap_ev),
             lw=2, label="Sigmoid fit")
    ax2.axvline(result.band_gap_ev, ls="--", lw=1, color="gray")
    ax2.annotate(f"$E_g$ = {result.band_gap_ev:.3f} eV",
                 xy=(result.band_gap_ev, result.amplitude / 2),
                 xytext=(result.band_gap_ev - 1.1, result.amplitude * 0.75),
                 arrowprops=dict(arrowstyle="->"))
    ax2.set_xlabel("Photon energy (eV)")
    ax2.set_ylabel(r"Absorption coefficient $\alpha$ (cm$^{-1}$)")
    ax2.set_title("Sigmoid fit of the absorption edge")
    ax2.legend()

    fig.tight_layout()
    out = ROOT / "figures" / "bandgap_fit.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
