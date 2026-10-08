"""Generate a SYNTHETIC UV-Vis transmittance spectrum of an AZO thin film.

The original measurements from the 2023 project are not available, so this
script creates realistic example data to demonstrate the analysis pipeline.
The data are NOT experimental measurements.

Model:
    alpha(E) = alpha_max / (1 + exp(-(E - Eg) / width)) + alpha_background
    T(lambda) = (1 - R) * exp(-alpha * d) + noise

On purpose, the file also includes a few common data problems
(percent units, a duplicated wavelength, missing and non-numeric values)
so that the cleaning step has something to do.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HC_EV_NM = 1239.84193

# Parameters chosen to resemble a 300 nm AZO film on glass.
BAND_GAP_EV = 3.46          # value used to generate the data
EDGE_WIDTH_EV = 0.05
ALPHA_MAX_CM = 1.5e5
ALPHA_BACKGROUND_CM = 2.0e3
THICKNESS_NM = 300.0
REFLECTANCE = 0.10
NOISE_STD = 0.002


def make_spectrum(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wavelength = np.arange(300.0, 900.5, 1.0)
    energy = HC_EV_NM / wavelength

    alpha = ALPHA_MAX_CM / (1 + np.exp(-(energy - BAND_GAP_EV) / EDGE_WIDTH_EV))
    alpha += ALPHA_BACKGROUND_CM
    transmittance = (1 - REFLECTANCE) * np.exp(-alpha * THICKNESS_NM * 1e-7)
    transmittance += rng.normal(0, NOISE_STD, size=transmittance.size)
    transmittance = np.clip(transmittance, 1e-4, 1.0)

    df = pd.DataFrame({
        "wavelength_nm": wavelength,
        "transmittance": np.round(transmittance * 100, 3),  # exported as %T
    }).astype({"transmittance": object})

    # Inject typical export problems.
    df.loc[150, "transmittance"] = np.nan          # missing reading
    df.loc[420, "transmittance"] = "ERR"           # instrument error flag
    df = pd.concat([df, df.iloc[[250]]])           # duplicated wavelength
    return df.sample(frac=1, random_state=seed)    # unsorted rows


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_azo_transmittance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_spectrum().to_csv(out, index=False)
    print(f"Synthetic spectrum written to {out}")
