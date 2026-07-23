"""Published priors for Speculate's five bundled HST observations."""

# Table A1 uses ``value ± uncertainty`` for Normal priors and a dash-separated
# lower/upper interval for Uniform priors. Keys are the fixed filenames shipped
# in observation_files/ and shared by Benchmark, Inference, and Quick Fit.
OBSERVATION_PRIORS = {
    "cv:ixvel_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 90.36, "sigma": 0.15},
        "inclination_deg": {"kind": "normal", "mean": 57.0, "sigma": 2.0},
    },
    "cv:rwsex_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 223.12, "sigma": 1.23},
        "inclination_deg": {"kind": "uniform", "min": 28.0, "max": 40.0},
    },
    "cv:rwtri_dered.csv": {
        "distance_pc": {"kind": "normal", "mean": 306.08, "sigma": 2.09},
        "inclination_deg": {"kind": "normal", "mean": 72.5, "sigma": 2.5},
    },
    "cv:uxuma_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 291.95, "sigma": 1.34},
        "inclination_deg": {"kind": "normal", "mean": 71.0, "sigma": 0.6},
    },
    "cv:v3885sgr_all.csv": {
        "distance_pc": {"kind": "normal", "mean": 128.97, "sigma": 0.57},
        "inclination_deg": {"kind": "uniform", "min": 45.0, "max": 70.0},
    },
}
