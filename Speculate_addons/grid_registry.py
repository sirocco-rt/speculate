"""
Shared metadata and conversion helpers for Speculate Sirocco grids.

The training, Quick Fit, inference, benchmark, and export tools all need the
same answers to grid-specific questions: which parameters exist, which axes are
stored in log space, which .spec columns correspond to inclination angles, how
filenames encode parameter IDs, and how emulator-space values map back to
physical Sirocco .pf keywords.  Keeping those facts in one module prevents CV
assumptions from leaking into AGN workflows and makes future grids easier to
add safely.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


# Physical constants used by the AGN conversion helpers.  They intentionally
# live here rather than in the .pf exporter so benchmark truth extraction and
# Tier 3 export use the same Mdot_Edd and R_g conventions.
# Constants (cgs units)
_MSOL = 1.989e33
_YR = 3.1556925e7
_GRAV_CONST = 6.67e-8
_C = 2.99792458e10
_EDD_LUMINOSITY_CONST = 1.39e38


# Sirocco .spec files store frequency in column 0, wavelength in column 1, and
# then one flux column per observer inclination starting at column 2.  These
# tuples define the physical angle represented by each flux column for each grid
# family; inclination_column() applies the shared +2 data-column offset.
CV_INCLINATION_COLUMNS = tuple(range(30, 90, 5))
AGN_INCLINATION_COLUMNS = (10, 25, 40, 55, 70, 85)

# Emulator grid points are already in the coordinate system consumed by the
# Starfish emulator.  Log-scaled axes are stored as log10(value) here; the raw
# physical values are recovered later by emulator_values_to_physical().  CV and
# AGN grids deliberately reuse some parameter IDs for different physical
# meanings, so every conversion path below must go through the active registry
# entry rather than assuming CV semantics.
CV_BL_POINTS: Dict[int, np.ndarray] = {
    1: np.log10(np.array([3e-9, 1e-8, 3e-8], dtype=float)),
    2: np.array([0.03, 0.1, 0.3], dtype=float),
    3: np.log10(np.array([0.55, 5.5, 55.0], dtype=float)),
    4: np.array([0.0, 0.25, 1.0], dtype=float),
    5: np.log10(np.array([7.25182e8, 7.25182e9, 7.25182e10], dtype=float)),
    6: np.array([0.5, 1.5, 4.5], dtype=float),
    7: np.array([0.0, 0.3, 1.0], dtype=float),
    8: np.array([0.1, 0.3, 1.0], dtype=float),
    9: np.array([30, 55, 80], dtype=float),
    10: np.array([30, 40, 50, 60, 70, 80], dtype=float),
    11: np.array(CV_INCLINATION_COLUMNS, dtype=float),
}

CV_NO_BL_POINTS: Dict[int, np.ndarray] = {
    key: value for key, value in CV_BL_POINTS.items() if key not in (7, 8)
}

# The AGN grid has three-point axes for parameters 1-5 and two-point axes for
# parameters 6-8.  Two-point axes use the higher value as their default because
# there is no central grid point; see default_grid_value().
AGN_POINTS: Dict[int, np.ndarray] = {
    1: np.log10(np.array([1e7, 1e8, 1e9], dtype=float)),
    2: np.log10(np.array([0.025, 0.1, 0.4], dtype=float)),
    3: np.log10(np.array([0.03, 0.3, 3.0], dtype=float)),
    4: np.log10(np.array([0.01, 0.1, 1.0], dtype=float)),
    5: np.log10(np.array([5.0, 30.0, 180.0], dtype=float)),
    6: np.array([0.0, 1.0], dtype=float),
    7: np.log10(np.array([750.0, 7500.0], dtype=float)),
    8: np.array([1.5, 3.5], dtype=float),
    9: np.array([10, 55, 85], dtype=float),
    10: np.array(AGN_INCLINATION_COLUMNS, dtype=float),
}

# Short labels are used in sliders, diagnostics, benchmark reports, and saved
# metadata.  They are deliberately stable and compact; longer descriptions live
# in *_PARAM_DESCRIPTIONS for GridInterface.parameters_description().
CV_PARAM_LABELS = {
    1: "Disk.mdot",
    2: "Wind.mdot",
    3: "KWD.d",
    4: "KWD.mdot_r_exponent",
    5: "KWD.acceleration_length",
    6: "KWD.acceleration_exponent",
    7: "Boundary_layer.luminosity",
    8: "Boundary_layer.temp",
    9: "Inclination (Sparse)",
    10: "Inclination (Mid)",
    11: "Inclination (Full)",
}

AGN_PARAM_LABELS = {
    1: "Central_object.mass",
    2: "Disk.mdot / Mdot_Edd",
    3: "Wind.mdot / Disk.mdot",
    4: "Wind.filling_factor",
    5: "KWD.d",
    6: "KWD.mdot_r_exponent",
    7: "KWD.acceleration_length / R_g",
    8: "KWD.acceleration_exponent",
    9: "Inclination (Sparse)",
    10: "Inclination (Full)",
}

# Descriptions are shown by Training and Quick Fit when building their parameter
# selectors from the grid interface.  Inclination descriptions include the axis
# coverage because those parameters select among existing .spec flux columns.
CV_PARAM_DESCRIPTIONS = {
    1: "Disk.mdot (msol/yr)",
    2: "Wind.mdot (Disk.mdot)",
    3: "KWD.d (in_units_of_Rstar)",
    4: "KWD.mdot_r_exponent",
    5: "KWD.acceleration_length (cm)",
    6: "KWD.acceleration_exponent",
    7: "Boundary_layer.luminosity(ergs /s)",
    8: "Boundary_layer.temp(K)",
    9: "Inclination angle - sparse (30, 55, 80 degrees)",
    10: "Inclination angle - mid (30, 40, 50, 60, 70, 80 degrees)",
    11: "Inclination angle - full (30-85 degrees, 5° steps)",
}

AGN_PARAM_DESCRIPTIONS = {
    1: "Central_object.mass (msol)",
    2: "Disk.mdot (Mdot_Edd fraction)",
    3: "Wind.mdot (Disk.mdot fraction)",
    4: "Wind.filling_factor (1=smooth, <1=clumped)",
    5: "KWD.d (in_units_of_rstar)",
    6: "KWD.mdot_r_exponent",
    7: "KWD.acceleration_length (R_g multiplier)",
    8: "KWD.acceleration_exponent",
    9: "Inclination angle - sparse (10, 55, 85 degrees)",
    10: "Inclination angle - full (10, 25, 40, 55, 70, 85 degrees)",
}

# Benchmark maps pair each friendly report label with the Sirocco .pf keyword
# that receives the physical value during Tier 3 export.
CV_BENCHMARK_MAP = {
    1: ("disk.mdot", "Disk.mdot(msol/yr)"),
    2: ("wind.mdot", "Wind.mdot(msol/yr)"),
    3: ("KWD.d", "KWD.d(in_units_of_rstar)"),
    4: ("KWD.mdot_r_exponent", "KWD.mdot_r_exponent"),
    5: ("KWD.acceleration_length", "KWD.acceleration_length(cm)"),
    6: ("KWD.acceleration_exponent", "KWD.acceleration_exponent"),
    7: ("Boundary_layer.luminosity", "Boundary_layer.luminosity(ergs/s)"),
    8: ("Boundary_layer.temp", "Boundary_layer.temp(K)"),
    9: ("Inclination", "Inclination"),
    10: ("Inclination", "Inclination"),
    11: ("Inclination", "Inclination"),
}

AGN_BENCHMARK_MAP = {
    1: ("Central_object.mass", "Central_object.mass(msol)"),
    2: ("Disk.mdot / Mdot_Edd", "Disk.mdot(msol/yr)"),
    3: ("Wind.mdot / Disk.mdot", "Wind.mdot(msol/yr)"),
    4: ("Wind.filling_factor", "Wind.filling_factor(1=smooth,<1=clumped)"),
    5: ("KWD.d", "KWD.d(in_units_of_rstar)"),
    6: ("KWD.mdot_r_exponent", "KWD.mdot_r_exponent"),
    7: ("KWD.acceleration_length / R_g", "KWD.acceleration_length(cm)"),
    8: ("KWD.acceleration_exponent", "KWD.acceleration_exponent"),
    9: ("Inclination", "Inclination"),
    10: ("Inclination", "Inclination"),
}

# Parameter IDs whose emulator coordinates are log10(value).  Keeping this list
# grid-specific avoids the historical CV mistake of treating every similarly
# named parameter as log-scaled.
CV_LOG_PARAM_IDS = {1, 3, 5}
AGN_LOG_PARAM_IDS = {1, 2, 3, 4, 5, 7}

# Registry entry fields:
# - class_name: GridInterface subclass name, resolved lazily by get_grid_configs
# - type: compact family key used by conversion/export helpers
# - usecols: default (wavelength, flux) columns for raw .spec reads
# - max_params/default_params: full Training Tool parameter set and defaults
# - quickfit_*: Quick Fit-specific parameter set and defaults when they differ
# - file_param_ids: axes that determine the runN.spec file index
# - physical_param_ids: axes required for .pf physical conversion
# - points/labels/descriptions/maps: shared UI, benchmark, and conversion facts
# - inclination_*: trainable inclination parameter IDs and raw .spec flux columns
# - test_grid_name: paired validation grid used by inference and Tier 2
GRID_REGISTRY: Dict[str, Dict[str, Any]] = {
    "speculate_cv_no-bl_grid_v87f": {
        "class_name": "Speculate_cv_no_bl_grid_v87f",
        "type": "cv_no-bl",
        "usecols": (1, 7),
        "name": "speculate_cv_no-bl_grid_v87f",
        "max_params": [1, 2, 3, 4, 5, 6, 9, 10, 11],
        "quickfit_max_params": [1, 2, 3, 4, 5, 6, 11],
        "default_params": [1, 2, 3, 4, 5, 6, 9],
        "quickfit_default_params": [1, 2, 3, 4, 5, 6, 11],
        "file_param_ids": [1, 2, 3, 4, 5, 6],
        "physical_param_ids": [1, 2, 3, 4, 5, 6],
        "points": CV_NO_BL_POINTS,
        "param_labels": CV_PARAM_LABELS,
        "param_descriptions": CV_PARAM_DESCRIPTIONS,
        "benchmark_map": CV_BENCHMARK_MAP,
        "log_param_ids": CV_LOG_PARAM_IDS,
        "inclination_param_ids": {9, 10, 11},
        "inclination_columns": CV_INCLINATION_COLUMNS,
        "default_fixed_inclination": 55,
        "test_grid_name": "speculate_cv_no-bl_testgrid_v87f",
    },
    "speculate_cv_bl_grid_v87f": {
        "class_name": "Speculate_cv_bl_grid_v87f",
        "type": "cv_bl",
        "usecols": (1, 7),
        "name": "speculate_cv_bl_grid_v87f",
        "max_params": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "quickfit_max_params": [1, 2, 3, 4, 5, 6, 7, 8, 11],
        "default_params": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "quickfit_default_params": [1, 2, 3, 4, 5, 6, 7, 8, 11],
        "file_param_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "physical_param_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "points": CV_BL_POINTS,
        "param_labels": CV_PARAM_LABELS,
        "param_descriptions": CV_PARAM_DESCRIPTIONS,
        "benchmark_map": CV_BENCHMARK_MAP,
        "log_param_ids": CV_LOG_PARAM_IDS,
        "inclination_param_ids": {9, 10, 11},
        "inclination_columns": CV_INCLINATION_COLUMNS,
        "default_fixed_inclination": 55,
        "test_grid_name": "speculate_cv_bl_testgrid_v87f",
    },
    "speculate_agn_grid_v1.3": {
        "class_name": "Speculate_agn_grid_v1_3",
        "type": "agn",
        "usecols": (1, 5),
        "name": "speculate_agn_grid_v1.3",
        "max_params": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "quickfit_max_params": [1, 2, 3, 4, 5, 6, 7, 8, 10],
        "default_params": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "quickfit_default_params": [1, 2, 3, 4, 5, 6, 7, 8, 10],
        "file_param_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "physical_param_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "points": AGN_POINTS,
        "param_labels": AGN_PARAM_LABELS,
        "param_descriptions": AGN_PARAM_DESCRIPTIONS,
        "benchmark_map": AGN_BENCHMARK_MAP,
        "log_param_ids": AGN_LOG_PARAM_IDS,
        "inclination_param_ids": {9, 10},
        "inclination_columns": AGN_INCLINATION_COLUMNS,
        "default_fixed_inclination": 55,
        "test_grid_name": "speculate_agn_testgrid_v1.3",
    },
}


def get_grid_config(grid_name: str | None) -> Dict[str, Any] | None:
    """Return the registry entry for an exact or inferable grid name."""
    if not grid_name:
        return None
    if grid_name in GRID_REGISTRY:
        return GRID_REGISTRY[grid_name]
    inferred = infer_grid_name(grid_name)
    return GRID_REGISTRY.get(inferred) if inferred else None


def get_grid_configs(*, quickfit: bool = False) -> Dict[str, Dict[str, Any]]:
    """Return registry entries augmented with concrete GridInterface classes.

    The import is intentionally inside the function to avoid a circular import:
    Spec_gridinterfaces imports this registry for parameter points and
    inclination helpers, while the notebooks need registry entries containing
    the actual classes.
    """
    from Speculate_addons.Spec_gridinterfaces import (
        Speculate_agn_grid_v1_3,
        Speculate_cv_bl_grid_v87f,
        Speculate_cv_no_bl_grid_v87f,
    )

    classes = {
        "Speculate_agn_grid_v1_3": Speculate_agn_grid_v1_3,
        "Speculate_cv_bl_grid_v87f": Speculate_cv_bl_grid_v87f,
        "Speculate_cv_no_bl_grid_v87f": Speculate_cv_no_bl_grid_v87f,
    }
    configs: Dict[str, Dict[str, Any]] = {}
    for grid_name, config in GRID_REGISTRY.items():
        item = deepcopy(config)
        item["class"] = classes[item["class_name"]]
        if quickfit and "quickfit_max_params" in item:
            # Quick Fit trains compact local models and uses its own preferred
            # default inclination axis, while the Training Tool exposes every
            # supported inclination resolution.
            item["max_params"] = list(item["quickfit_max_params"])
            item["default_params"] = list(item.get("quickfit_default_params", item["max_params"]))
        configs[grid_name] = item
    return configs


def infer_grid_name(value: str | None) -> str | None:
    """Infer a registered grid key from a filename, path, or emulator name."""
    if not value:
        return None
    for grid_name in sorted(GRID_REGISTRY, key=len, reverse=True):
        if str(value).startswith(grid_name) or grid_name in str(value):
            return grid_name
    return None


def grid_type(value: str | None) -> str | None:
    """Return the compact grid family key, such as ``cv_bl`` or ``agn``."""
    config = get_grid_config(value)
    return config.get("type") if config else None


def _grid_config_or_default(grid_name: str | None) -> Dict[str, Any]:
    """Return a registry entry, keeping missing grid names as legacy CV no-BL."""
    config = get_grid_config(grid_name)
    if config is not None:
        return config
    if not grid_name:
        return GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    raise ValueError(f"Unknown grid: {grid_name}")


def parameter_points(grid_name: str, param_id: int) -> np.ndarray:
    """Return emulator-space grid points for one parameter axis."""
    config = get_grid_config(grid_name)
    if config is None:
        raise KeyError(f"Unknown grid: {grid_name}")
    return np.asarray(config["points"][int(param_id)], dtype=float)


def selected_points(grid_name: str, param_ids: Iterable[int]) -> list[np.ndarray]:
    """Return emulator-space grid points for a selected parameter tuple."""
    return [parameter_points(grid_name, int(param_id)) for param_id in param_ids]


def default_grid_value(values: Sequence[float]) -> float:
    """Choose the default value for an axis fixed outside the emulator.

    Three-point and denser axes use the middle value.  Two-point axes use the
    higher value (index 1), matching the AGN-grid requirement where no midpoint
    exists.
    """
    values_array = np.asarray(values, dtype=float)
    if len(values_array) == 0:
        raise ValueError("Cannot choose a default from an empty grid axis")
    if len(values_array) == 2:
        return float(values_array[1])
    return float(values_array[len(values_array) // 2])


def default_parameter_value(grid_name: str, param_id: int) -> float:
    """Return the registry default for one parameter axis."""
    return default_grid_value(parameter_points(grid_name, param_id))


def inclination_values(grid_name: str | None, param_id: int | None = None) -> list[int]:
    """Return supported inclination angles for a grid or inclination axis.

    Without ``param_id`` this returns the raw .spec flux-column angles.  With an
    inclination parameter ID it returns that trainable axis, e.g. sparse AGN
    inclinations for param9.
    """
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    if param_id is not None and int(param_id) in config["points"]:
        return [int(round(value)) for value in np.asarray(config["points"][int(param_id)], dtype=float)]
    return [int(value) for value in config["inclination_columns"]]


def inclination_column(grid_name: str | None, angle: float) -> int:
    """Map a physical inclination angle to its .spec flux column index."""
    values = inclination_values(grid_name)
    angle_int = int(round(float(angle)))
    if angle_int not in values:
        raise ValueError(f"Inclination {angle:g} is not available for {grid_name}: {values}")
    # +2 skips the frequency and wavelength columns at the front of every .spec file.
    return 2 + values.index(angle_int)


def inclination_to_usecols(grid_name: str | None, base_usecols: Sequence[int], angle: float) -> tuple[int, int]:
    """Return ``usecols`` with the flux column replaced by an inclination angle."""
    return int(base_usecols[0]), inclination_column(grid_name, angle)


def has_trainable_inclination(grid_name: str | None, param_values: Iterable[int] | None) -> bool:
    """Return True when the selected emulator axes include inclination."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    selected = {int(param_value) for param_value in (param_values or [])}
    return bool(selected & set(config["inclination_param_ids"]))


def default_fixed_inclination(grid_name: str | None) -> int:
    """Return the grid-specific observer angle used when inclination is fixed."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    angle = int(config["default_fixed_inclination"])
    if angle not in inclination_values(config["name"]):
        raise ValueError(
            f"Default fixed inclination {angle} is not available for {config['name']}"
        )
    return angle


def parameter_label_map(grid_name: str | None) -> Mapping[int, str]:
    """Return short display labels keyed by parameter ID."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    return config["param_labels"]


def parameter_description_map(grid_name: str | None) -> Mapping[int, str]:
    """Return longer GridInterface parameter descriptions keyed by ID."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    return config["param_descriptions"]


def benchmark_param_map(grid_name: str | None) -> Mapping[int, tuple[str, str]]:
    """Return friendly report labels and Sirocco .pf keys keyed by ID."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    return config["benchmark_map"]


def log_param_ids(grid_name: str | None) -> set[int]:
    """Return parameter IDs stored as log10 coordinates for a grid."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    return set(config["log_param_ids"])


def _param_id_from_name(param_name: str) -> int:
    """Extract the numeric ID from labels such as ``param1`` or ``param10``."""
    match = re.search(r"\d+", str(param_name))
    if match is None:
        raise ValueError(f"Cannot extract parameter ID from {param_name!r}")
    return int(match.group(0))


def defaulted_physical_param_ids(grid_name: str | None, param_names: Sequence[str]) -> list[int]:
    """Return physical parameter IDs that will be filled from registry defaults."""
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    selected = {
        _param_id_from_name(param_name)
        for param_name in param_names
    }
    return [
        int(param_id)
        for param_id in config["physical_param_ids"]
        if int(param_id) not in selected
    ]


def param_map_db(grid_name: str | None) -> dict[int, tuple[str, float, float, bool]]:
    """Return notebook-friendly parameter metadata tuples.

    The marimo apps historically used ``(label, min, max, is_log10)`` records.
    This adapter preserves that interface while sourcing the values from the
    registry.
    """
    config = get_grid_config(grid_name) or GRID_REGISTRY["speculate_cv_no-bl_grid_v87f"]
    labels = config["param_labels"]
    log_ids = set(config["log_param_ids"])
    out = {}
    for param_id, values in config["points"].items():
        values_array = np.asarray(values, dtype=float)
        out[int(param_id)] = (
            labels.get(int(param_id), f"param{param_id}"),
            float(values_array.min()),
            float(values_array.max()),
            int(param_id) in log_ids,
        )
    return out


def format_param_tag(param_ids: Iterable[int]) -> str:
    """Encode parameter IDs using Speculate's original concatenated filename tag.

    Tags are written from sorted, progressively increasing IDs, so ``12391011``
    means params 1, 2, 3, 9, 10, and 11.  The reader uses that increasing-order
    contract to disambiguate two-digit IDs while preserving old filenames.
    """
    ids = sorted(int(param_id) for param_id in param_ids)
    return "".join(str(param_id) for param_id in ids)


def parse_param_tag(tag: str) -> list[int]:
    """Decode a progressively increasing concatenated parameter tag."""
    text = str(tag)
    max_param_id = max(
        int(param_id)
        for config in GRID_REGISTRY.values()
        for param_id in config["points"]
    )

    def _parse_from(position: int, previous: int) -> list[int] | None:
        """Backtracking parser for sorted legacy tags such as ``1231011``."""
        if position == len(text):
            return []
        for width in (1, 2):
            token = text[position:position + width]
            if len(token) != width:
                continue
            if not token.isdigit():
                continue
            value = int(token)
            if value < 1 or value > max_param_id or value <= previous:
                continue
            rest = _parse_from(position + width, value)
            if rest is not None:
                return [value, *rest]
        return None

    parsed = _parse_from(0, 0)
    if parsed is None:
        raise ValueError(f"Cannot parse parameter tag: {tag}")
    return parsed


def gravitational_radius_cm(mass_msol: float) -> float:
    """Return one gravitational radius, GM/c^2, for an AGN mass in solar masses."""
    return (_GRAV_CONST * float(mass_msol) * _MSOL) / _C**2


def eddington_mdot_msolyr(mass_msol: float) -> float:
    """Return the Eddington accretion rate in solar masses per year."""
    l_edd = _EDD_LUMINOSITY_CONST * float(mass_msol)
    r_isco = 6.0
    eta = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * r_isco))
    mdot_edd_cgs = l_edd / (eta * _C**2)
    return (mdot_edd_cgs / _MSOL) * _YR


def _normalise_lookup_key(key: str) -> str:
    """Return a punctuation-insensitive key for lookup-table columns."""
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def _row_keys(row: Any) -> list[Any]:
    """Return available keys for common pandas/dict/record row objects."""
    if hasattr(row, "index"):
        return list(row.index)
    if isinstance(row, Mapping):
        return list(row.keys())
    dtype = getattr(row, "dtype", None)
    dtype_names = getattr(dtype, "names", None)
    if dtype_names:
        return list(dtype_names)
    return []


def _lookup_row_key(row: Any, key: str) -> Any:
    """Find a lookup-table column using exact, normalised, then fuzzy matching."""
    keys = _row_keys(row)
    if key in keys:
        return key

    target = _normalise_lookup_key(key)
    normalised_matches = [candidate for candidate in keys if _normalise_lookup_key(candidate) == target]
    if len(normalised_matches) == 1:
        return normalised_matches[0]

    fuzzy_matches = [
        candidate for candidate in keys
        if target in _normalise_lookup_key(candidate) or _normalise_lookup_key(candidate) in target
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]

    raise KeyError(key)


def _row_value(row: Any, key: str) -> float:
    """Read a scalar value from a pandas row-like object as float."""
    return float(row[_lookup_row_key(row, key)])


def lookup_row_to_emulator_values(grid_name: str | None, row: Any, inclination: float | None = None) -> Dict[str, float]:
    """Convert a lookup-table row from physical Sirocco units to emulator labels.

    Test-grid lookup tables store physical simulation inputs.  Inference and
    Tier 2 compare against emulator-space values, so this function applies the
    same forward transforms used by each grid interface.
    """
    config = _grid_config_or_default(grid_name)
    grid_kind = config["type"]
    if grid_kind == "agn":
        # AGN lookup rows are physical: mass in Msol, disk/wind mdot in Msol/yr,
        # KWD acceleration length in cm.  The emulator stores mass, Eddington
        # fraction, wind/disk ratio, filling factor, KWD.d, and R_g-scaled
        # acceleration length on log10 axes.
        mass = _row_value(row, "Central_object.mass(msol)")
        disk_mdot = _row_value(row, "Disk.mdot(msol/yr)")
        wind_mdot = _row_value(row, "Wind.mdot(msol/yr)")
        filling_factor = _row_value(row, "Wind.filling_factor(1=smooth,<1=clumped)")
        kwd_d = _row_value(row, "KWD.d(in_units_of_rstar)")
        acceleration_length = _row_value(row, "KWD.acceleration_length(cm)")
        rg = gravitational_radius_cm(mass)
        values = {
            "Central_object.mass": np.log10(mass),
            "Disk.mdot / Mdot_Edd": np.log10(disk_mdot / eddington_mdot_msolyr(mass)),
            "Wind.mdot / Disk.mdot": np.log10(wind_mdot / disk_mdot),
            "Wind.filling_factor": np.log10(filling_factor),
            "KWD.d": np.log10(kwd_d),
            "KWD.mdot_r_exponent": _row_value(row, "KWD.mdot_r_exponent"),
            "KWD.acceleration_length / R_g": np.log10(acceleration_length / rg),
            "KWD.acceleration_exponent": _row_value(row, "KWD.acceleration_exponent"),
        }
    elif grid_kind in {"cv_no-bl", "cv_bl"}:
        # CV lookup rows store disk and wind mass-loss rates separately; the
        # emulator uses log10(disk.mdot) and the wind/disk ratio.
        disk_mdot = _row_value(row, "Disk.mdot(msol/yr)")
        wind_mdot = _row_value(row, "Wind.mdot(msol/yr)")
        values = {
            "disk.mdot": np.log10(disk_mdot),
            "wind.mdot": wind_mdot / disk_mdot,
            "KWD.d": np.log10(_row_value(row, "KWD.d(in_units_of_rstar)")),
            "KWD.mdot_r_exponent": _row_value(row, "KWD.mdot_r_exponent"),
            "KWD.acceleration_length": np.log10(_row_value(row, "KWD.acceleration_length(cm)")),
            "KWD.acceleration_exponent": _row_value(row, "KWD.acceleration_exponent"),
        }
        if grid_kind == "cv_bl":
            for friendly, key in (
                ("Boundary_layer.luminosity", "Boundary_layer.luminosity(ergs/s)"),
                ("Boundary_layer.temp", "Boundary_layer.temp(K)"),
            ):
                try:
                    values[friendly] = _row_value(row, key)
                except KeyError:
                    pass
    else:
        raise ValueError(f"Grid type {grid_kind!r} is not supported for lookup conversion")

    if inclination is not None:
        values["Inclination"] = float(inclination)
    return values


def emulator_values_to_physical(grid_name: str | None, param_names: Sequence[str], values: np.ndarray) -> dict[str, float]:
    """Convert fitted emulator coordinates into physical Sirocco .pf values.

    Lower-dimensional emulators omit some physical axes.  Those omitted axes are
    filled from registry defaults before conversion so Tier 3 can still write a
    complete Sirocco template.
    """
    config = _grid_config_or_default(grid_name)
    param_map = benchmark_param_map(config["name"])
    vals = np.atleast_1d(values)

    complete_values: dict[int, float] = {
        int(param_id): default_parameter_value(config["name"], int(param_id))
        for param_id in config["physical_param_ids"]
    }
    for pn, value in zip(param_names, vals):
        # Supplied emulator axes override the defaults one by one; any omitted
        # physical axis remains at the grid-specific default.
        idx = _param_id_from_name(pn)
        complete_values[idx] = float(value)

    physical: dict[str, float] = {}
    if config["type"] == "agn":
        # Invert the AGN scaling used by lookup_row_to_emulator_values():
        # mass and several ratios are log10 axes, disk.mdot is stored as an
        # Eddington fraction, and acceleration length is stored in R_g units.
        mass = 10 ** complete_values[1]
        disk_edd_fraction = 10 ** complete_values[2]
        disk_mdot = disk_edd_fraction * eddington_mdot_msolyr(mass)
        wind_ratio = 10 ** complete_values[3]
        rg = gravitational_radius_cm(mass)
        converted = {
            1: mass,
            2: disk_mdot,
            3: wind_ratio * disk_mdot,
            4: 10 ** complete_values[4],
            5: 10 ** complete_values[5],
            6: complete_values[6],
            7: (10 ** complete_values[7]) * rg,
            8: complete_values[8],
        }
    elif config["type"] in {"cv_no-bl", "cv_bl"}:
        # CV grids use log10 for disk.mdot, KWD.d, and acceleration length.
        # Wind.mdot is stored as a ratio until disk.mdot has been recovered.
        cv_log_ids = set(config["log_param_ids"])
        converted = {}
        for param_id, value in complete_values.items():
            if param_id in cv_log_ids:
                converted[param_id] = 10 ** value
            else:
                converted[param_id] = value
        if 1 in converted and 2 in converted:
            converted[2] = converted[2] * converted[1]
    else:
        raise ValueError(f"Grid type {config['type']!r} is not supported for physical conversion")

    for param_id, value in converted.items():
        _, sirocco_key = param_map.get(param_id, (f"param{param_id}", f"param{param_id}"))
        physical[sirocco_key] = float(value)

    for pn, value in zip(param_names, vals):
        idx = _param_id_from_name(pn)
        if idx in config["inclination_param_ids"]:
            # Inclination is not part of physical_param_ids because it does not
            # help choose a run file, but Tier 3 needs it for observer angles.
            _, sirocco_key = param_map.get(idx, ("Inclination", "Inclination"))
            physical[sirocco_key] = float(value)

    return physical
