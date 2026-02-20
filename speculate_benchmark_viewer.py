# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Benchmark Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    return (mo,)

@app.cell(hide_code=True)
def _():
    logo_path = "assets/logos/Speculate_logo2.png"

    # Left column: Title and Description
    title_col = mo.vstack([
        mo.md("# Benchmark Suite"),
        mo.md("Evaluate emulator performance across three tiers: "
               "grid reconstruction, parameter recovery, and observational spectra."),
    ])
    # Right column: Logo with link
    # Using flex-end to align it to the right
    logo_col = mo.vstack([
        mo.image(src=logo_path, width=400),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    # Combine in horizontal stack
    mo.hstack([title_col, logo_col], justify="space-between", align="center")
    return (mo,)

@app.cell
def _(mo):
    # Static sidebar - always shows all options
    mo.sidebar(
        mo.vstack([
                    mo.md("# 🔭 Speculate"),
                    mo.md(" "),
                    mo.md(" "),
                    mo.md("---"),
                    mo.md("---"),
                    mo.md(" "),
                    mo.md(" "),
                    mo.nav_menu({
                        "/": f"###{mo.icon('lucide:home')} Home",
                    }, orientation="vertical"),
                    mo.md(" "),
                    mo.md("---"),
                    mo.md("---"),
                    mo.nav_menu({
                    "https://github.com/sirocco-rt/speculate": f"###{mo.icon('lucide:github')} Speculate Github",
                    "https://github.com/sirocco-rt/speculate/wiki": f"###{mo.icon('lucide:book-open')} Speculate Docs",
                    }, orientation="vertical"),
                    mo.md(" "),
                    mo.md("---"),
                    mo.nav_menu({
                    "https://github.com/sirocco-rt/sirocco": f"###{mo.icon('lucide:wind')} Sirocco Github",
                    "https://sirocco-rt.readthedocs.io/en/latest/": f"###{mo.icon('lucide:wind')} Sirocco Docs"
                    }, orientation="vertical")
                ])
    )
    return

@app.cell(hide_code=True)
def _():
    import json
    import os
    import re
    import glob
    import time
    import numpy as np
    import pandas as pd
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return alt, glob, json, np, os, pd, plt, re, time


@app.cell
def _(mo):
    get_report, set_report = mo.state(None)
    get_tier1_arrays, set_tier1_arrays = mo.state(None)
    get_comparison_reports, set_comparison_reports = mo.state([])
    get_status_msg, set_status_msg = mo.state("")
    return (
        get_comparison_reports,
        get_report,
        get_status_msg,
        get_tier1_arrays,
        set_comparison_reports,
        set_report,
        set_status_msg,
        set_tier1_arrays,
    )


@app.cell(hide_code=True)
def _(glob, mo, os):
    # Discover existing reports
    _report_dir = "benchmark_results"
    _existing = sorted(glob.glob(os.path.join(_report_dir, "benchmark_report*.json")), reverse=True)
    _existing_labels = [os.path.basename(_f) for _f in _existing]

    report_picker = mo.ui.dropdown(
        options=dict(zip(_existing_labels, _existing)) if _existing else {"(no reports found)": ""},
        label="Select a report",
    )

    upload_btn = mo.ui.file(
        filetypes=[".json"],
        label="Upload JSON report",
        kind="button",
    )

    compare_picker = mo.ui.multiselect(
        options=dict(zip(_existing_labels, _existing)) if _existing else {},
        label="Select reports to compare",
    )

    load_btn = mo.ui.run_button(label="Load")
    compare_btn = mo.ui.run_button(label="Load Comparison")

    mo.vstack([
        mo.md("### Load Benchmark Report"),
        mo.hstack([report_picker, load_btn], gap=1),
        mo.hstack([upload_btn], gap=1),
        mo.md("---"),
        mo.md("### Compare Reports"),
        mo.hstack([compare_picker, compare_btn], gap=1),
    ])
    return compare_btn, compare_picker, load_btn, report_picker, upload_btn


@app.cell
def _(
    compare_btn,
    compare_picker,
    json,
    load_btn,
    os,
    report_picker,
    set_comparison_reports,
    set_report,
    set_tier1_arrays,
    upload_btn,
):
    # Trigger side-effects on button clicks
    if load_btn.value:
        _path = report_picker.value
        if _path and os.path.isfile(_path):
            with open(_path) as _f:
                set_report(json.load(_f))
            # Loaded-from-file reports have no in-memory arrays
            set_tier1_arrays(None)

    if upload_btn.value:
        for _f in upload_btn.value:
            set_report(json.loads(_f.contents.decode()))
            set_tier1_arrays(None)
            break

    if compare_btn.value:
        _reports = []
        for _path in (compare_picker.value or []):
            if os.path.isfile(_path):
                with open(_path) as _fh:
                    _reports.append(json.load(_fh))
        set_comparison_reports(_reports)
    return


@app.cell(hide_code=True)
def _(get_report, mo, np, plt):
    _report = get_report()

    if _report is None:
        mo.output.replace(mo.md("*Load a report above, or run a benchmark below.*"))
        mo.stop(True)

    # Build tab content
    _tabs = {}

    # ---- TIER 1 TAB ----
    _t1 = _report.get("tier1")
    if _t1:
        _t1_items = []
        _t1_items.append(mo.md("## Tier 1 — Grid Reconstruction Fidelity"))

        # Summary metrics
        _summary_rows = []
        for _key in ["n_components", "n_grid_points", "n_params", "tier1_time_s",
                      "pca_explained_variance", "loo_flux_rmse_median",
                      "loo_flux_rmse_95", "max_fractional_resid"]:
            if _key in _t1:
                _val = _t1[_key]
                if isinstance(_val, float):
                    _summary_rows.append({"Metric": _key, "Value": f"{_val:.6g}"})
                else:
                    _summary_rows.append({"Metric": _key, "Value": str(_val)})

        if _summary_rows:
            _t1_items.append(mo.ui.table(_summary_rows, label="Summary"))

        # Per-component RMSE
        _rmses = _t1.get("loo_rmse_per_comp", [])
        if _rmses:
            _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
            _axes[0].bar(range(len(_rmses)), _rmses, color="steelblue")
            _axes[0].set_xlabel("PCA Component")
            _axes[0].set_ylabel("LOO RMSE")
            _axes[0].set_title("LOO RMSE by Component")

            # Standardised residuals
            _std_resid = _t1.get("loo_std_resid", [])
            if _std_resid:
                _flat = np.array(_std_resid).flatten()
                _axes[1].hist(_flat, bins=50, density=True, alpha=0.7, color="steelblue", label="LOO")
                _xgauss = np.linspace(-4, 4, 200)
                _axes[1].plot(_xgauss, 1.0/(np.sqrt(2*np.pi))*np.exp(-0.5*_xgauss**2), "r-", label="N(0,1)")
                _axes[1].set_xlabel("Standardised Residual")
                _axes[1].set_title("LOO Residual Distribution")
                _axes[1].legend()

            plt.tight_layout()
            _t1_items.append(mo.as_html(_fig))
            plt.close()

        _tabs["Tier 1"] = mo.vstack(_t1_items)

    # ---- TIER 2 TAB ----
    _t2 = _report.get("tier2")
    if _t2:
        _t2_items = []
        _t2_items.append(mo.md("## Tier 2 — Test Grid Parameter Recovery"))
        _t2_items.append(mo.md(
            f"**{_t2.get('n_processed', '?')}/{_t2.get('n_spectra', '?')} spectra** "
            f"processed ({_t2.get('n_failures', 0)} failures) in "
            f"{_t2.get('tier2_time_s', 0):.0f}s"
        ))

        _agg = _t2.get("aggregate", {})
        if _agg:
            _agg_rows = []
            for _pn, _m in _agg.items():
                _agg_rows.append({
                    "Parameter": _pn,
                    "RMSE": f"{_m.get('rmse', float('nan')):.4f}",
                    "Bias": f"{_m.get('bias', float('nan')):.4f}",
                    "CRPS": f"{_m.get('crps', float('nan')):.4f}",
                    "Shrinkage": f"{_m.get('shrinkage', float('nan')):.2%}",
                    "Cov@68%": f"{_m.get('coverage_68', float('nan')):.2f}",
                    "Cov@95%": f"{_m.get('coverage_95', float('nan')):.2f}",
                })
            _t2_items.append(mo.ui.table(_agg_rows, label="Aggregate Metrics"))

            _fig2, _ax2 = plt.subplots(figsize=(6, 6))
            _ax2.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal")
            for _pn, _m in _agg.items():
                if "coverage_alphas" in _m:
                    _ax2.plot(_m["coverage_alphas"], _m["coverage_values"], label=_pn)
            _ax2.set_xlabel("Nominal Credible Level")
            _ax2.set_ylabel("Empirical Coverage")
            _ax2.set_title("PP-Plot (Calibration)")
            _ax2.legend(fontsize=8)
            _ax2.set_xlim(0, 1)
            _ax2.set_ylim(0, 1)
            plt.tight_layout()
            _t2_items.append(mo.as_html(_fig2))
            plt.close()

            _param_names = list(_agg.keys())
            _fig3, _axes3 = plt.subplots(1, 2, figsize=(12, 4))
            _xp = np.arange(len(_param_names))
            _axes3[0].bar(_xp, [_agg[_p]["rmse"] for _p in _param_names], color="steelblue")
            _axes3[0].set_xticks(_xp)
            _axes3[0].set_xticklabels(_param_names, rotation=45, ha="right", fontsize=8)
            _axes3[0].set_ylabel("RMSE")
            _axes3[0].set_title("Parameter RMSE")
            _axes3[1].bar(_xp, [_agg[_p]["bias"] for _p in _param_names], color="salmon")
            _axes3[1].axhline(0, color="k", lw=0.5)
            _axes3[1].set_xticks(_xp)
            _axes3[1].set_xticklabels(_param_names, rotation=45, ha="right", fontsize=8)
            _axes3[1].set_ylabel("Bias")
            _axes3[1].set_title("Parameter Bias")
            plt.tight_layout()
            _t2_items.append(mo.as_html(_fig3))
            plt.close()

        _tabs["Tier 2"] = mo.vstack(_t2_items)

    # ---- TIER 3 TAB ----
    _t3 = _report.get("tier3")
    if _t3:
        _t3_items = []
        _t3_items.append(mo.md("## Tier 3 — Observational Spectra"))

        _obs_rows = []
        for _r in _t3:
            _obs_rows.append({
                "Observation": _r.get("obs_file", "?"),
                "Reduced chi2": f"{_r.get('reduced_chi2', float('nan')):.2f}",
                "PPC Coverage": f"{_r.get('ppc_coverage', float('nan')):.2f}",
                "Converged": "yes" if _r.get("mcmc_converged") else "no",
            })
        _t3_items.append(mo.ui.table(_obs_rows, label="Observational Results"))

        try:
            _obs_names = [_r.get("obs_file", f"#{_i}") for _i, _r in enumerate(_t3)]
            _chi2s = [_r.get("reduced_chi2", float("nan")) for _r in _t3]
            _ppcs = [_r.get("ppc_coverage", float("nan")) for _r in _t3]

            _fig4, _axes4 = plt.subplots(1, 2, figsize=(max(10, len(_obs_names)*2), 4))
            _xr = np.arange(len(_obs_names))
            _axes4[0].bar(_xr, _chi2s, color="steelblue")
            _axes4[0].axhline(1.0, color="r", lw=1, ls="--")
            _axes4[0].set_xticks(_xr)
            _axes4[0].set_xticklabels(_obs_names, rotation=45, ha="right", fontsize=8)
            _axes4[0].set_ylabel("Reduced chi2")
            _axes4[0].set_title("Reduced chi2")
            _axes4[1].bar(_xr, _ppcs, color="mediumseagreen")
            _axes4[1].axhline(0.95, color="r", lw=1, ls="--")
            _axes4[1].set_xticks(_xr)
            _axes4[1].set_xticklabels(_obs_names, rotation=45, ha="right", fontsize=8)
            _axes4[1].set_ylabel("PPC Coverage")
            _axes4[1].set_title("Posterior Predictive Coverage")
            plt.tight_layout()
            _t3_items.append(mo.as_html(_fig4))
            plt.close()
        except Exception:
            pass

        _tabs["Tier 3"] = mo.vstack(_t3_items)

    # ---- CONFIG TAB ----
    _cfg = _report.get("config")
    if _cfg:
        _cfg_items = [mo.md("## Run Configuration")]
        _cfg_rows = [{"Key": _k, "Value": str(_v)} for _k, _v in _cfg.items()]
        _cfg_items.append(mo.ui.table(_cfg_rows, label="Configuration"))
        _cfg_items.append(mo.md(
            f"**Benchmark version:** {_report.get('speculate_benchmark_version', '?')}  \n"
            f"**Timestamp:** {_report.get('timestamp', '?')}"
        ))
        _tabs["Config"] = mo.vstack(_cfg_items)

    if not _tabs:
        mo.output.replace(mo.md("*Report loaded but contains no tier data.*"))
        mo.stop(True)

    mo.ui.tabs(_tabs)
    return


@app.cell(hide_code=True)
def _(get_tier1_arrays, mo):
    _arrs = get_tier1_arrays()
    if _arrs is None:
        mo.output.replace(mo.md(
            "*Run a Tier 1 benchmark below to view the interactive reconstruction plot.*"
        ))
        mo.stop(True)

    _M = _arrs["original_flux"].shape[0]
    _pnames = _arrs["param_names"]

    spectrum_slider = mo.ui.slider(
        start=0, stop=_M - 1, value=0, step=1, show_value=True,
        label="Grid point index",
    )
    recon_n_grid = _M
    recon_param_names = _pnames

    mo.vstack([
        mo.md("## Interactive Reconstruction Explorer"),
        mo.md(f"Browse all **{_M}** training spectra. "
               "Overlay original grid spectrum, PCA reconstruction, and LOO GP reconstruction."),
        spectrum_slider,
    ])
    return recon_param_names, spectrum_slider


@app.cell(hide_code=True)
def _(alt, get_tier1_arrays, mo, np, pd, recon_param_names, spectrum_slider):
    _arrs = get_tier1_arrays()
    _idx = spectrum_slider.value

    _wl = _arrs["wavelength"]
    _orig = _arrs["original_flux"][_idx]
    _pca = _arrs["pca_recon_flux"][_idx]
    _loo = _arrs["loo_recon_flux"][_idx]
    _gp = _arrs["grid_points"][_idx]
    _pnames = recon_param_names

    # --- Readout: parameter values and metrics ---
    _param_badges = " | ".join(
        f"**{_pnames[_j]}** = {_gp[_j]:.4g}" for _j in range(len(_pnames))
    )
    _rmse_loo = float(np.sqrt(np.mean((_orig - _loo) ** 2)))
    _rmse_pca = float(np.sqrt(np.mean((_orig - _pca) ** 2)))
    _max_frac = float(np.nanmax(np.abs(_orig - _loo) / (np.abs(_orig) + 1e-30)))

    _readout = mo.md(
        f"{_param_badges}  \n"
        f"LOO RMSE: **{_rmse_loo:.6g}** &nbsp; PCA RMSE: **{_rmse_pca:.6g}** "
        f"&nbsp; Max fractional residual: **{_max_frac:.4g}**"
    )

    # --- Altair overlay chart ---
    _df_spec = pd.DataFrame({
        "Wavelength": np.tile(_wl, 3),
        "Flux": np.concatenate([_orig, _pca, _loo]),
        "Series": (
            ["Original"] * len(_wl) +
            ["PCA Recon"] * len(_wl) +
            ["LOO GP Recon"] * len(_wl)
        ),
    })

    _color_scale = alt.Scale(
        domain=["Original", "PCA Recon", "LOO GP Recon"],
        range=["#4c78a8", "#f58518", "#54a24b"],
    )
    _dash_scale = alt.Scale(
        domain=["Original", "PCA Recon", "LOO GP Recon"],
        range=[[0, 0], [6, 4], [4, 2]],
    )

    _spec_chart = (
        alt.Chart(_df_spec)
        .mark_line(strokeWidth=1.5, opacity=0.85)
        .encode(
            x=alt.X("Wavelength:Q", title="Wavelength (AA)"),
            y=alt.Y("Flux:Q", title="Normalised Flux"),
            color=alt.Color("Series:N", scale=_color_scale, legend=alt.Legend(title="Series")),
            strokeDash=alt.StrokeDash("Series:N", scale=_dash_scale, legend=None),
            tooltip=["Wavelength:Q", "Flux:Q", "Series:N"],
        )
        .properties(width="container", height=350, title=f"Spectrum #{_idx}")
        .interactive()
    )

    # --- Residual panel ---
    _resid_loo = _orig - _loo
    _resid_pca = _orig - _pca
    _df_resid = pd.DataFrame({
        "Wavelength": np.tile(_wl, 2),
        "Residual": np.concatenate([_resid_loo, _resid_pca]),
        "Series": ["LOO GP Residual"] * len(_wl) + ["PCA Residual"] * len(_wl),
    })

    _resid_color = alt.Scale(
        domain=["LOO GP Residual", "PCA Residual"],
        range=["#54a24b", "#f58518"],
    )

    _zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.5)
        .encode(y="y:Q")
    )

    _resid_chart = (
        alt.Chart(_df_resid)
        .mark_line(strokeWidth=1, opacity=0.7)
        .encode(
            x=alt.X("Wavelength:Q", title="Wavelength (AA)"),
            y=alt.Y("Residual:Q", title="Residual (Original - Recon)"),
            color=alt.Color("Series:N", scale=_resid_color, legend=alt.Legend(title="")),
            tooltip=["Wavelength:Q", "Residual:Q", "Series:N"],
        )
        .properties(width="container", height=180)
        .interactive()
    )

    _combined_resid = _zero_rule + _resid_chart

    display = mo.vstack([
        _readout,
        _spec_chart,
        mo.md("#### Residuals"),
        _combined_resid,
    ])
    display
    return


@app.cell(hide_code=True)
def _(get_tier1_arrays, mo, np):
    _arrs = get_tier1_arrays()
    if _arrs is None:
        mo.stop(True)

    _gp = _arrs["grid_points"]
    _pnames = _arrs["param_names"]
    _orig = _arrs["original_flux"]
    _pca = _arrs["pca_recon_flux"]
    _loo = _arrs["loo_recon_flux"]
    _M = _gp.shape[0]

    # Compute per-spectrum metrics
    _loo_rmse = np.sqrt(np.mean((_orig - _loo) ** 2, axis=1))
    _pca_rmse = np.sqrt(np.mean((_orig - _pca) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        _max_frac = np.nanmax(
            np.abs(_orig - _loo) / (np.abs(_orig) + 1e-30), axis=1
        )

    # Sort by descending LOO RMSE
    _order = np.argsort(-_loo_rmse)

    _rows = []
    for _rank, _i in enumerate(_order):
        _row = {"Rank": _rank + 1, "Index": int(_i)}
        for _j, _pn in enumerate(_pnames):
            _row[_pn] = round(float(_gp[_i, _j]), 4)
        _row["LOO RMSE"] = f"{_loo_rmse[_i]:.6g}"
        _row["PCA RMSE"] = f"{_pca_rmse[_i]:.6g}"
        _row["Max Frac Resid"] = f"{_max_frac[_i]:.4g}"
        _rows.append(_row)

    mo.vstack([
        mo.md("#### Worst-Reconstructed Spectra"),
        mo.md("Sorted by descending LOO RMSE. Use the slider above to inspect any spectrum."),
        mo.ui.table(_rows, label="All spectra", page_size=15),
    ])
    return


@app.cell(hide_code=True)
def _(get_comparison_reports, mo, np, plt):
    _comp_reports = get_comparison_reports()
    if not _comp_reports or len(_comp_reports) < 2:
        mo.output.replace(mo.md("*Select 2+ reports above to compare.*"))
        mo.stop(True)

    _comp_items = [mo.md("## Report Comparison")]

    _t1_data = {}
    for _i, _rep in enumerate(_comp_reports):
        _tag = _rep.get("config", {}).get("tag", f"report_{_i}")
        _t1 = _rep.get("tier1", {})
        if "loo_flux_rmse_median" in _t1:
            _t1_data[_tag] = _t1["loo_flux_rmse_median"]

    if _t1_data:
        _rows = [{"Report": _k, "Tier 1 LOO Flux RMSE (median)": f"{_v:.6g}"} for _k, _v in _t1_data.items()]
        _comp_items.append(mo.md("### Tier 1 Comparison"))
        _comp_items.append(mo.ui.table(_rows))

    _all_params = set()
    for _rep in _comp_reports:
        _agg = _rep.get("tier2", {}).get("aggregate", {})
        _all_params.update(_agg.keys())

    if _all_params:
        _comp_items.append(mo.md("### Tier 2 Parameter RMSE Comparison"))
        _params = sorted(_all_params)
        _fig, _ax = plt.subplots(figsize=(max(10, len(_params)*2), 5))
        _bar_width = 0.8 / len(_comp_reports)
        for _i, _rep in enumerate(_comp_reports):
            _tag = _rep.get("config", {}).get("tag", f"report_{_i}")
            _agg = _rep.get("tier2", {}).get("aggregate", {})
            _rmses = [_agg.get(_p, {}).get("rmse", float("nan")) for _p in _params]
            _xb = np.arange(len(_params)) + _i * _bar_width
            _ax.bar(_xb, _rmses, _bar_width, label=_tag)
        _ax.set_xticks(np.arange(len(_params)) + _bar_width * (len(_comp_reports)-1)/2)
        _ax.set_xticklabels(_params, rotation=45, ha="right", fontsize=8)
        _ax.set_ylabel("RMSE")
        _ax.legend()
        _ax.set_title("Tier 2 RMSE — Side by Side")
        plt.tight_layout()
        _comp_items.append(mo.as_html(_fig))
        plt.close()

    mo.vstack(_comp_items)
    return


@app.cell(hide_code=True)
def _(glob, mo, os):
    mo.md("---")

    # Discover files
    emu_files = sorted(glob.glob("Grid-Emulator_Files/*emu*.npz"))
    obs_csvs = sorted(glob.glob("observation_files/*.csv"))

    emu_picker = mo.ui.dropdown(
        options=dict(zip([os.path.basename(_f) for _f in emu_files], emu_files)) if emu_files else {"(none)": ""},
        label="Emulator",
    )
    obs_picker = mo.ui.multiselect(
        options=dict(zip([os.path.basename(_f) for _f in obs_csvs], obs_csvs)) if obs_csvs else {},
        label="Observations (Tier 3)",
    )

    tier_picker = mo.ui.multiselect(
        options={"Tier 1": 1, "Tier 2": 2, "Tier 3": 3},
        label="Tiers to run",
        value=["Tier 1"],
    )

    max_spectra_slider = mo.ui.slider(
        start=1, stop=50, value=5, step=1, show_value=True,
        label="Max test spectra (Tier 2)",
    )
    mcmc_steps_slider = mo.ui.slider(
        start=100, stop=5000, value=1000, step=100, show_value=True,
        label="MCMC steps",
    )
    return (
        emu_picker,
        max_spectra_slider,
        mcmc_steps_slider,
        obs_picker,
        tier_picker,
    )


@app.cell(hide_code=True)
def _(emu_picker, glob, mo, os, re):
    """Auto-populate grid and test-grid paths from the emulator selection."""
    _emu_val = emu_picker.value or ""
    _emu_base = os.path.basename(_emu_val)

    matched_grid_path = ""
    matched_testgrid_path = ""
    emu_grid_info = mo.md("")  # default: empty

    if "_emu_" in _emu_base:
        # Extract grid stem (everything before _emu_)
        _grid_stem = _emu_base.split("_emu_")[0]  # e.g. speculate_cv_bl_grid_v87f

        # Extract param-scale tag: digits + scale + optional "smooth"
        # e.g. from "1234_linear_55inc_850-1850AA_15PCA.npz" → "1234_linear"
        _after_emu = _emu_base.split("_emu_")[1]  # 1234_linear_55inc_850-...
        # Tag is everything before the first segment matching \d+inc
        _tag_match = re.match(r"(.+?)_\d+inc_", _after_emu)
        if _tag_match:
            _param_tag = _tag_match.group(1)
        else:
            # Fallback: try splitting on known suffixes
            _param_tag = _after_emu.split("_850")[0].split("_1000")[0]

        # Find matching grid NPZ
        _grid_pattern = f"Grid-Emulator_Files/{_grid_stem}_grid_{_param_tag}.npz"
        _grid_matches = sorted(glob.glob(_grid_pattern))
        if _grid_matches:
            matched_grid_path = _grid_matches[0]

        # Find matching test grid directory
        _testgrid_stem = _grid_stem.replace("_grid_", "_testgrid_")
        _testgrid_pattern = f"sirocco_grids/{_testgrid_stem}*"
        _tg_matches = sorted(glob.glob(_testgrid_pattern))
        if _tg_matches:
            matched_testgrid_path = _tg_matches[0]

        # Build info display
        _grid_display = os.path.basename(matched_grid_path) if matched_grid_path else "*(not found)*"
        _tg_display = os.path.basename(matched_testgrid_path) if matched_testgrid_path else "*(not found)*"

        emu_grid_info = mo.md(
            f"**Grid (Tier 1):** {_grid_display}  \n"
            f"**Test Grid (Tier 2):** {_tg_display}"
        )
    elif _emu_val:
        emu_grid_info = mo.callout(
            mo.md("Could not parse emulator filename to find matching grid. "
                   "Expected pattern: `{stem}_emu_{params}_{inc}inc_{wl}_{PCA}PCA.npz`"),
            kind="warn",
        )
    return emu_grid_info, matched_grid_path, matched_testgrid_path


@app.cell(hide_code=True)
def _(
    emu_grid_info,
    emu_picker,
    max_spectra_slider,
    mcmc_steps_slider,
    mo,
    obs_picker,
    tier_picker,
):
    mo.vstack([
        mo.md("### Run Benchmark"),
        mo.hstack([emu_picker], gap=1),
        emu_grid_info,
        mo.hstack([obs_picker], gap=1),
        mo.hstack([tier_picker, max_spectra_slider, mcmc_steps_slider], gap=1),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    run_btn = mo.ui.run_button(label="Run Benchmark")
    run_btn
    return (run_btn,)


@app.cell
def _(
    emu_picker,
    matched_grid_path,
    matched_testgrid_path,
    max_spectra_slider,
    mcmc_steps_slider,
    mo,
    obs_picker,
    os,
    run_btn,
    set_report,
    set_status_msg,
    set_tier1_arrays,
    tier_picker,
    time,
):
    if not run_btn.value:
        mo.stop(True)

    set_status_msg("Running benchmark...")
    _t0 = time.time()

    try:
        from Starfish.emulator import Emulator as _Emulator
        from Speculate_addons.speculate_benchmark import (
            run_tier1 as _run_tier1,
            run_tier2 as _run_tier2,
            run_tier3_single as _run_tier3_single,
            build_report_card as _build_report_card,
            save_report as _save_report,
        )

        _emu_path = emu_picker.value
        if not _emu_path or not os.path.isfile(_emu_path):
            set_status_msg("Error: select a valid emulator file.")
            mo.stop(True)

        _emu = _Emulator.load(_emu_path)

        _tiers = [_v for _v in (tier_picker.value or [])]
        _tier1_result = None
        _tier2_result = None
        _tier3_results = None

        if 1 in _tiers and matched_grid_path:
            set_status_msg("Running Tier 1...")
            _tier1_result = _run_tier1(_emu, matched_grid_path)

            # Store flux arrays in state for the interactive plot
            _t1_arrays = _tier1_result.pop("_arrays", None)
            if _t1_arrays is not None:
                set_tier1_arrays(_t1_arrays)

        if 2 in _tiers and matched_testgrid_path:
            set_status_msg("Running Tier 2...")
            _tier2_result = _run_tier2(
                _emu,
                matched_testgrid_path,
                mcmc_steps=mcmc_steps_slider.value,
                max_spectra=max_spectra_slider.value,
            )

        if 3 in _tiers and obs_picker.value:
            set_status_msg("Running Tier 3...")
            _tier3_results = []
            for _obs_path in obs_picker.value:
                _r = _run_tier3_single(
                    _emu, _obs_path,
                    mcmc_steps=mcmc_steps_slider.value,
                )
                _tier3_results.append(_r)

        _config = {
            "emulator": _emu_path,
            "grid": matched_grid_path,
            "test_grid": matched_testgrid_path,
            "tiers": _tiers,
            "mcmc_steps": mcmc_steps_slider.value,
            "max_spectra": max_spectra_slider.value,
        }
        _report = _build_report_card(_tier1_result, _tier2_result, _tier3_results, _config)

        # Auto-save
        os.makedirs("benchmark_results", exist_ok=True)
        _ts = time.strftime("%Y%m%d_%H%M%S")
        _out_path = f"benchmark_results/benchmark_report_live_{_ts}.json"
        _save_report(_report, _out_path)

        set_report(_report)
        _elapsed = time.time() - _t0
        set_status_msg(f"Benchmark complete in {_elapsed:.1f}s — saved to {_out_path}")

    except Exception as _e:
        import traceback as _tb
        set_status_msg(f"Error: {_e}\n```\n{_tb.format_exc()}\n```")
    return


@app.cell(hide_code=True)
def _(get_status_msg, mo):
    _msg = get_status_msg()
    if _msg:
        mo.output.replace(mo.callout(mo.md(_msg), kind="info"))
    return


if __name__ == "__main__":
    app.run()
