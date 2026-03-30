# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
#
# Speculate Benchmark Viewer
# ==========================
# Interactive notebook for viewing, comparing, and running emulator
# benchmark reports produced by ``Speculate_addons/speculate_benchmark.py``.
#
# Features:
#   • Load saved JSON reports and render per-tier result tabs
#     (Tier 1 reconstruction, Tier 2 parameter recovery, Tier 3 observations).
#   • Interactive Tier 1 reconstruction explorer (slider over all grid spectra
#     with original / PCA / LOO overlays and residuals).
#   • Side-by-side comparison of 2+ reports on shared metric axes.
#   • Inline live benchmark runner with nested progress bars for Tier 1–3.

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full", app_title="Speculate Benchmark Viewer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    _logo_path = "assets/logos/Speculate_logo2.png"

    # Left column: Title and Description
    _title_col = mo.vstack([
        mo.md("# Benchmark Suite"),
        mo.md("Evaluate emulator performance across three tiers: "
               "grid reconstruction, parameter recovery, and observational spectra."),
    ])
    # Right column: Logo with link
    # Using flex-end to align it to the right
    _logo_col = mo.vstack([
        mo.image(src=_logo_path, width=400, height=95),
        mo.md('<p style="text-align: center; font-size: 0.8em;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    ], align="center")

    # Combine in horizontal stack
    mo.hstack([_title_col, _logo_col], justify="space-between", align="center")
    return

@app.cell
def _(mo):
    # Static sidebar - always shows all options
    mo.sidebar(
        mo.vstack([
                    mo.md(f"# Speculate {mo.icon('lucide:telescope')}"),
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
    # ── Third-party imports ──
    # VegaFusion is enabled so Altair charts degrade gracefully when
    # the dataset exceeds the default 5 000-row inline limit.
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
    matplotlib.use("Agg")  # non-interactive backend for server environments
    import matplotlib.pyplot as plt
    return alt, glob, json, np, os, pd, plt, re, time


@app.cell
def _(mo):
    # ── Reactive state ──
    # report         – the active JSON report being viewed in the tabs
    # tier1_arrays   – full in-memory flux matrices from a *live* Tier 1 run
    #                  (original, PCA-recon, LOO-recon); not stored in JSON
    # comparison_reports – list of JSON payloads for the side-by-side view
    # status_msg     – latest benchmark runner status text
    get_report, set_report = mo.state(None)
    get_tier1_arrays, set_tier1_arrays = mo.state(None)
    get_comparison_reports, set_comparison_reports = mo.state([])
    get_status_msg, set_status_msg = mo.state("")
    get_emulator, set_emulator = mo.state(None)
    return (
        get_comparison_reports,
        get_emulator,
        get_report,
        get_status_msg,
        get_tier1_arrays,
        set_comparison_reports,
        set_emulator,
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
    # Each button mutates notebook state only when clicked; the returned state is
    # then consumed by the rendering cells below.
    if load_btn.value:
        _path = report_picker.value
        if _path and os.path.isfile(_path):
            with open(_path) as _f:
                set_report(json.load(_f))
            # Reports loaded from disk contain summaries only, so clear any stale
            # Tier 1 arrays left behind by a previous live benchmark run.
            set_tier1_arrays(None)

    if upload_btn.value:
        # Accept the first uploaded JSON file and treat it the same way as a
        # report loaded from the local picker.
        for _f in upload_btn.value:
            set_report(json.loads(_f.contents.decode()))
            set_tier1_arrays(None)
            break

    if compare_btn.value:
        # Comparison works on multiple independent report payloads rather than
        # the single active report shown in the main tabs.
        _reports = []
        for _path in (compare_picker.value or []):
            if os.path.isfile(_path):
                with open(_path) as _fh:
                    _reports.append(json.load(_fh))
        set_comparison_reports(_reports)
    return


@app.cell(hide_code=True)
def _(get_report, get_tier1_arrays, mo, np, plt):
    # ── Tier result tabs ──
    # Render the active report as a set of tabs, one per benchmark tier.
    # Each tab is self-contained: Tier 1 shows EAS + LOO diagnostics,
    # Tier 2 shows aggregate RMSE/bias/CRPS/coverage + PP-plot, and
    # Tier 3 shows chi² + PPC bar charts per observation.  A "Config"
    # tab records the execution settings so reports remain reproducible.
    _report = get_report()
    mo.stop(
        _report is None,
        mo.md("*Load a report above, or run a benchmark below.*"),
    )

    # Build tab content
    _tabs = {}

    # ---- TIER 1 TAB ----
    _t1 = _report.get("tier1")
    if _t1:
        _t1_items = []
        _t1_items.append(mo.md("## Tier 1 — Grid Reconstruction Fidelity"))

        # Surface the composite Tier 1 score first because it is the quickest
        # sanity check for whether an emulator is usable before inspecting the
        # more granular diagnostic plots below.
        _eas = _t1.get("emulator_accuracy_score")
        if _eas is not None:
            _eas_color = (
                "#2ecc71" if _eas >= 95
                else "#f39c12" if _eas >= 85
                else "#e74c3c"
            )
            _pca_ev_raw = _t1.get("pca_explained_variance")
            _loo_rmse_raw = _t1.get("loo_flux_rmse_median")
            _q2_raw = _t1.get("q2_aggregate")
            _pca_str = f"{_pca_ev_raw*100:.1f}%" if _pca_ev_raw is not None else "n/a"
            _loo_str = f"{_loo_rmse_raw:.4f}" if _loo_rmse_raw is not None else "n/a"
            _q2_str = f"{_q2_raw*100:.1f}%" if _q2_raw is not None else "n/a"
            # Warn when a component drags EAS down
            _pca_warn = " \u26a0 PCA EV low \u2014 consider more components" if (_pca_ev_raw is not None and _pca_ev_raw < 0.95) else ""
            _loo_warn = " \u26a0 RMSE > normalised flux scale" if (_loo_rmse_raw is not None and _loo_rmse_raw > 1.0) else ""
            _q2_warn = " \u26a0 GP worse than mean predictor" if (_q2_raw is not None and _q2_raw < 0) else ""
            _t1_items.append(mo.md(
                f'<div style="border:2px solid {_eas_color}; border-radius:8px; '
                f'padding:12px 20px; display:inline-block; margin-bottom:8px;">'
                f'<span style="font-size:1.1em; color:#aaa;">Emulator Accuracy Score (Napkin Math)</span><br>'
                f'<span style="font-size:2.4em; font-weight:bold; color:{_eas_color};">{_eas:.2f}%</span><br>'
                f'<span style="font-size:0.85em; color:#aaa;">'
                f'PCA explained variance: <b>{_pca_str}</b>{_pca_warn} &nbsp;|&nbsp; '
                f'Q\u00b2 aggregate: <b>{_q2_str}</b>{_q2_warn} &nbsp;|&nbsp; '
                f'LOO flux RMSE (median): <b>{_loo_str}</b>{_loo_warn}</span><br>'
                f'<span style="font-size:0.75em; color:#888;">'
                f'EAS = clamp(PCA EV, 0\u20131) &times; max(0, 1 &minus; LOO RMSE) &times; 100 '
                f'&mdash; PCA EV is fixed for a given grid; Q\u00b2 and LOO RMSE depend on training</span>'
                f'</div>'
            ))
        else:
            _t1_items.append(mo.callout(
                mo.md("*Emulator Accuracy Score unavailable — re-run Tier 1 with a grid path to compute flux-space metrics.*"),
                kind="warn",
            ))

        # Keep the raw Tier 1 metrics visible even when EAS is present so the
        # user can see which component of the score is limiting performance.
        _metric_info = {
            "n_components": ("PCA Components", ""),
            "n_grid_points": ("Grid Points", ""),
            "n_params": ("Parameters", ""),
            "tier1_time_s": ("Runtime (s)", ""),
            "pca_explained_variance": ("PCA Explained Variance", "higher is better; \u22650.999 typical"),
            "q2_aggregate": ("Aggregate Q\u00b2 (LOO R\u00b2)", "higher is better; \u22650.99 ideal"),
            "nlpd_mean": ("Mean NLPD", "lower is better; compare across configs"),
            "std_resid_mean": ("Std. Residual Mean", "\u22480 ideal; bias if far from 0"),
            "std_resid_var": ("Std. Residual Variance", "\u22481 ideal; >1 \u2192 under-confident"),
            "pca_recon_rmse_median": ("PCA RMSE (median)", "lower is better; PCA truncation floor"),
            "pca_recon_rmse_95": ("PCA RMSE (95th pctl)", "lower is better; worst-case PCA error"),
            "pca_max_fractional_resid": ("PCA Max Frac. Residual", "lower is better; worst PCA pixel error"),
            "loo_flux_rmse_median": ("LOO Flux RMSE (median)", "lower is better; total emulator error"),
            "loo_flux_rmse_95": ("LOO Flux RMSE (95th pctl)", "lower is better; worst-case emulator error"),
            "max_fractional_resid": ("Max Frac. Residual", "lower is better; worst pixel error overall"),
        }
        _summary_rows = []
        for _key in _metric_info:
            if _key in _t1:
                _name, _notes = _metric_info[_key]
                _val = _t1[_key]
                _formatted = f"{_val:.6g}" if isinstance(_val, float) else str(_val)
                _summary_rows.append({"Name": _name, "Metric": _key, "Value": _formatted, "Notes": _notes})

        if _summary_rows:
            _t1_items.append(mo.ui.table(_summary_rows, label="Summary"))

        # --- LOO RMSE per component ---
        _rmses = _t1.get("loo_rmse_per_comp", [])
        _q2s = _t1.get("q2_per_comp", [])
        _nlpds = _t1.get("nlpd_per_comp", [])

        if _rmses:
            _fig_rmse, _ax_rmse = plt.subplots(figsize=(7, 4))
            _ax_rmse.bar(range(len(_rmses)), _rmses, color="steelblue")
            _ax_rmse.set_xlabel("PCA Component")
            _ax_rmse.set_ylabel("LOO RMSE")
            _ax_rmse.set_title("Leave-One-Out RMSE")
            plt.tight_layout()
            _rmse_info = mo.md(
                "### LOO RMSE per Component\n\n"
                "**What this shows:** The root-mean-square prediction error for each "
                "PCA weight when that training point is left out of the GP fit.\n\n"
                "**Better looks like:** Short bars across all components, with no single "
                "component dominating the total error.\n\n"
                "**Watch out for:** A sharp increase in RMSE for higher-order components — "
                "the GP may struggle to model low-variance PCA modes.\n\n"
                "**How to improve:** Add more training grid points, or reduce the number of "
                "PCA components to drop poorly-modelled trailing components."
            )
            _t1_items.append(mo.accordion(
                {"LOO RMSE per Component": mo.hstack([mo.as_html(_fig_rmse), _rmse_info], widths=[3, 1])}
            ))
            plt.close()

        # --- Q² per component ---
        if _q2s:
            _fig_q2, _ax_q2 = plt.subplots(figsize=(7, 4))
            _colors = ["#2ecc71" if v >= 0.99 else "#f39c12" if v >= 0.9 else "#e74c3c" for v in _q2s]
            _ax_q2.bar(range(len(_q2s)), _q2s, color=_colors)
            _ax_q2.set_xlabel("PCA Component")
            _ax_q2.set_ylabel("Q\u00b2")
            _ax_q2.set_title("Per-Component Q\u00b2 (LOO R\u00b2)")
            _ax_q2.axhline(1.0, color="grey", ls="--", lw=0.8)
            plt.tight_layout()
            _q2_info = mo.md(
                "### Per-Component Q\u00b2\n\n"
                "**What this shows:** The leave-one-out predictive R\u00b2 for each PCA "
                "component. Bars are colour-coded: green (\u22650.99), amber (\u22650.9), "
                "red (<0.9).\n\n"
                "**Better looks like:** All bars green and close to 1.0.\n\n"
                "**Watch out for:** Red or amber bars, especially on the leading "
                "components which carry the most spectral variance.\n\n"
                "**How to improve:** More training points generally raise Q\u00b2. "
                "If only trailing components are red, consider reducing the PCA rank."
            )
            _t1_items.append(mo.accordion(
                {"Per-Component Q\u00b2": mo.hstack([mo.as_html(_fig_q2), _q2_info], widths=[3, 1])}
            ))
            plt.close()

        # --- NLPD per component ---
        if _nlpds:
            _fig_nlpd, _ax_nlpd = plt.subplots(figsize=(7, 4))
            _ax_nlpd.bar(range(len(_nlpds)), _nlpds, color="darkorange")
            _ax_nlpd.set_xlabel("PCA Component")
            _ax_nlpd.set_ylabel("NLPD")
            _ax_nlpd.set_title("LOO Neg. Log Predictive Density")
            plt.tight_layout()
            _nlpd_info = mo.md(
                "### LOO Negative Log Predictive Density\n\n"
                "**What this shows:** A proper scoring rule that penalises both inaccurate "
                "predictions and miscalibrated uncertainties. Lower is better.\n\n"
                "**Better looks like:** Low, roughly uniform bars across components.\n\n"
                "**Watch out for:** Spikes on specific components — the GP is confidently "
                "wrong there (high error + tight variance).\n\n"
                "**How to improve:** NLPD spikes often respond to a different GP optimiser "
                "or to adding training points near the problematic region of parameter space."
            )
            _t1_items.append(mo.accordion(
                {"NLPD per Component": mo.hstack([mo.as_html(_fig_nlpd), _nlpd_info], widths=[3, 1])}
            ))
            plt.close()

        # --- LOO residual distribution ---
        _std_resid = _t1.get("loo_std_resid", [])
        if _std_resid:
            _flat = np.array(_std_resid).flatten()
            _fig_hist, _ax_hist = plt.subplots(figsize=(7, 4))
            _ax_hist.hist(_flat, bins=50, density=True, alpha=0.7, color="steelblue", label="Leave-One-Out")
            _xgauss = np.linspace(-4, 4, 200)
            _ax_hist.plot(_xgauss, 1.0/(np.sqrt(2*np.pi))*np.exp(-0.5*_xgauss**2), "r-", label="N(0,1)")
            _ax_hist.set_xlabel("Standardised Residual")
            _ax_hist.set_title("LOO Residual Distribution")
            _ax_hist.legend()
            plt.tight_layout()
            _hist_info = mo.md(
                "### Residual Distribution\n\n"
                "**What this shows:** Histogram of standardised LOO residuals across all "
                "components, overlaid with the expected N(0,1) Gaussian (red curve).\n\n"
                "**Better looks like:** The histogram closely tracks the red curve — "
                "symmetric, centred on zero, with the same width.\n\n"
                "**Watch out for:** Heavy tails (the GP underestimates uncertainty), "
                "a shifted centre (systematic bias), or a narrower/wider peak than N(0,1) "
                "(over/under-confident variances).\n\n"
                "**How to improve:** Bias \u2192 check for a bug in the mean function. "
                "Heavy tails \u2192 add training points or try a different optimiser."
            )
            _t1_items.append(mo.accordion(
                {"LOO Residual Distribution": mo.hstack([mo.as_html(_fig_hist), _hist_info], widths=[3, 1])}
            ))
            plt.close()

        # --- Q-Q plot of standardised residuals ---
        _std_resid = _t1.get("loo_std_resid", [])
        if _std_resid:
            from scipy import stats as _sp_stats
            _flat = np.sort(np.array(_std_resid).flatten())
            _n_pts = len(_flat)
            _theoretical = _sp_stats.norm.ppf(
                (np.arange(1, _n_pts + 1) - 0.5) / _n_pts
            )
            _fig_qq, _ax_qq = plt.subplots(figsize=(5, 5))
            _ax_qq.scatter(_theoretical, _flat, s=1, alpha=0.4, color="steelblue")
            _qq_lim = max(abs(_theoretical[0]), abs(_theoretical[-1]), abs(_flat[0]), abs(_flat[-1]))
            _ax_qq.plot([-_qq_lim, _qq_lim], [-_qq_lim, _qq_lim], "r-", lw=1.0)
            _ax_qq.set_xlabel("Theoretical Quantiles (N(0,1))")
            _ax_qq.set_ylabel("Sample Quantiles")
            _ax_qq.set_title("Q\u2013Q Plot of Standardised LOO Residuals")
            _ax_qq.set_aspect("equal")
            plt.tight_layout()
            _qq_info = mo.md(
                "### Q\u2013Q Plot\n\n"
                "**What this shows:** Compares the quantiles of the standardised LOO residuals "
                "against a standard normal distribution. If the GP uncertainty estimates are "
                "well-calibrated the points should lie on the red diagonal.\n\n"
                "**Better looks like:** All points hugging the 1:1 red line with minimal scatter.\n\n"
                "**Watch out for:** An S-shape (heavy tails \u2014 under-confident or outlier-prone), "
                "a banana curve (skew), or systematic departure from the diagonal.\n\n"
                "**How to improve:** Heavy tails suggest the GP variance is too small for some "
                "predictions \u2014 try adding training points or a different kernel/optimiser."
            )
            _t1_items.append(mo.accordion(
                {"Q\u2013Q Plot": mo.hstack([mo.as_html(_fig_qq), _qq_info], widths=[2, 1])}
            ))
            plt.close()

        # --- Per-wavelength RMSE envelope ---
        _arrays = get_tier1_arrays() or {}
        _pca_wl = _arrays.get("pca_per_wl_rmse")
        _loo_wl = _arrays.get("loo_per_wl_rmse")
        _wl = _arrays.get("wavelength")
        if _pca_wl is not None and _wl is not None:
            _fig_wl, _ax_wl = plt.subplots(figsize=(12, 4))
            _ax_wl.plot(_wl, _pca_wl, label="PCA truncation only", color="#3498db", lw=1.0)
            if _loo_wl is not None:
                _ax_wl.plot(_wl, _loo_wl, label="LOO (PCA + GP)", color="#e74c3c", lw=1.0, alpha=0.8)
            _ax_wl.set_xlabel("Wavelength (\u00c5)")
            _ax_wl.set_ylabel("RMSE (normalised flux)")
            _ax_wl.set_title("Per-Wavelength Reconstruction Error")
            _ax_wl.legend()
            _ax_wl.set_xlim(_wl[0], _wl[-1])
            plt.tight_layout()
            _wl_info = mo.md(
                "### Per-Wavelength RMSE\n\n"
                "**What this shows:** Reconstruction error at every wavelength bin. The blue "
                "curve is the irreducible PCA truncation error; the red curve adds the GP "
                "prediction error on top.\n\n"
                "**Better looks like:** Both curves low and close together. The PCA curve "
                "sets a floor that the GP cannot beat.\n\n"
                "**Watch out for:** A large gap between the two curves (GP struggling); "
                "localised spikes at specific wavelengths (e.g. strong spectral lines "
                "that are hard to capture in few PCA components).\n\n"
                "**How to improve:** High PCA error \u2192 increase the number of PCA components. "
                "High GP error above the PCA floor \u2192 add more training grid points or "
                "try a different optimiser."
            )
            _t1_items.append(mo.accordion(
                {"Per-Wavelength RMSE Envelope": mo.hstack([mo.as_html(_fig_wl), _wl_info], widths=[3, 1])}
            ))
            plt.close()

        # --- Worst-case spectra overlay ---
        _orig = _arrays.get("original_flux")
        _pca_rec = _arrays.get("pca_recon_flux")
        _loo_rec = _arrays.get("loo_recon_flux")
        if _orig is not None and _loo_rec is not None and _wl is not None:
            _loo_rmse_arr = _arrays.get("loo_flux_rmse")
            if _loo_rmse_arr is None:
                _loo_rmse_arr = _t1.get("loo_flux_rmse", [])
            if hasattr(_loo_rmse_arr, '__len__') and len(_loo_rmse_arr) > 0:
                _sorted_idx = np.argsort(_loo_rmse_arr)
                _worst_idx = _sorted_idx[-3:][::-1]
                _fig_worst, _axes_worst = plt.subplots(len(_worst_idx), 1, figsize=(12, 3 * len(_worst_idx)), sharex=True)
                if len(_worst_idx) == 1:
                    _axes_worst = [_axes_worst]
                for _wi, _idx in enumerate(_worst_idx):
                    _ax = _axes_worst[_wi]
                    _ax.plot(_wl, _orig[_idx], label="Original", color="#f1c40f", lw=0.8)
                    if _pca_rec is not None:
                        _ax.plot(_wl, _pca_rec[_idx], label="PCA recon", color="#3498db", lw=0.8, ls="--")
                    _ax.plot(_wl, _loo_rec[_idx], label="LOO recon", color="#e74c3c", lw=0.8, ls="--")
                    _ax.set_ylabel("Normalised Flux")
                    _rmse_val = _loo_rmse_arr[_idx] if hasattr(_loo_rmse_arr, '__getitem__') else "?"
                    _ax.set_title(f"Spectrum #{_idx} (LOO RMSE = {_rmse_val:.5f})")
                    if _wi == 0:
                        _ax.legend(fontsize=8)
                _axes_worst[-1].set_xlabel("Wavelength (\u00c5)")
                plt.tight_layout()
                _worst_info = mo.md(
                    "### Worst-Case Spectra\n\n"
                    "**What this shows:** The three grid-point spectra with the largest "
                    "LOO reconstruction error, overlaid with the PCA-only and full "
                    "LOO reconstructions.\n\n"
                    "**Better looks like:** Dashed reconstructions closely tracking the "
                    "yellow original, with residuals invisible at this scale.\n\n"
                    "**Watch out for:** Systematic offsets in the continuum, poor fits "
                    "around strong absorption lines, or large discrepancies concentrated "
                    "at the edges of the wavelength range.\n\n"
                    "**How to improve:** Worst cases often sit at the edges of parameter "
                    "space. Ensure the training grid has adequate coverage there. If the "
                    "PCA reconstruction (blue) is already poor, more PCA components are needed."
                )
                _t1_items.append(mo.accordion(
                    {"Worst-Case Spectra Overlay": mo.hstack([mo.as_html(_fig_worst), _worst_info], widths=[3, 1])}
                ))
                plt.close()

                # --- Best-case spectra overlay ---
                _best_idx = _sorted_idx[:3]
                _fig_best, _axes_best = plt.subplots(len(_best_idx), 1, figsize=(12, 3 * len(_best_idx)), sharex=True)
                if len(_best_idx) == 1:
                    _axes_best = [_axes_best]
                for _bi, _idx in enumerate(_best_idx):
                    _ax = _axes_best[_bi]
                    _ax.plot(_wl, _orig[_idx], label="Original", color="#f1c40f", lw=0.8)
                    if _pca_rec is not None:
                        _ax.plot(_wl, _pca_rec[_idx], label="PCA recon", color="#3498db", lw=0.8, ls="--")
                    _ax.plot(_wl, _loo_rec[_idx], label="LOO recon", color="#e74c3c", lw=0.8, ls="--")
                    _ax.set_ylabel("Normalised Flux")
                    _rmse_val = _loo_rmse_arr[_idx] if hasattr(_loo_rmse_arr, '__getitem__') else "?"
                    _ax.set_title(f"Spectrum #{_idx} (LOO RMSE = {_rmse_val:.5f})")
                    if _bi == 0:
                        _ax.legend(fontsize=8)
                _axes_best[-1].set_xlabel("Wavelength (\u00c5)")
                plt.tight_layout()
                _best_info = mo.md(
                    "### Best-Case Spectra\n\n"
                    "**What this shows:** The three grid-point spectra with the smallest "
                    "LOO reconstruction error. These represent the emulator at its best.\n\n"
                    "**Better looks like:** All three reconstructions virtually "
                    "indistinguishable from the original.\n\n"
                    "**Watch out for:** Even in the best cases, check for systematic "
                    "offsets at specific wavelengths \u2014 these hint at features the PCA "
                    "basis cannot capture regardless of the GP fit.\n\n"
                    "**How to improve:** If even the best cases show visible residuals, "
                    "the PCA truncation is too aggressive \u2014 increase the number of components."
                )
                _t1_items.append(mo.accordion(
                    {"Best-Case Spectra Overlay": mo.hstack([mo.as_html(_fig_best), _best_info], widths=[3, 1])}
                ))
                plt.close()

        _tabs["Tier 1"] = mo.vstack(_t1_items)

    # ---- TIER 2 TAB ----
    _t2 = _report.get("tier2")
    if _t2:
        _t2_items = []
        _t2_items.append(mo.md("## Tier 2 — Test Grid Parameter Recovery"))
        _n_proc = _t2.get('n_processed', 0)
        _n_spec = _t2.get('n_spectra', '?')
        _n_fail = _t2.get('n_failures', 0)
        _n_nc = _t2.get('n_not_converged', 0)
        _t2_time = _t2.get('tier2_time_s', 0)
        _status_parts = [f"**{_n_proc}/{_n_spec} spectra** processed"]
        if _n_fail:
            _status_parts.append(f"{_n_fail} failed")
        if _n_nc:
            _status_parts.append(f"{_n_nc} not converged")
        _status_parts.append(f"in {_t2_time:.0f}s")
        _t2_items.append(mo.md(" — ".join(_status_parts)))

        # Preserve stage-specific failures so a partial Tier 2 run is still
        # diagnosable instead of silently collapsing into aggregate metrics.
        _flog = _t2.get("failure_log", [])
        if _flog:
            _fail_kind = "danger" if _n_proc == 0 else "warn"
            _flog_rows = [
                {"Run": _e["run"], "Stage": _e["stage"], "Error": _e["error"]}
                for _e in _flog
            ]
            _tb_blocks = "\n\n".join(
                f"**{_e['run']} ({_e['stage']}):**\n```\n{_e.get('traceback','').strip()}\n```"
                for _e in _flog if _e.get("traceback", "").strip()
            )
            _t2_items.append(mo.callout(
                mo.vstack([
                    mo.md(f"**{len(_flog)} failure(s) recorded.** "
                          "Check the table and tracebacks below to diagnose."),
                    mo.ui.table(_flog_rows, label="Failure Log"),
                    mo.md("#### Tracebacks") if _tb_blocks else mo.md(""),
                    mo.md(_tb_blocks) if _tb_blocks else mo.md(""),
                ]),
                kind=_fail_kind,
            ))
        elif _n_proc == 0:
            _t2_items.append(mo.callout(
                mo.md("All spectra failed but no failure log was recorded. "
                      "Re-run the benchmark to capture error details."),
                kind="danger",
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
    mo.stop(
        _arrs is None,
        mo.md("*Run a Tier 1 benchmark below to view the interactive reconstruction plot.*"),
    )

    _M = _arrs["original_flux"].shape[0]
    _pnames = _arrs["param_names"]

    # The slider indexes the original training grid ordering used when the live
    # Tier 1 run cached the reconstruction arrays in notebook state.
    spectrum_slider = mo.ui.slider(
        start=0, stop=_M - 1, value=0, step=1, show_value=True,
        label="Grid point index", full_width=True
    )
    add_mean_switch = mo.ui.switch(
        value=False, label="Add mean spectrum to individual components"
    )
    recon_n_grid = _M
    recon_param_names = _pnames

    mo.vstack([
        mo.md("## Interactive Reconstruction Explorer"),
        mo.md(f"Browse all **{_M}** training spectra. "
               "Overlay original grid spectrum, PCA reconstruction, and Leave-One-Out GP reconstruction."),
        spectrum_slider,
    ])
    return add_mean_switch, recon_param_names, spectrum_slider


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

    # Pair the flux plots with the exact parameter point and scalar error metrics
    # so the user can correlate bad reconstructions with specific regions of the
    # training grid.
    _param_badges = " | ".join(
        f"**{_pnames[_j]}** = {_gp[_j]:.4g}" for _j in range(len(_pnames))
    )
    _rmse_loo = float(np.sqrt(np.mean((_orig - _loo) ** 2)))
    _rmse_pca = float(np.sqrt(np.mean((_orig - _pca) ** 2)))
    _max_frac = float(np.nanmax(np.abs(_orig - _loo) / (np.abs(_orig) + 1e-30)))

    _readout = mo.md(
        f"{_param_badges}  \n"
        f"Leave-One-Out RMSE: **{_rmse_loo:.6g}** &nbsp; PCA RMSE: **{_rmse_pca:.6g}** "
        f"&nbsp; Max fractional residual: **{_max_frac:.4g}**"
    )

    # Plot all three spectra on the same wavelength grid because the key Tier 1
    # question is how much structure the PCA-only and GP-assisted models miss.
    _df_spec = pd.DataFrame({
        "Wavelength": np.tile(_wl, 3),
        "Flux": np.concatenate([_orig, _pca, _loo]),
        "Series": (
            ["Original"] * len(_wl) +
            ["PCA Recon"] * len(_wl) +
            ["Leave-One-Out GP Recon"] * len(_wl)
        ),
    })

    _color_scale = alt.Scale(
        domain=["Original", "PCA Recon", "Leave-One-Out GP Recon"],
        range=["#4c78a8", "#f58518", "#54a24b"],
    )
    _dash_scale = alt.Scale(
        domain=["Original", "PCA Recon", "Leave-One-Out GP Recon"],
        range=[[0, 0], [6, 4], [4, 2]],
    )

    # LOO GP confidence band (±2σ) — propagated from per-component LOO
    # predictive variance through the PCA inverse transform.
    _loo_recon_var = _arrs.get("loo_recon_var")
    _ci_chart = alt.LayerChart()
    if _loo_recon_var is not None:
        _sigma = np.sqrt(np.maximum(_loo_recon_var[_idx], 0.0))
        _df_ci = pd.DataFrame({
            "Wavelength": _wl,
            "Lower (2σ)": _loo - 2 * _sigma,
            "Upper (2σ)": _loo + 2 * _sigma,
        })
        _ci_chart = (
            alt.Chart(_df_ci)
            .mark_area(opacity=0.18, color="#54a24b")
            .encode(
                x=alt.X("Wavelength:Q"),
                y=alt.Y("Lower (2σ):Q"),
                y2=alt.Y2("Upper (2σ):Q"),
            )
        )

    _spec_lines = (
        alt.Chart(_df_spec)
        .mark_line(strokeWidth=1.5, opacity=0.85)
        .encode(
            x=alt.X("Wavelength:Q", title="Wavelength (Å)"),
            y=alt.Y("Flux:Q", title="Normalised Flux"),
            color=alt.Color("Series:N", scale=_color_scale, legend=alt.Legend(title="Series")),
            strokeDash=alt.StrokeDash("Series:N", scale=_dash_scale, legend=None),
            tooltip=["Wavelength:Q", "Flux:Q", "Series:N"],
        )
    )

    _spec_chart = (
        (_ci_chart + _spec_lines)
        .properties(width="container", height=350, title=f"Spectrum #{_idx}")
        .interactive()
    )

    # Residuals make small local mismatches visible even when the overlaid flux
    # curves look nearly identical by eye.
    _resid_loo = _orig - _loo
    _resid_pca = _orig - _pca
    _df_resid = pd.DataFrame({
        "Wavelength": np.tile(_wl, 2),
        "Residual": np.concatenate([_resid_loo, _resid_pca]),
        "Series": ["Leave-One-Out GP Residual"] * len(_wl) + ["PCA Residual"] * len(_wl),
    })

    _resid_color = alt.Scale(
        domain=["Leave-One-Out GP Residual", "PCA Residual"],
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
            x=alt.X("Wavelength:Q", title="Wavelength (Å)"),
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

    # Rank every cached Tier 1 spectrum by reconstruction quality so the worst
    # cases are easy to inspect without manually scrubbing the slider.
    _loo_rmse = np.sqrt(np.mean((_orig - _loo) ** 2, axis=1))
    _pca_rmse = np.sqrt(np.mean((_orig - _pca) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        _max_frac = np.nanmax(
            np.abs(_orig - _loo) / (np.abs(_orig) + 1e-30), axis=1
        )

    # Leave-One-Out RMSE is used as the primary ranking because it reflects the
    # full PCA+GP reconstruction path that Tier 1 is benchmarking.
    _order = np.argsort(-_loo_rmse)

    _rows = []
    for _rank, _i in enumerate(_order):
        _row = {"Rank": _rank + 1, "Index": int(_i)}
        for _j, _pn in enumerate(_pnames):
            _row[_pn] = round(float(_gp[_i, _j]), 4)
        _row["Leave-One-Out RMSE"] = f"{_loo_rmse[_i]:.6g}"
        _row["PCA RMSE"] = f"{_pca_rmse[_i]:.6g}"
        _row["Max Frac Resid"] = f"{_max_frac[_i]:.4g}"
        _rows.append(_row)

    mo.vstack([
        mo.md("#### Worst-Reconstructed Spectra"),
        mo.md("Sorted by descending Leave-One-Out RMSE. Use the slider above to inspect any spectrum."),
        mo.ui.table(_rows, label="All spectra", page_size=15),
    ])
    return

@app.cell(hide_code=True)
def _(add_mean_switch, get_emulator, get_tier1_arrays, mo, np, pd, plt, spectrum_slider):
    # ── Cumulative PCA Component Reconstruction Diagnostic ──
    # Builds up the spectrum component-by-component to reveal exactly which
    # eigenspectrum introduces "ghosting" artifacts in the LOO GP reconstruction.
    _emu = get_emulator()
    _arrs = get_tier1_arrays()
    if _emu is None or _arrs is None:
        mo.stop(True)

    _loo_mu = _arrs.get("loo_mu")     # (K, M)
    _loo_var = _arrs.get("loo_var")    # (K, M)
    if _loo_mu is None or _loo_var is None:
        mo.stop(True)

    _idx = spectrum_slider.value
    _wl = _arrs["wavelength"]
    _orig = _arrs["original_flux"][_idx]
    _ncomps = _emu.ncomps
    _X = _emu.eigenspectra * _emu.flux_std         # (K, P)
    _w_true = _emu.weights[_idx]                   # (K,)  PCA true weights
    _w_loo = _loo_mu[:, _idx]                      # (K,)  LOO predicted weights
    _flux_mean = _emu.flux_mean                    # (P,)

    # ── Per-component weight error table ──
    _delta_w = _w_loo - _w_true
    _loo_sigma = np.sqrt(np.maximum(_loo_var[:, _idx], 0.0))
    _df_table = pd.DataFrame({
        "Component": list(range(_ncomps)),
        "w_true": [f"{v:.4e}" for v in _w_true],
        "w_LOO": [f"{v:.4e}" for v in _w_loo],
        "|Δw|": [f"{abs(v):.4e}" for v in _delta_w],
        "LOO σ": [f"{v:.4e}" for v in _loo_sigma],
        "|Δw|/σ": [f"{abs(d)/max(s, 1e-30):.2f}" for d, s in zip(_delta_w, _loo_sigma)],
    })

    # ── Matplotlib multi-subplot: cumulative reconstruction ──
    _ncols = min(3, _ncomps)
    _nrows = int(np.ceil(_ncomps / _ncols))
    _fig, _axes = plt.subplots(
        _nrows, _ncols,
        figsize=(6 * _ncols, 4 * _nrows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    _fig.patch.set_facecolor("black")

    for _j in range(_ncomps):
        _ax = _axes[_j // _ncols][_j % _ncols]
        _ax.set_facecolor("black")

        # Cumulative reconstruction: sum of components 0..j
        _pca_cum = _w_true[:_j + 1] @ _X[:_j + 1] + _flux_mean
        _loo_cum = _w_loo[:_j + 1] @ _X[:_j + 1] + _flux_mean

        _ax.plot(_wl, _orig, color="#f0c75e", lw=0.8, alpha=0.7, label="Original")
        _ax.plot(_wl, _pca_cum, color="#4c78a8", lw=0.9, ls="--", alpha=0.9, label="PCA cumul.")
        _ax.plot(_wl, _loo_cum, color="#e74c3c", lw=0.9, ls="--", alpha=0.9, label="LOO cumul.")

        _dw = _delta_w[_j]
        _ax.set_title(
            f"0–{_j}  |  Δw$_{_j}$ = {_dw:+.3e}",
            fontsize=8, color="white",
        )
        _ax.tick_params(colors="white", labelsize=7)
        for _spine in _ax.spines.values():
            _spine.set_color("white")

    # Turn off unused subplots
    for _k in range(_ncomps, _nrows * _ncols):
        _axes[_k // _ncols][_k % _ncols].set_visible(False)

    # Legend + labels only on first subplot
    _axes[0][0].legend(fontsize=7, loc="upper right",
                       facecolor="black", edgecolor="white",
                       labelcolor="white")
    _fig.supxlabel("Wavelength (Å)", color="white", fontsize=9)
    _fig.supylabel("Normalised Flux", color="white", fontsize=9)
    _fig.suptitle(
        f"Cumulative PCA Component Reconstruction — Spectrum #{_idx}",
        color="white", fontsize=11, y=1.01,
    )
    _fig.tight_layout()

    # ── Matplotlib multi-subplot: individual component contribution ──
    # Each subplot shows only component j's contribution (w_j * X_j),
    # optionally offset by the mean spectrum, isolating each eigenspectrum's effect.
    _add_mean = add_mean_switch.value
    _offset = _flux_mean if _add_mean else np.zeros_like(_flux_mean)

    _fig2, _axes2 = plt.subplots(
        _nrows, _ncols,
        figsize=(5 * _ncols, 2.8 * _nrows),
        sharex=True,
        squeeze=False,
    )
    _fig2.patch.set_facecolor("black")

    for _j in range(_ncomps):
        _ax2 = _axes2[_j // _ncols][_j % _ncols]
        _ax2.set_facecolor("black")

        # Individual component: w_j * X_j (+ mean if toggled on)
        _pca_ind = _w_true[_j] * _X[_j] + _offset
        _loo_ind = _w_loo[_j] * _X[_j] + _offset

        if _add_mean:
            _ax2.plot(_wl, _orig, color="#f0c75e", lw=0.8, alpha=0.7, label="Original")
        _ax2.plot(_wl, _pca_ind, color="#4c78a8", lw=0.9, ls="--", alpha=0.9, label="PCA")
        _ax2.plot(_wl, _loo_ind, color="#e74c3c", lw=0.9, ls="--", alpha=0.9, label="LOO")

        _dw = _delta_w[_j]
        _ax2.set_title(
            f"Comp {_j}  |  Δw$_{_j}$ = {_dw:+.3e}",
            fontsize=8, color="white",
        )
        _ax2.tick_params(colors="white", labelsize=7)
        for _spine in _ax2.spines.values():
            _spine.set_color("white")

    for _k in range(_ncomps, _nrows * _ncols):
        _axes2[_k // _ncols][_k % _ncols].set_visible(False)

    _axes2[0][0].legend(fontsize=7, loc="upper right",
                        facecolor="black", edgecolor="white",
                        labelcolor="white")
    _fig2.supxlabel("Wavelength (Å)", color="white", fontsize=9)
    _fig2.supylabel("Normalised Flux", color="white", fontsize=9)
    _fig2.suptitle(
        f"Individual PCA Component Contribution — Spectrum #{_idx}",
        color="white", fontsize=11, y=1.01,
    )
    _fig2.tight_layout()

    # ── Accordion summaries ──
    _table_info = mo.md(
        "### Per-Component Weight Table\n\n"
        "**What this shows:** For the selected spectrum, compares the true PCA "
        "weights against the LOO GP-predicted weights for each component. "
        "The |Δw|/σ column measures deviation in units of the GP's own "
        "uncertainty — values > 2 indicate the GP is confidently wrong.\n\n"
        "**Watch out for:** Large |Δw|/σ on a specific component — that "
        "component's eigenspectrum likely drives any visible ghosting artifact."
    )
    _cum_info = mo.md(
        "### Cumulative Component Reconstruction\n\n"
        "**What this shows:** Builds the reconstructed spectrum one PCA "
        "component at a time (0, 0–1, 0–2, …). The yellow line is the original "
        "spectrum; blue dashed is the PCA reconstruction using true weights; red "
        "dashed is the LOO GP reconstruction.\n\n"
        "**Better looks like:** Blue and red lines converging onto yellow with "
        "each added component, with no sudden divergence.\n\n"
        "**Watch out for:** A subplot where the red line suddenly deviates from "
        "blue — that component is where the GP weight error enters the spectrum."
    )
    _ind_info = mo.md(
        "### Individual Component Contribution\n\n"
        "**What this shows:** Each subplot isolates a single eigenspectrum's "
        "contribution: $w_j \\times X_j$. Blue = PCA (true weight), Red = LOO "
        "(GP prediction). Toggle the switch above to add the mean spectrum as "
        "an offset for context.\n\n"
        "**Better looks like:** Blue and red overlapping closely in every subplot.\n\n"
        "**Watch out for:** A subplot where blue and red diverge sharply — that "
        "eigenspectrum encodes the spectral feature the GP is failing to predict."
    )

    _table_panel = mo.vstack([
        mo.md("##### Per-Component Weight Comparison"),
        mo.ui.table(_df_table, selection=None, page_size=_ncomps),
    ])
    _ind_panel = mo.vstack([
        add_mean_switch,
        mo.hstack([_fig2, _ind_info], widths=[3, 1]),
    ])
    mo.accordion({
        "Per-Component Weight Table": mo.hstack([_table_panel, _table_info], widths=[3, 1]),
        "Cumulative Component Reconstruction": mo.hstack([_fig, _cum_info], widths=[3, 1]),
        "Individual Component Contribution": _ind_panel,
    })
    return



@app.cell(hide_code=True)
def _(get_comparison_reports, mo, np, plt):
    _comp_reports = get_comparison_reports()
    mo.stop(
        not _comp_reports or len(_comp_reports) < 2,
        mo.md("*Select 2+ reports above to compare.*"),
    )

    _comp_items = [mo.md("## Report Comparison")]

    # Comparison reduces each report to a small shared schema so reports from
    # different runs can still be compared even if some optional fields differ.
    _t1_data = {}
    for _i, _rep in enumerate(_comp_reports):
        _tag = _rep.get("config", {}).get("tag", f"report_{_i}")
        _t1 = _rep.get("tier1", {})
        _t1_data[_tag] = {
            "eas": _t1.get("emulator_accuracy_score"),
            "loo_rmse": _t1.get("loo_flux_rmse_median"),
            "pca_ev": _t1.get("pca_explained_variance"),
            "nlpd": _t1.get("nlpd_mean"),
            "q2": _t1.get("q2_aggregate"),
        }

    if _t1_data:
        _rows = [
            {
                "Report": _k,
                "EAS (%)": f"{_v['eas']:.2f}" if _v["eas"] is not None else "\u2014",
                "Q\u00b2": f"{_v['q2']:.6g}" if _v["q2"] is not None else "\u2014",
                "NLPD": f"{_v['nlpd']:.4f}" if _v["nlpd"] is not None else "\u2014",
                "LOO RMSE (med)": f"{_v['loo_rmse']:.6g}" if _v["loo_rmse"] is not None else "\u2014",
                "PCA Expl. Var.": f"{_v['pca_ev']:.6g}" if _v["pca_ev"] is not None else "\u2014",
            }
            for _k, _v in _t1_data.items()
        ]
        _comp_items.append(mo.md("### Tier 1 Comparison"))
        _comp_items.append(mo.md(
            "**Q\u00b2** (LOO R\u00b2) and **NLPD** (Neg. Log Predictive Density) are publishable metrics. "
            "**EAS** is a convenience score: PCA EV \u00d7 (1 \u2212 LOO RMSE) \u00d7 100. "
            "Lower NLPD is better; higher Q\u00b2 and EAS are better."
        ))
        _comp_items.append(mo.ui.table(_rows))

    _all_params = set()
    for _rep in _comp_reports:
        _agg = _rep.get("tier2", {}).get("aggregate", {})
        _all_params.update(_agg.keys())

    if _all_params:
        _comp_items.append(mo.md("### Tier 2 Parameter RMSE Comparison"))
        _params = sorted(_all_params)
        _fig, _ax = plt.subplots(figsize=(max(10, len(_params)*2), 5))
        # Grouped bars keep each report aligned on the same parameter axis so
        # changes in recovery quality are easy to scan across experiments.
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
    # ── Live benchmark runner: file pickers ──
    # Discover available emulators and observation CSVs on disk so the user
    # can launch a fresh Tier 1/2/3 benchmark without leaving this notebook.
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
        # The naming convention lets the viewer recover the matching training
        # grid and test-grid without asking the user to pick three separate paths.
        _grid_stem = _emu_base.split("_emu_")[0]  # e.g. speculate_cv_bl_grid_v87f

        # Extract the parameter-scale tag from the emulator filename so the
        # lookup lands on the matching processed grid NPZ.
        _after_emu = _emu_base.split("_emu_")[1]  # 1234_linear_55inc_850-...
        # Try several delimiters in order of specificity:
        #   1. Inclination segment:  _XXinc_
        #   2. Wavelength range:     _NNNN-NNNNAA_  (always present)
        _tag_match = re.match(r"(.+?)_\d+inc_", _after_emu)
        if not _tag_match:
            _tag_match = re.match(r"(.+?)_\d+-\d+AA_", _after_emu)
        if _tag_match:
            _param_tag = _tag_match.group(1)
        else:
            # Fallback for older filenames that predate the stricter pattern.
            _param_tag = _after_emu.split("_850")[0].split("_1000")[0]

        # Tier 1 consumes the processed NPZ grid, while Tier 2 consumes the raw
        # test-grid directory that contains individual .spec files.
        _grid_pattern = f"Grid-Emulator_Files/{_grid_stem}_grid_{_param_tag}.npz"
        _grid_matches = sorted(glob.glob(_grid_pattern))
        if _grid_matches:
            matched_grid_path = _grid_matches[0]

        # The test-grid stem mirrors the training-grid stem with a name swap.
        _testgrid_stem = _grid_stem.replace("_grid_", "_testgrid_")
        _testgrid_pattern = f"sirocco_grids/{_testgrid_stem}*"
        _tg_matches = sorted(glob.glob(_testgrid_pattern))
        if _tg_matches:
            matched_testgrid_path = _tg_matches[0]

        # Show the inferred paths even when matching fails so the user can catch
        # naming mismatches before launching a live benchmark.
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
    set_emulator,
    set_report,
    set_status_msg,
    set_tier1_arrays,
    tier_picker,
    time,
):
    # Keep this cell inert during normal reactive reruns; the heavy work only
    # starts when the explicit run button is pressed.
    if not run_btn.value:
        mo.stop(True)

    # Surface an immediate status update before any imports or disk I/O.
    set_status_msg("Running benchmark...")
    _t0 = time.time()

    try:
        # Import the runtime-only benchmark machinery inside the execution cell.
        from pathlib import Path as _Path
        from Starfish.emulator import Emulator as _Emulator
        from Speculate_addons.speculate_benchmark import (
            run_tier1 as _run_tier1,
            run_tier3_single as _run_tier3_single,
            build_report_card as _build_report_card,
            save_report as _save_report,
            # Tier 2 helpers — viewer drives the loop for nested progress
            load_test_grid_spectrum as _load_spec,
            extract_ground_truth as _extract_gt,
            ensure_lookup_table as _ensure_lookup,
            internal_to_friendly as _to_friendly,
            run_mle_single as _run_mle,
            run_mcmc_single as _run_mcmc,
            aggregate_tier2_results as _agg_t2,
        )
        import numpy as _np
        import traceback as _tb

        _emu_path = emu_picker.value
        if not _emu_path or not os.path.isfile(_emu_path):
            set_status_msg("Error: select a valid emulator file.")
            mo.stop(True)

        # ---- Load emulator (spinner — single long operation) ----
        with mo.status.spinner(
            title="Loading Emulator",
            subtitle=f"Loading {os.path.basename(_emu_path)}…",
            remove_on_exit=True,
        ):
            _emu = _Emulator.load(_emu_path)

        # Persist the emulator in reactive state so downstream diagnostic
        # cells (e.g. cumulative component analysis) can access eigenspectra,
        # weights, flux scaling, etc. without reloading from disk.
        set_emulator(_emu)

        # The picker stores numeric tier ids; keep the selected run isolated to
        # this execution so later reactive reruns do not reuse stale results.
        _tiers = [_v for _v in (tier_picker.value or [])]
        _tier1_result = None
        _tier2_result = None
        _tier3_results = None

        # ---- Tier 1 (spinner — single LOO cross-validation pass) ----
        if 1 in _tiers and matched_grid_path:
            with mo.status.spinner(
                title="Tier 1 — Grid Reconstruction",
                subtitle="Running LOO cross-validation…",
                remove_on_exit=True,
            ):
                _tier1_result = _run_tier1(_emu, matched_grid_path)

            # Keep the large flux arrays in marimo state rather than in the JSON
            # report so the interactive reconstruction explorer can reuse them.
            _t1_arrays = _tier1_result.pop("_arrays", None)
            if _t1_arrays is not None:
                set_tier1_arrays(_t1_arrays)

        # ---- Tier 2 (progress_bar per spectrum + spinner per stage) ----
        if 2 in _tiers and matched_testgrid_path:
            _t2_t0 = time.time()
            _test_path = _Path(matched_testgrid_path)
            # Tier 2 works spectrum-by-spectrum from the decompressed test-grid files.
            _spec_files = sorted(_test_path.glob("run*.spec"))
            if max_spectra_slider.value:
                _spec_files = _spec_files[: max_spectra_slider.value]
            _n_t2 = len(_spec_files)

            # The lookup table is optional metadata used to compare recovered
            # parameters against the known test-grid inputs.
            _parquet = _ensure_lookup(_test_path)
            _friendly = _to_friendly(_emu.param_names)
            _n_params = len(_emu.param_names)

            # Accumulate both per-spectrum summaries and the raw posterior draws
            # needed by the aggregate Tier 2 metrics.
            _per_spectrum = []
            _all_samples = {n: [] for n in _friendly}
            _all_truths = {n: [] for n in _friendly}
            _failures = 0
            _n_not_converged = 0
            _failure_log = []
            _mcmc_steps_val = mcmc_steps_slider.value

            with mo.status.progress_bar(
                total=_n_t2,
                title="Tier 2 — Parameter Recovery",
                subtitle="Starting…",
                completion_title="Tier 2 — Parameter Recovery",
                completion_subtitle="Complete ✓",
                show_rate=True,
                show_eta=True,
                remove_on_exit=True,
            ) as _t2_bar:
                for _si, _sf in enumerate(_spec_files):
                    # Spectrum filenames encode the run number used by the lookup table.
                    _run_idx = int(_sf.stem.replace("run", ""))
                    _sname = _sf.name

                    # -- Load spectrum --
                    _t2_bar.update(increment=0, subtitle=f"Loading {_sname}…")
                    try:
                        _wl, _flux, _sigma = _load_spec(str(_sf), 55.0, (850, 1850))
                    except Exception as _e:
                        _failure_log.append({
                            "run": _sname, "stage": "load",
                            "error": str(_e), "traceback": _tb.format_exc(),
                        })
                        _failures += 1
                        _t2_bar.update(increment=1, subtitle=f"✗ {_sname} failed to load")
                        continue

                    # Ground truth is best-effort only; missing lookup metadata should
                    # not block the inference pass.
                    _gt = {}
                    if _parquet.exists():
                        try:
                            _gt = _extract_gt(str(_parquet), _run_idx, _emu.param_names)
                        except Exception:
                            pass

                    # -- MLE (spinner with iteration updates) --
                    with mo.status.spinner(
                        title=f"MLE — {_sname}",
                        subtitle=f"Optimising ({_si + 1}/{_n_t2})…",
                        remove_on_exit=True,
                    ) as _mle_spin:
                        # Forward the optimizer callback into the spinner so long
                        # runs still expose iteration and NLL progress in the UI.
                        def _mle_cb(it, mx, best_nll, elapsed):
                            _mle_spin.update(
                                subtitle=(
                                    f"Iteration {it}/{mx} | "
                                    f"Best NLL: {best_nll:.2f} | "
                                    f"Time: {elapsed:.1f}s"
                                )
                            )
                        try:
                            _mle = _run_mle(
                                _emu, _wl, _flux, _sigma, flux_scale="linear",
                                max_iter=10_000, iteration_callback=_mle_cb,
                            )
                        except Exception as _e:
                            _failure_log.append({
                                "run": _sname, "stage": "MLE",
                                "error": str(_e), "traceback": _tb.format_exc(),
                            })
                            _failures += 1
                            _t2_bar.update(increment=1, subtitle=f"✗ {_sname} MLE failed")
                            continue

                    # -- MCMC (spinner with step updates) --
                    with mo.status.spinner(
                        title=f"MCMC — {_sname}",
                        subtitle=f"Sampling {_mcmc_steps_val} steps ({_si + 1}/{_n_t2})…",
                        remove_on_exit=True,
                    ) as _mcmc_spin:
                        # Mirror the sampler callback into the spinner for coarse
                        # step-level progress during the posterior draw phase.
                        def _mcmc_cb(step, total, elapsed):
                            _mcmc_spin.update(
                                subtitle=(
                                    f"Step {step}/{total} | "
                                    f"Time: {elapsed:.1f}s"
                                )
                            )
                        try:
                            _mcmc = _run_mcmc(
                                _mle["model"], _mle["priors"],
                                nwalkers=32,
                                nsteps=_mcmc_steps_val,
                                burnin=200,
                                iteration_callback=_mcmc_cb,
                            )
                        except Exception as _e:
                            _failure_log.append({
                                "run": _sname, "stage": "MCMC",
                                "error": str(_e), "traceback": _tb.format_exc(),
                            })
                            _failures += 1
                            _t2_bar.update(increment=1, subtitle=f"✗ {_sname} MCMC failed")
                            continue

                    if not _mcmc["converged"]:
                        _n_not_converged += 1

                    # Persist the compact per-spectrum summary used by the Tier 2 report.
                    _spec_res = {
                        "run": _run_idx,
                        "mle_success": _mle["success"],
                        "mcmc_converged": _mcmc["converged"],
                        "n_effective": _mcmc["n_effective"],
                        "mle_grid_params": _mle["grid_params"],
                    }
                    for _pi, _fn in enumerate(_friendly):
                        if _fn in _mcmc["summary"]:
                            # Keep both the scalar posterior summary and the raw draws;
                            # the aggregate step uses the draws to compute coverage-style
                            # metrics across all spectra.
                            _spec_res[f"{_fn}_mean"] = _mcmc["summary"][_fn]["mean"]
                            _spec_res[f"{_fn}_std"] = _mcmc["summary"][_fn]["std"]
                            _all_samples[_fn].append(_mcmc["samples"][:, _pi])
                            if _fn in _gt:
                                # When ground truth is available, record the truth and the
                                # offset in posterior standard deviations for later review.
                                _all_truths[_fn].append(_gt[_fn])
                                _spec_res[f"{_fn}_truth"] = _gt[_fn]
                                _spec_res[f"{_fn}_delta_sigma"] = (
                                    _mcmc["summary"][_fn]["mean"] - _gt[_fn]
                                ) / max(_mcmc["summary"][_fn]["std"], 1e-10)

                    _per_spectrum.append(_spec_res)

                    _status_lbl = (
                        f"✓ {_sname} complete"
                        if _mcmc["converged"]
                        else f"⚠ {_sname} (not converged)"
                    )
                    _t2_bar.update(increment=1, subtitle=_status_lbl)

            # Collapse the per-spectrum bookkeeping into the JSON-safe Tier 2 report.
            _tier2_result = _agg_t2(
                per_spectrum=_per_spectrum,
                all_samples=_all_samples,
                all_truths=_all_truths,
                friendly_names=_friendly,
                n_params=_n_params,
                emu_min_params=_emu.min_params,
                emu_max_params=_emu.max_params,
                spec_files_count=_n_t2,
                failures=_failures,
                failure_log=_failure_log,
                mcmc_walkers=32,
                mcmc_steps=_mcmc_steps_val,
                mcmc_burnin=200,
                elapsed=time.time() - _t2_t0,
                n_not_converged=_n_not_converged,
            )

        # ---- Tier 3 (progress bar — one step per observation) ----
        if 3 in _tiers and obs_picker.value:
            _obs_list = obs_picker.value
            _tier3_results = []
            with mo.status.progress_bar(
                total=len(_obs_list),
                title="Tier 3 — Observational Spectra",
                subtitle="Starting…",
                completion_title="Tier 3 — Observational Spectra",
                completion_subtitle="Complete ✓",
                show_rate=True,
                show_eta=True,
                remove_on_exit=True,
            ) as _t3_bar:
                # Derive the grid name from the emulator filename so Tier 3
                # can export a valid Sirocco .pf for each observation.
                _emu_base = os.path.basename(emu_picker.value or "")
                _grid_stem = None
                if "_emu_" in _emu_base:
                    _grid_stem = _emu_base.split("_emu_")[0]

                for _obs_path in _obs_list:
                    # Tier 3 reports are independent, so the loop only needs to append
                    # each completed result and advance the progress bar.
                    _t3_bar.update(
                        increment=0,
                        subtitle=f"Fitting {os.path.basename(_obs_path)}…",
                    )
                    _r = _run_tier3_single(
                        _emu, _obs_path,
                        mcmc_steps=mcmc_steps_slider.value,
                        grid_name=_grid_stem,
                        output_dir="exports",
                    )
                    _tier3_results.append(_r)
                    _pf_status = ""
                    if "pf_path" in _r:
                        _pf_status = f" → {os.path.basename(_r['pf_path'])}"
                    _t3_bar.update(
                        increment=1,
                        subtitle=f"✓ {os.path.basename(_obs_path)} complete{_pf_status}",
                    )

        # Store the execution settings alongside the benchmark outputs so a saved
        # report still records which emulator, grid, and limits produced it.
        _config = {
            "emulator": _emu_path,
            "grid": matched_grid_path,
            "test_grid": matched_testgrid_path,
            "tiers": _tiers,
            "mcmc_steps": mcmc_steps_slider.value,
            "max_spectra": max_spectra_slider.value,
        }
        _report = _build_report_card(_tier1_result, _tier2_result, _tier3_results, _config)

        # Save each live run under a timestamped filename to avoid clobbering earlier reports.
        os.makedirs("benchmark_results", exist_ok=True)
        _ts = time.strftime("%Y%m%d_%H%M%S")
        _out_path = f"benchmark_results/benchmark_report_live_{_ts}.json"
        _save_report(_report, _out_path)

        set_report(_report)
        _elapsed = time.time() - _t0

        # Build a short markdown summary for the callout shown below the runner.
        _summary_parts = [f"**Benchmark complete in {_elapsed:.1f}s** — saved to `{_out_path}`"]
        if _tier1_result:
            _eas = _tier1_result.get("emulator_accuracy_score")
            if _eas is not None:
                _summary_parts.append(f"Tier 1 EAS: **{_eas:.2f}%**")
        if _tier2_result:
            _t2_fail = _tier2_result.get('n_failures', 0)
            _t2_nc = _tier2_result.get('n_not_converged', 0)
            _t2_parts = [
                f"Tier 2: {_tier2_result.get('n_processed', 0)}/{_tier2_result.get('n_spectra', '?')} spectra"
            ]
            if _t2_fail:
                _t2_parts.append(f"{_t2_fail} failed")
            if _t2_nc:
                _t2_parts.append(f"{_t2_nc} not converged")
            _summary_parts.append(" — ".join(_t2_parts))
        if _tier3_results:
            _summary_parts.append(f"Tier 3: {len(_tier3_results)} observation(s)")

        set_status_msg("  \n".join(_summary_parts))

    except Exception as _e:
        import traceback as _tb
        # Surface the traceback in the status callout so notebook users can debug
        # failures without opening a separate terminal session.
        set_status_msg(f"Error: {_e}\n```\n{_tb.format_exc()}\n```")
    return


@app.cell(hide_code=True)
def _(get_status_msg, mo):
    _msg = get_status_msg()
    if _msg:
        _kind = "success" if "complete" in _msg.lower() else "danger" if "error" in _msg.lower() else "info"
        mo.output.replace(mo.callout(mo.md(_msg), kind=_kind))
    return


if __name__ == "__main__":
    app.run()
