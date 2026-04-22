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


@app.cell(hide_code=True)
def _(mo):
    usage_toggle = mo.ui.switch(value=False, label=f"{mo.icon('lucide:activity')} System Resources")
    usage_refresh = mo.ui.refresh(default_interval="10s", label="")
    return usage_refresh, usage_toggle


@app.cell(hide_code=True)
def _(mo, usage_refresh, usage_toggle):
    from Speculate_addons.speculate_usage_bars import get_usage_html
    if usage_toggle.value:
        usage_refresh
        _html = get_usage_html()
        usage_bars = mo.vstack([usage_toggle, mo.Html(_html)])
    else:
        usage_bars = usage_toggle
    return (usage_bars,)


@app.cell
def _(mo, os, usage_bars):
    _is_hf = os.environ.get("SPACE_ID") is not None

    _items = [mo.md(f"# Speculate {mo.icon('lucide:telescope')}")]
    _items.extend([mo.md(" "), mo.md("---"), mo.md(" ")])

    if _is_hf:
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
        }, orientation="vertical"))
        _items.extend([
            mo.md(" "),
            mo.md("---"),
            mo.md(f"### {mo.icon('lucide:lock')} Locked Tools:"),
            mo.md("Install Speculate Locally"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:download')} Grid Downloader"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:brain')} Training Tool"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:sparkles')} Inference Tool"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:test-tubes')} Benchmark Suite"),
        ])
    else:
        _items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/downloader": f"###{mo.icon('lucide:download')} Grid Downloader",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/training": f"###{mo.icon('lucide:brain')} Training Tool",
            "/inference": f"###{mo.icon('lucide:sparkles')} Inference Tool",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
            "/benchmark": f"###{mo.icon('lucide:test-tubes')} Benchmark Suite",
        }, orientation="vertical"))

    _items.extend([
        mo.md(" "), mo.md("---"),
        mo.nav_menu({
            "https://github.com/sirocco-rt/speculate": f"###{mo.icon('lucide:github')} Speculate Github",
            "https://github.com/sirocco-rt/speculate/wiki": f"###{mo.icon('lucide:book-open')} Speculate Docs",
        }, orientation="vertical"),
        mo.md(" "), mo.md("---"),
        mo.nav_menu({
            "https://github.com/sirocco-rt/sirocco": f"###{mo.icon('lucide:wind')} Sirocco Github",
            "https://sirocco-rt.readthedocs.io/en/latest/": f"###{mo.icon('lucide:wind')} Sirocco Docs",
        }, orientation="vertical"),
    ])
    _items.extend([mo.md("---"), usage_bars])
    mo.sidebar(mo.vstack(_items))
    return

@app.cell(hide_code=True)
def _():
    # ── Third-party imports ──
    # VegaFusion is enabled so Altair charts degrade gracefully when
    # the dataset exceeds the default 5 000-row inline limit.
    import base64
    import importlib
    import io
    import json
    import os
    import re
    import glob
    import time
    import numpy as np
    import pandas as pd
    import altair as alt
    from Speculate_addons import Spec_functions as spec_functions
    if (
        not hasattr(spec_functions, "build_bestfit_spectrum_altair")
        or not hasattr(spec_functions, "enable_speculate_altair_theme")
    ):
        spec_functions = importlib.reload(spec_functions)
    spec_functions.enable_speculate_altair_theme(alt)
    build_bestfit_spectrum_altair = spec_functions.build_bestfit_spectrum_altair
    alt.data_transformers.enable("vegafusion")
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for server environments
    import matplotlib.pyplot as plt

    def render_fixed_matplotlib(mo, fig, width_px=900, dpi=160, close=True):
        """Render a matplotlib figure at a fixed visual width inside marimo."""
        _buf = io.BytesIO()
        fig.savefig(
            _buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        _data = base64.b64encode(_buf.getvalue()).decode("ascii")
        if close:
            plt.close(fig)
        return mo.Html(
            "<div style='display:flex; justify-content:center; width:100%;'>"
            f"<img src='data:image/png;base64,{_data}' "
            f"style='width:{int(width_px)}px; max-width:100%; height:auto; display:block;' />"
            "</div>"
        )

    return alt, build_bestfit_spectrum_altair, glob, json, np, os, pd, plt, re, render_fixed_matplotlib, time


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
    # tier2_posteriors – list of per-spectrum dicts, each containing MCMC
    # flat samples, labels, ground-truth values, and summary stats so the
    # interactive corner-plot explorer can render any spectrum post-run.
    get_tier2_posteriors, set_tier2_posteriors = mo.state(None)
    return (
        get_comparison_reports,
        get_emulator,
        get_report,
        get_status_msg,
        get_tier1_arrays,
        get_tier2_posteriors,
        set_comparison_reports,
        set_emulator,
        set_report,
        set_status_msg,
        set_tier1_arrays,
        set_tier2_posteriors,
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
    np,
    os,
    report_picker,
    set_comparison_reports,
    set_report,
    set_tier1_arrays,
    set_tier2_posteriors,
    upload_btn,
):
    def _reconstruct_posteriors(report):
        """Rebuild tier2_posteriors list from a loaded report's embedded data."""
        t2 = report.get("tier2", {})
        posteriors_data = t2.get("posteriors", [])
        if not posteriors_data:
            return None
        posteriors = []
        for p in posteriors_data:
            entry = {
                "run": p["run"],
                "filename": p.get("filename", f"run{p['run']}.spec"),
                "inclination": p.get("inclination", 55.0),
                "labels": p.get("labels", []),
                "summary": p.get("summary", {}),
                "converged": p.get("converged", False),
                "truths": p.get("truths", {}),
            }
            if "samples" in p:
                entry["samples"] = np.array(p["samples"])
            if "full_chain" in p:
                entry["full_chain"] = np.array(p["full_chain"])
                entry["burnin_used"] = p.get("burnin_used", 500)
            if "bestfit_spec" in p:
                entry["bestfit_spec"] = p["bestfit_spec"]
            if "prior_ranges" in p:
                entry["prior_ranges"] = p["prior_ranges"]
            posteriors.append(entry)
        return posteriors if posteriors else None

    # Each button mutates notebook state only when clicked; the returned state is
    # then consumed by the rendering cells below.
    if load_btn.value:
        _path = report_picker.value
        if _path and os.path.isfile(_path):
            with open(_path) as _f:
                _loaded = json.load(_f)
            set_report(_loaded)
            # Reports loaded from disk contain summaries only, so clear any stale
            # Tier 1 arrays left behind by a previous live benchmark run.
            set_tier1_arrays(None)
            # Reconstruct posteriors from embedded data (if present)
            set_tier2_posteriors(_reconstruct_posteriors(_loaded))

    if upload_btn.value:
        # Accept the first uploaded JSON file and treat it the same way as a
        # report loaded from the local picker.
        for _f in upload_btn.value:
            _loaded = json.loads(_f.contents.decode())
            set_report(_loaded)
            set_tier1_arrays(None)
            set_tier2_posteriors(_reconstruct_posteriors(_loaded))
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
def _(alt, get_report, get_tier1_arrays, mo, np, plt):
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
            "pca_explained_variance": ("PCA Explained Variance", "higher is better; 1 = 100% = perfect reconstruction"),
            "q2_aggregate": ("Aggregate Q\u00b2 (LOO R\u00b2)", "higher is better; \u22650.80 good for sparse physics grids, \u22650.95 excellent"),
            "nlpd_mean": ("Mean NLPD", "lower is better; compare across configs (drops with k due to shrinking weight scale, not better fit)"),
            "std_resid_mean": ("Std. Residual Mean", "\u22480 ideal; bias if far from 0"),
            "std_resid_var": ("Std. Residual Variance", "\u22481 ideal; >1 \u2192 over-confident (\u03c3 too small); <1 \u2192 under-confident (\u03c3 too large)"),
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
            # This bar chart shows the leave-one-out RMSE for each PCA component.
            # It highlights which PCA weights are hardest for the GP to reconstruct accurately.
            _fig_rmse = alt.Chart(
                alt.Data(
                    values=[
                        {"PCA Component": _component, "LOO RMSE": _rmse}
                        for _component, _rmse in enumerate(_rmses)
                    ]
                )
            ).mark_bar(color="steelblue").encode(
                x=alt.X("PCA Component:O", title="PCA Component"),
                y=alt.Y("LOO RMSE:Q", title="LOO RMSE"),
                tooltip=[
                    alt.Tooltip("PCA Component:O", title="PCA Component"),
                    alt.Tooltip("LOO RMSE:Q", title="LOO RMSE", format=".4e"),
                ],
            ).properties(
                width="container",
                height=320,
                title="Leave-One-Out RMSE",
            )
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
                {"LOO RMSE per Component": mo.hstack([_fig_rmse, _rmse_info], widths=[3, 1])}
            ))

        # --- Q² per component ---
        if _q2s:
            _colors = ["#2ecc71" if v >= 0.90 else "#f39c12" if v >= 0.70 else "#e74c3c" for v in _q2s]

            # This bar chart shows the leave-one-out predictive Q² score for each PCA component.
            # It highlights which components are well predicted by the GP and which ones are degrading.
            _q2_values = [
                {"PCA Component": _component, "Q2": _q2, "BarColor": _color}
                for _component, (_q2, _color) in enumerate(zip(_q2s, _colors))
            ]
            _q2_rule = alt.Chart(alt.Data(values=[{"Reference": 1.0}])).mark_rule(
                color="grey",
                strokeDash=[6, 4],
                strokeWidth=1.2,
            ).encode(y="Reference:Q")
            _q2_bars = alt.Chart(
                alt.Data(values=_q2_values)
            ).mark_bar().encode(
                x=alt.X("PCA Component:O", title="PCA Component"),
                y=alt.Y("Q2:Q", title="Q²"),
                color=alt.Color("BarColor:N", scale=None, legend=None),
                tooltip=[
                    alt.Tooltip("PCA Component:O", title="PCA Component"),
                    alt.Tooltip("Q2:Q", title="Q²", format=".4f"),
                ],
            )
            _fig_q2 = (_q2_rule + _q2_bars).properties(
                width="container",
                height=320,
                title="Per-Component Q² (LOO R²)",
            )
            _q2_info = mo.md(
                "### Per-Component Q\u00b2\n\n"
                "**What this shows:** The leave-one-out predictive R\u00b2 for each PCA "
                "component. Bars are colour-coded: green (\u22650.90), amber (\u22650.70), "
                "red (<0.70).\n\n"
                "**Better looks like:** All bars green and close to 1.0.\n\n"
                "**Watch out for:** Red or amber bars, especially on the leading "
                "components which carry the most spectral variance.\n\n"
                "**How to improve:** More training points generally raise Q\u00b2. "
                "If only trailing components are red, consider reducing the PCA rank."
            )
            _t1_items.append(mo.accordion(
                {"Per-Component Q\u00b2": mo.hstack([_fig_q2, _q2_info], widths=[3, 1])}
            ))

        # --- NLPD per component ---
        if _nlpds:
            # This bar chart shows the leave-one-out negative log predictive density for each PCA component.
            # It highlights which components have the poorest combined accuracy and uncertainty calibration.
            _fig_nlpd = alt.Chart(
                alt.Data(
                    values=[
                        {"PCA Component": _component, "NLPD": _nlpd}
                        for _component, _nlpd in enumerate(_nlpds)
                    ]
                )
            ).mark_bar(color="darkorange").encode(
                x=alt.X("PCA Component:O", title="PCA Component"),
                y=alt.Y("NLPD:Q", title="NLPD"),
                tooltip=[
                    alt.Tooltip("PCA Component:O", title="PCA Component"),
                    alt.Tooltip("NLPD:Q", title="NLPD", format=".4f"),
                ],
            ).properties(
                width="container",
                height=320,
                title="LOO Neg. Log Predictive Density",
            )
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
                {"NLPD per Component": mo.hstack([_fig_nlpd, _nlpd_info], widths=[3, 1])}
            ))

        # --- LOO residual distribution ---
        _std_resid = _t1.get("loo_std_resid", [])
        if _std_resid:
            _flat = np.array(_std_resid).flatten()

            # This histogram shows the distribution of standardised LOO residuals across all PCA components.
            # It compares the empirical residual density against the ideal N(0,1) Gaussian reference curve.
            # Clip the displayed x-range to the central residual bulk so extreme outliers do not squash the plot.
            _display_limit = max(4.0, min(6.0, float(np.nanpercentile(np.abs(_flat), 99.5))))
            _hist_density, _hist_edges = np.histogram(
                _flat,
                bins=50,
                range=(-_display_limit, _display_limit),
                density=True,
            )
            _hist_values = [
                {
                    "Bin Start": float(_left),
                    "Bin End": float(_right),
                    "Density": float(_density),
                    "Baseline": 0.0,
                }
                for _left, _right, _density in zip(_hist_edges[:-1], _hist_edges[1:], _hist_density)
            ]
            _xgauss = np.linspace(-_display_limit, _display_limit, 400)
            _gaussian_values = [
                {
                    "Residual": float(_x),
                    "Density": float((1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * _x**2)),
                    "Series": "N(0,1)",
                }
                for _x in _xgauss
            ]
            _hist_bars = alt.Chart(alt.Data(values=_hist_values)).mark_rect(
                color="steelblue",
                opacity=0.7,
            ).encode(
                x=alt.X(
                    "Bin Start:Q",
                    title="Standardised Residual",
                    scale=alt.Scale(domain=[-_display_limit, _display_limit]),
                ),
                x2="Bin End:Q",
                y=alt.Y("Density:Q", title="Density"),
                y2="Baseline:Q",
                tooltip=[
                    alt.Tooltip("Bin Start:Q", title="Bin Start", format=".3f"),
                    alt.Tooltip("Bin End:Q", title="Bin End", format=".3f"),
                    alt.Tooltip("Density:Q", title="Density", format=".4f"),
                ],
            )
            _hist_gaussian = alt.Chart(alt.Data(values=_gaussian_values)).mark_line(
                color="red",
                strokeWidth=2,
            ).encode(
                x=alt.X("Residual:Q", scale=alt.Scale(domain=[-_display_limit, _display_limit])),
                y=alt.Y("Density:Q"),
                tooltip=[
                    alt.Tooltip("Residual:Q", title="Residual", format=".3f"),
                    alt.Tooltip("Density:Q", title="N(0,1)", format=".4f"),
                ],
            )
            _fig_hist = (_hist_bars + _hist_gaussian).properties(
                width="container",
                height=320,
                title="LOO Residual Distribution",
            )
            _hist_info = mo.md(
                "### Residual Distribution\n\n"
                "**What this shows:** Histogram of standardised LOO residuals across all "
                "components, overlaid with the expected N(0,1) Gaussian (red curve).\n\n"
                "**Better looks like:** The histogram closely tracks the red curve — "
                "symmetric, centred on zero, with the same width.\n\n"
                "**Watch out for:** Heavy tails (the GP is over-confident \u2014 \u03c3 too small), "
                "a shifted centre (systematic bias), or a narrower peak than N(0,1) "
                "(under-confident \u2014 \u03c3 too large).\n\n"
                "**How to improve:** Bias \u2192 check for a bug in the mean function. "
                "Heavy tails \u2192 add training points or try a different optimiser."
            )
            _t1_items.append(mo.accordion(
                {"LOO Residual Distribution": mo.hstack([_fig_hist, _hist_info], widths=[3, 1])}
            ))

        # --- Q-Q plot of standardised residuals ---
        _std_resid = _t1.get("loo_std_resid", [])
        if _std_resid:
            from scipy import stats as _sp_stats
            _flat = np.sort(np.array(_std_resid).flatten())
            _n_pts = len(_flat)
            _theoretical = _sp_stats.norm.ppf(
                (np.arange(1, _n_pts + 1) - 0.5) / _n_pts
            )
            _qq_lim = max(abs(_theoretical[0]), abs(_theoretical[-1]), abs(_flat[0]), abs(_flat[-1]))

            # This scatter plot compares empirical residual quantiles against standard normal quantiles.
            # It shows whether the standardised LOO residuals follow the expected N(0,1) distribution.
            _qq_values = [
                {
                    "Theoretical Quantile": float(_theory),
                    "Sample Quantile": float(_sample),
                }
                for _theory, _sample in zip(_theoretical, _flat)
            ]
            _qq_reference = [
                {"Theoretical Quantile": float(-_qq_lim), "Reference": float(-_qq_lim)},
                {"Theoretical Quantile": float(_qq_lim), "Reference": float(_qq_lim)},
            ]
            _qq_line = alt.Chart(alt.Data(values=_qq_reference)).mark_line(
                color="red",
                strokeWidth=1.5,
            ).encode(
                x=alt.X(
                    "Theoretical Quantile:Q",
                    title="Theoretical Quantiles (N(0,1))",
                    scale=alt.Scale(domain=[-_qq_lim, _qq_lim]),
                ),
                y=alt.Y(
                    "Reference:Q",
                    title="Sample Quantiles",
                    scale=alt.Scale(domain=[-_qq_lim, _qq_lim]),
                ),
            )
            _qq_points = alt.Chart(alt.Data(values=_qq_values)).mark_circle(
                color="steelblue",
                opacity=0.4,
                size=14,
            ).encode(
                x=alt.X(
                    "Theoretical Quantile:Q",
                    title="Theoretical Quantiles (N(0,1))",
                    scale=alt.Scale(domain=[-_qq_lim, _qq_lim]),
                ),
                y=alt.Y(
                    "Sample Quantile:Q",
                    title="Sample Quantiles",
                    scale=alt.Scale(domain=[-_qq_lim, _qq_lim]),
                ),
                tooltip=[
                    alt.Tooltip("Theoretical Quantile:Q", title="Theoretical", format=".3f"),
                    alt.Tooltip("Sample Quantile:Q", title="Sample", format=".3f"),
                ],
            )
            _fig_qq = (_qq_line + _qq_points).properties(
                width=400,
                height=400,
                title="Q\u2013Q Plot of Standardised LOO Residuals",
            ).interactive()
            _qq_info = mo.md(
                "### Q\u2013Q Plot\n\n"
                "**What this shows:** Compares the quantiles of the standardised LOO residuals "
                "against a standard normal distribution. If the GP uncertainty estimates are "
                "well-calibrated the points should lie on the red diagonal.\n\n"
                "**Better looks like:** All points hugging the 1:1 red line with minimal scatter.\n\n"
                "**Watch out for:** An S-shape or slope > 1 (heavy tails \u2014 GP is over-confident, "
                "\u03c3 too small), a banana curve (skew), or systematic departure from the diagonal. "
                "Note: marginal standardised residuals are correlated, which can induce apparent "
                "heavy tails even in a well-calibrated GP (Bastos & O'Hagan 2009). See the "
                "decorrelated Q\u2013Q panel below for a cleaner diagnostic.\n\n"
                "**How to improve:** Heavy tails suggest the GP variance is too small for some "
                "predictions \u2014 try adding training points or a different kernel/optimiser."
            )
            _t1_items.append(mo.accordion(
                {"Q\u2013Q Plot": mo.hstack([_fig_qq, _qq_info], widths=[2, 1])}
            ))

        # --- Decorrelated Q-Q plot (Bastos & O'Hagan 2009) ---
        _decorr_resid = _t1.get("loo_decorr_resid", [])
        if _decorr_resid:
            from scipy import stats as _sp_stats
            _dflat = np.sort(np.array(_decorr_resid).flatten())
            _dn_pts = len(_dflat)
            _dtheoretical = _sp_stats.norm.ppf(
                (np.arange(1, _dn_pts + 1) - 0.5) / _dn_pts
            )
            _dqq_lim = max(abs(_dtheoretical[0]), abs(_dtheoretical[-1]),
                           abs(_dflat[0]), abs(_dflat[-1]))

            # This scatter plot compares decorrelated residual quantiles against standard normal quantiles.
            # It shows whether the Cholesky-decorrelated LOO residuals follow the expected N(0,1) distribution.
            _dqq_values = [
                {
                    "Theoretical Quantile": float(_theory),
                    "Decorrelated Sample Quantile": float(_sample),
                }
                for _theory, _sample in zip(_dtheoretical, _dflat)
            ]
            _dqq_reference = [
                {"Theoretical Quantile": float(-_dqq_lim), "Reference": float(-_dqq_lim)},
                {"Theoretical Quantile": float(_dqq_lim), "Reference": float(_dqq_lim)},
            ]
            _dqq_line = alt.Chart(alt.Data(values=_dqq_reference)).mark_line(
                color="red",
                strokeWidth=1.5,
            ).encode(
                x=alt.X(
                    "Theoretical Quantile:Q",
                    title="Theoretical Quantiles (N(0,1))",
                    scale=alt.Scale(domain=[-_dqq_lim, _dqq_lim]),
                ),
                y=alt.Y(
                    "Reference:Q",
                    title="Decorrelated Sample Quantiles",
                    scale=alt.Scale(domain=[-_dqq_lim, _dqq_lim]),
                ),
            )
            _dqq_points = alt.Chart(alt.Data(values=_dqq_values)).mark_circle(
                color="darkorange",
                opacity=0.4,
                size=14,
            ).encode(
                x=alt.X(
                    "Theoretical Quantile:Q",
                    title="Theoretical Quantiles (N(0,1))",
                    scale=alt.Scale(domain=[-_dqq_lim, _dqq_lim]),
                ),
                y=alt.Y(
                    "Decorrelated Sample Quantile:Q",
                    title="Decorrelated Sample Quantiles",
                    scale=alt.Scale(domain=[-_dqq_lim, _dqq_lim]),
                ),
                tooltip=[
                    alt.Tooltip("Theoretical Quantile:Q", title="Theoretical", format=".3f"),
                    alt.Tooltip("Decorrelated Sample Quantile:Q", title="Decorrelated Sample", format=".3f"),
                ],
            )
            _fig_dqq = (_dqq_line + _dqq_points).properties(
                width=400,
                height=400,
                title="Decorrelated Q\u2013Q Plot (Bastos & O\u2019Hagan 2009)",
            ).interactive()
            _dqq_info = mo.md(
                "### Decorrelated Q\u2013Q Plot\n\n"
                "**What this shows:** Quantiles of Cholesky-decorrelated LOO residuals "
                "against N(0,1). Unlike the marginal Q\u2013Q above, this removes the "
                "correlation between LOO predictions (Bastos & O\u2019Hagan 2009 \u00a74), "
                "giving a correctly interpretable diagnostic.\n\n"
                "**Better looks like:** Points on the 1:1 red line.\n\n"
                "**Watch out for:** Slope > 1 \u2192 GP is over-confident (\u03c3 too small). "
                "Slope < 1 \u2192 GP is under-confident (\u03c3 too large). "
                "S-shape \u2192 heavy-tailed or non-Gaussian residuals.\n\n"
                "**How to improve:** Systematic slope \u2260 1 \u2192 retrain with different "
                "kernel or optimiser. Localised outliers \u2192 add training points."
            )
            _t1_items.append(mo.accordion(
                {"Decorrelated Q\u2013Q Plot": mo.hstack([_fig_dqq, _dqq_info], widths=[2, 1])}
            ))

        # --- Per-wavelength RMSE envelope ---
        # Source data from live arrays (if available) or serialised report.
        _arrays = get_tier1_arrays() or {}
        _pca_wl = _arrays.get("pca_per_wl_rmse")
        _loo_wl = _arrays.get("loo_per_wl_rmse")
        _wl = _arrays.get("wavelength")
        # Fall back to serialised lists in the loaded report
        if _pca_wl is None:
            _pca_wl = _t1.get("pca_per_wl_rmse")
        if _loo_wl is None:
            _loo_wl = _t1.get("loo_per_wl_rmse")
        if _wl is None:
            _wl = _t1.get("wavelength")
        # Convert to numpy for consistency
        if _pca_wl is not None:
            _pca_wl = np.asarray(_pca_wl)
        if _loo_wl is not None:
            _loo_wl = np.asarray(_loo_wl)
        if _wl is not None:
            _wl = np.asarray(_wl)

        if _pca_wl is not None and _wl is not None:
            _model_label = "LOO (PCA + GP)"
            _pca_df = pd.DataFrame({"Wavelength": _wl, "RMSE": _pca_wl, "Source": "PCA truncation only"})
            _rmse_frames = [_pca_df]
            if _loo_wl is not None:
                _total_df = pd.DataFrame({"Wavelength": _wl, "RMSE": _loo_wl, "Source": _model_label})
                _rmse_frames.append(_total_df)
            else:
                _model_label = "PCA truncation only"   # only one line
            _rmse_df = pd.concat(_rmse_frames, ignore_index=True)

            _rmse_chart = alt.Chart(_rmse_df).mark_line(
                strokeWidth=1.5,
            ).encode(
                x=alt.X("Wavelength:Q", title="Wavelength (Å)",
                         scale=alt.Scale(domain=[float(_wl.min()), float(_wl.max())])),
                y=alt.Y("RMSE:Q", title="RMSE (normalised flux)", axis=alt.Axis(format=".1e")),
                color=alt.Color("Source:N", title="",
                                scale=alt.Scale(
                                    domain=["PCA truncation only", _model_label],
                                    range=["#3498db", "#e74c3c"]),
                                legend=alt.Legend(orient="top")),
                tooltip=[
                    alt.Tooltip("Source:N"),
                    alt.Tooltip("Wavelength:Q", title="Wavelength (Å)", format=".1f"),
                    alt.Tooltip("RMSE:Q", format=".4e"),
                ],
            ).properties(
                width="container", height=200,
                title="Per-Wavelength Reconstruction Error"
            )
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
                {"Per-Wavelength RMSE Envelope": mo.hstack([_rmse_chart, _wl_info], widths=[3, 1])}
            ))

        # --- Worst-case spectra overlay ---
        _orig = _arrays.get("display_original_flux", _arrays.get("original_flux"))
        _pca_rec = _arrays.get("display_pca_recon_flux", _arrays.get("pca_recon_flux"))
        _loo_rec = _arrays.get("display_loo_recon_flux", _arrays.get("loo_recon_flux"))
        _flux_axis_title = _arrays.get("flux_axis_title", "Flux")
        if _orig is not None and _loo_rec is not None and _wl is not None:
            _loo_rmse_arr = _arrays.get("loo_flux_rmse")
            if _loo_rmse_arr is None:
                _loo_rmse_arr = _t1.get("loo_flux_rmse", [])
            if hasattr(_loo_rmse_arr, '__len__') and len(_loo_rmse_arr) > 0:
                _sorted_idx = np.argsort(_loo_rmse_arr)
                _worst_idx = _sorted_idx[-3:][::-1]
                _series_domain = ["Original", "LOO recon"]
                _series_colors = ["#f1c40f", "#e74c3c"]
                _series_dashes = [[1, 0], [6, 4]]
                if _pca_rec is not None:
                    _series_domain.insert(1, "PCA recon")
                    _series_colors.insert(1, "#3498db")
                    _series_dashes.insert(1, [4, 4])

                # These stacked line charts show the three worst-reconstructed spectra in Tier 1.
                # Each panel overlays the original spectrum with the PCA-only and full LOO reconstructions.
                _worst_charts = []
                for _wi, _idx in enumerate(_worst_idx):
                    _rmse_text = (
                        f"{float(_loo_rmse_arr[_idx]):.5f}"
                        if hasattr(_loo_rmse_arr, '__getitem__')
                        else "?"
                    )
                    _worst_values = []
                    for _wavelength, _flux in zip(_wl, _orig[_idx]):
                        _worst_values.append({
                            "Wavelength": float(_wavelength),
                            "Flux": float(_flux),
                            "Series": "Original",
                        })
                    if _pca_rec is not None:
                        for _wavelength, _flux in zip(_wl, _pca_rec[_idx]):
                            _worst_values.append({
                                "Wavelength": float(_wavelength),
                                "Flux": float(_flux),
                                "Series": "PCA recon",
                            })
                    for _wavelength, _flux in zip(_wl, _loo_rec[_idx]):
                        _worst_values.append({
                            "Wavelength": float(_wavelength),
                            "Flux": float(_flux),
                            "Series": "LOO recon",
                        })

                    _worst_chart = alt.Chart(alt.Data(values=_worst_values)).mark_line(
                        strokeWidth=1.5,
                    ).encode(
                        x=alt.X(
                            "Wavelength:Q",
                            title="Wavelength (Å)",
                            scale=alt.Scale(domain=[float(np.min(_wl)), float(np.max(_wl))]),
                        ),
                        y=alt.Y(
                            "Flux:Q",
                            title=_flux_axis_title,
                            scale=alt.Scale(zero=False),
                        ),
                        color=alt.Color(
                            "Series:N",
                            title="Series",
                            scale=alt.Scale(domain=_series_domain, range=_series_colors),
                            legend=alt.Legend(orient="top") if _wi == 0 else None,
                        ),
                        strokeDash=alt.StrokeDash(
                            "Series:N",
                            scale=alt.Scale(domain=_series_domain, range=_series_dashes),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Wavelength:Q", title="Wavelength (Å)", format=".1f"),
                            alt.Tooltip("Flux:Q", title=_flux_axis_title, format=".4e"),
                        ],
                    ).properties(
                        width="container",
                        height=180,
                        title=f"Spectrum #{_idx} (LOO RMSE = {_rmse_text})",
                    )
                    _worst_charts.append(_worst_chart)

                _fig_worst = alt.vconcat(*_worst_charts, spacing=12).resolve_scale(
                    y="independent"
                ).interactive()
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
                    {"Worst-Case Spectra Overlay": mo.hstack([_fig_worst, _worst_info], widths=[3, 1])}
                ))

                # --- Best-case spectra overlay ---
                _best_idx = _sorted_idx[:3]
                # These stacked line charts show the three best-reconstructed spectra in Tier 1.
                # Each panel shows how closely the PCA-only and LOO reconstructions track the original spectrum.
                _best_charts = []
                for _bi, _idx in enumerate(_best_idx):
                    _rmse_text = (
                        f"{float(_loo_rmse_arr[_idx]):.5f}"
                        if hasattr(_loo_rmse_arr, '__getitem__')
                        else "?"
                    )
                    _best_values = []
                    for _wavelength, _flux in zip(_wl, _orig[_idx]):
                        _best_values.append({
                            "Wavelength": float(_wavelength),
                            "Flux": float(_flux),
                            "Series": "Original",
                        })
                    if _pca_rec is not None:
                        for _wavelength, _flux in zip(_wl, _pca_rec[_idx]):
                            _best_values.append({
                                "Wavelength": float(_wavelength),
                                "Flux": float(_flux),
                                "Series": "PCA recon",
                            })
                    for _wavelength, _flux in zip(_wl, _loo_rec[_idx]):
                        _best_values.append({
                            "Wavelength": float(_wavelength),
                            "Flux": float(_flux),
                            "Series": "LOO recon",
                        })

                    _best_chart = alt.Chart(alt.Data(values=_best_values)).mark_line(
                        strokeWidth=1.5,
                    ).encode(
                        x=alt.X(
                            "Wavelength:Q",
                            title="Wavelength (Å)",
                            scale=alt.Scale(domain=[float(np.min(_wl)), float(np.max(_wl))]),
                        ),
                        y=alt.Y(
                            "Flux:Q",
                            title=_flux_axis_title,
                            scale=alt.Scale(zero=False),
                        ),
                        color=alt.Color(
                            "Series:N",
                            title="Series",
                            scale=alt.Scale(domain=_series_domain, range=_series_colors),
                            legend=alt.Legend(orient="top") if _bi == 0 else None,
                        ),
                        strokeDash=alt.StrokeDash(
                            "Series:N",
                            scale=alt.Scale(domain=_series_domain, range=_series_dashes),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Series:N", title="Series"),
                            alt.Tooltip("Wavelength:Q", title="Wavelength (Å)", format=".1f"),
                            alt.Tooltip("Flux:Q", title=_flux_axis_title, format=".4e"),
                        ],
                    ).properties(
                        width="container",
                        height=180,
                        title=f"Spectrum #{_idx} (LOO RMSE = {_rmse_text})",
                    )
                    _best_charts.append(_best_chart)

                _fig_best = alt.vconcat(*_best_charts, spacing=12).resolve_scale(
                    y="independent"
                ).interactive()
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
                    {"Best-Case Spectra Overlay": mo.hstack([_fig_best, _best_info], widths=[3, 1])}
                ))

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
                    "Cov@68%": f"{_m.get('coverage_68', float('nan')):.3f}",
                    "Cov@95%": f"{_m.get('coverage_95', float('nan')):.3f}",
                    "Cov@99.7%": f"{_m.get('coverage_997', float('nan')):.3f}",
                })
            _t2_items.append(mo.ui.table(_agg_rows, label="Aggregate Metrics"))

            # This calibration plot compares nominal credible levels against empirical coverage.
            # Points should fall near the y=x diagonal if the posterior intervals are well calibrated.
            _pp_values = [
                {
                    "Parameter": _pn,
                    "Nominal Credible Level": float(_alpha),
                    "Empirical Coverage": float(_coverage),
                }
                for _pn, _m in _agg.items()
                if "coverage_alphas" in _m and "coverage_values" in _m
                for _alpha, _coverage in zip(_m["coverage_alphas"], _m["coverage_values"])
            ]
            _pp_reference = [
                {"Nominal Credible Level": 0.0, "Ideal Coverage": 0.0},
                {"Nominal Credible Level": 1.0, "Ideal Coverage": 1.0},
            ]
            _pp_reference_line = alt.Chart(alt.Data(values=_pp_reference)).mark_line(
                color="grey",
                strokeDash=[6, 4],
                strokeWidth=1.5,
                opacity=0.8,
            ).encode(
                x=alt.X(
                    "Nominal Credible Level:Q",
                    title="Nominal Credible Level",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "Ideal Coverage:Q",
                    title="Empirical Coverage",
                    scale=alt.Scale(domain=[0, 1]),
                ),
            )
            _pp_lines = alt.Chart(alt.Data(values=_pp_values)).mark_line(strokeWidth=2).encode(
                x=alt.X(
                    "Nominal Credible Level:Q",
                    title="Nominal Credible Level",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "Empirical Coverage:Q",
                    title="Empirical Coverage",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color("Parameter:N", title="", legend=alt.Legend(orient="top-left")),
            )
            _pp_points = alt.Chart(alt.Data(values=_pp_values)).mark_circle(size=40).encode(
                x=alt.X(
                    "Nominal Credible Level:Q",
                    title="Nominal Credible Level",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "Empirical Coverage:Q",
                    title="Empirical Coverage",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color("Parameter:N", title="", legend=None),
                tooltip=[
                    alt.Tooltip("Parameter:N", title="Parameter"),
                    alt.Tooltip("Nominal Credible Level:Q", title="Nominal", format=".3f"),
                    alt.Tooltip("Empirical Coverage:Q", title="Empirical", format=".3f"),
                ],
            )
            _fig2 = (_pp_reference_line + _pp_lines + _pp_points).properties(
                width=420,
                height=420,
                title="PP-Plot (Calibration)",
            ).interactive()
            _pp_info = mo.md(
                "### PP-Plot (Calibration)\n\n"
                "**What this shows:** For each parameter, the chart evaluates 50 nominal "
                "credible levels between 0.01 and 0.999. Each point is the empirical "
                "fraction of processed spectra whose ground-truth value lies inside that "
                "posterior interval.\n\n"
                "**Better looks like:** Curves track the dashed y=x line across the full "
                "range of nominal credible levels.\n\n"
                "**Why the step pattern:** Coverage is computed over a finite number of test "
                "spectra, so it can only change in jumps of 1/n_processed. For example, if "
                "10 spectra were processed, the vertical jumps are 0.1.\n\n"
                "**How to read the endpoints:** A curve reaching 1.0 at high nominal "
                "coverage means every processed truth falls inside the widest tested "
                "interval. That does not imply the 68% or 95% summary coverages are also 1.0.\n\n"
                "**How to improve:** Curves below the diagonal indicate over-confident "
                "intervals; curves above the diagonal indicate under-confident intervals."
            )
            _t2_items.append(mo.accordion(
                {"PP-Plot (Calibration)": mo.hstack([_fig2, _pp_info], widths=[2, 1])}
            ))

            _param_names = list(_agg.keys())

            # This paired bar chart shows the aggregate recovery RMSE and signed bias for each Tier 2 parameter.
            # It highlights which parameters are least accurate overall and whether the errors are systematically high or low.
            _param_metric_values = [
                {
                    "Parameter": _param,
                    "RMSE": float(_agg[_param]["rmse"]),
                    "Bias": float(_agg[_param]["bias"]),
                }
                for _param in _param_names
            ]
            _fig3_rmse = alt.Chart(alt.Data(values=_param_metric_values)).mark_bar(
                color="steelblue",
            ).encode(
                x=alt.X(
                    "Parameter:N",
                    title="Parameter",
                    sort=_param_names,
                    axis=alt.Axis(labelAngle=-35, labelLimit=220),
                ),
                y=alt.Y("RMSE:Q", title="RMSE"),
                tooltip=[
                    alt.Tooltip("Parameter:N", title="Parameter"),
                    alt.Tooltip("RMSE:Q", title="RMSE", format=".4f"),
                ],
            ).properties(
                width=280,
                height=320,
                title="Parameter RMSE",
            )
            _fig3_bias_rule = alt.Chart(alt.Data(values=[{"Reference": 0.0}])).mark_rule(
                color="grey",
                strokeWidth=1.2,
            ).encode(y="Reference:Q")
            _fig3_bias_bars = alt.Chart(alt.Data(values=_param_metric_values)).mark_bar(
                color="salmon",
            ).encode(
                x=alt.X(
                    "Parameter:N",
                    title="Parameter",
                    sort=_param_names,
                    axis=alt.Axis(labelAngle=-35, labelLimit=220),
                ),
                y=alt.Y("Bias:Q", title="Bias"),
                tooltip=[
                    alt.Tooltip("Parameter:N", title="Parameter"),
                    alt.Tooltip("Bias:Q", title="Bias", format=".4f"),
                ],
            )
            _fig3 = (_fig3_rmse | (_fig3_bias_rule + _fig3_bias_bars).properties(
                width=280,
                height=320,
                title="Parameter Bias",
            )).resolve_scale(y="independent")
            _fig3_info = mo.md(
                "### Parameter RMSE and Bias\n\n"
                "**What this shows:** The left panel shows the aggregate recovery RMSE for "
                "each Tier 2 parameter, while the right panel shows the signed mean bias. "
                "Use these charts to compare runs for the same parameter; absolute RMSE "
                "heights are not directly comparable across parameters with different scales.\n\n"
                "**Better looks like:** Short RMSE bars and bias bars clustered around the "
                "zero reference line.\n\n"
                "**Watch out for:** Large RMSE with small bias means the posteriors are noisy "
                "or broad but not systematically shifted. Large positive or negative bias means "
                "the inference is systematically overshooting or undershooting that parameter.\n\n"
                "**How to improve:** Persistent bias usually points to emulator mismatch, model "
                "degeneracy, or priors pushing the fit. High RMSE without strong bias often "
                "improves with more informative spectra, tighter priors, or more reliable "
                "posterior sampling."
            )
            _t2_items.append(mo.accordion(
                {"Parameter RMSE and Bias": mo.hstack([_fig3, _fig3_info], widths=[3, 1])}
            ))

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

            # This paired bar chart shows reduced chi2 and posterior predictive coverage for each observation.
            # It highlights which observed spectra are fit well and which ones fail basic calibration checks.
            _obs_metric_values = [
                {
                    "Observation": _name,
                    "Reduced chi2": float(_row.get("reduced_chi2", float("nan"))),
                    "PPC Coverage": float(_row.get("ppc_coverage", float("nan"))),
                }
                for _name, _row in zip(_obs_names, _t3)
            ]
            _obs_chart_width = max(280, min(700, 40 * len(_obs_names)))
            _fig4_chi2_rule = alt.Chart(alt.Data(values=[{"Reference": 1.0}])).mark_rule(
                color="red",
                strokeDash=[6, 4],
                strokeWidth=1.2,
            ).encode(y="Reference:Q")
            _fig4_chi2_bars = alt.Chart(alt.Data(values=_obs_metric_values)).mark_bar(
                color="steelblue",
            ).encode(
                x=alt.X(
                    "Observation:N",
                    title="Observation",
                    sort=_obs_names,
                    axis=alt.Axis(labelAngle=-35, labelLimit=240),
                ),
                y=alt.Y("Reduced chi2:Q", title="Reduced chi2"),
                tooltip=[
                    alt.Tooltip("Observation:N", title="Observation"),
                    alt.Tooltip("Reduced chi2:Q", title="Reduced chi2", format=".3f"),
                ],
            )
            _fig4_ppc_rule = alt.Chart(alt.Data(values=[{"Reference": 0.95}])).mark_rule(
                color="red",
                strokeDash=[6, 4],
                strokeWidth=1.2,
            ).encode(y="Reference:Q")
            _fig4_ppc_bars = alt.Chart(alt.Data(values=_obs_metric_values)).mark_bar(
                color="mediumseagreen",
            ).encode(
                x=alt.X(
                    "Observation:N",
                    title="Observation",
                    sort=_obs_names,
                    axis=alt.Axis(labelAngle=-35, labelLimit=240),
                ),
                y=alt.Y("PPC Coverage:Q", title="PPC Coverage"),
                tooltip=[
                    alt.Tooltip("Observation:N", title="Observation"),
                    alt.Tooltip("PPC Coverage:Q", title="PPC Coverage", format=".3f"),
                ],
            )
            _fig4 = ((_fig4_chi2_rule + _fig4_chi2_bars).properties(
                width=_obs_chart_width,
                height=320,
                title="Reduced chi2",
            ) | (_fig4_ppc_rule + _fig4_ppc_bars).properties(
                width=_obs_chart_width,
                height=320,
                title="Posterior Predictive Coverage",
            )).resolve_scale(y="independent").interactive()
            _t3_items.append(_fig4)
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
def _(get_tier2_posteriors, mo):
    _posteriors = get_tier2_posteriors()
    mo.stop(
        _posteriors is None or len(_posteriors) == 0,
        mo.md("*Run a Tier 2 benchmark to view per-spectrum corner plots.*"),
    )
    _n = len(_posteriors)
    t2_spectrum_slider = mo.ui.slider(
        start=0, stop=_n - 1, value=0, step=1, show_value=True,
        label="Test spectrum index", full_width=True,
    )
    mo.vstack([
        mo.md("## Tier 2 — Per-Spectrum Posterior Explorer"),
        mo.md(f"Browse **{_n}** completed spectra. "
               "Each corner plot shows the MCMC posterior with ground-truth values (blue lines)."),
        t2_spectrum_slider,
    ])
    return (t2_spectrum_slider,)


@app.cell(hide_code=True)
def _(get_tier2_posteriors, mo, np, plt, render_fixed_matplotlib, t2_spectrum_slider):
    import corner as _corner

    _posteriors = get_tier2_posteriors()
    mo.stop(_posteriors is None or len(_posteriors) == 0)

    _idx = t2_spectrum_slider.value
    _post = _posteriors[_idx]
    _samples = _post["samples"]       # (N, ndim)
    _labels = _post["labels"]         # list of friendly labels
    _summary = _post["summary"]       # dict of {label: {mean, std, median, ...}}
    _gt = _post["truths"]             # dict of {friendly_name: value}
    _run = _post["run"]
    _inc = _post["inclination"]
    _converged = _post["converged"]
    _filename = _post.get("filename", f"run{_run}.spec")

    # Build truth vector aligned to sample columns
    _truths = []
    _has_any_truth = False
    for _lbl in _labels:
        _gt_key = _lbl
        # Collapse inclination variants onto a common key
        if "Inclination" in _lbl:
            _gt_key = "Inclination"
        if _gt_key in _gt:
            _truths.append(_gt[_gt_key])
            _has_any_truth = True
        else:
            _truths.append(None)

    _fig = _corner.corner(
        _samples,
        labels=_labels,
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.6827, 0.9545, 0.9973],
        title_fmt=".4f",
        truths=_truths if _has_any_truth else None,
        truth_color="#ff4444",
        truth_kwargs={"linewidth": 2},
    )

    # Prior-range corner plot: axes span the full emulator grid bounds so
    # the user can judge posterior breadth relative to the full prior.
    _pr_dict = _post.get("prior_ranges", {})
    if _pr_dict:
        _pr_list = []
        for _lbl in _labels:
            _key = "Inclination" if "Inclination" in _lbl else _lbl
            if _key in _pr_dict:
                _pr_list.append(tuple(_pr_dict[_key]))
            else:
                _pr_list.append((1.0,))  # auto-range sentinel for corner
        _fig_prior = _corner.corner(
            _samples,
            labels=_labels,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            levels=[0.6827, 0.9545, 0.9973],
            title_fmt=".4f",
            truths=_truths if _has_any_truth else None,
            truth_color="#ff4444",
            truth_kwargs={"linewidth": 2},
            range=_pr_list,
        )
        _corner_display = mo.ui.tabs({
            "Posterior (Auto Range)": render_fixed_matplotlib(mo, _fig, width_px=920),
            "Full Prior Range": render_fixed_matplotlib(mo, _fig_prior, width_px=920),
        })
    else:
        _corner_display = render_fixed_matplotlib(mo, _fig, width_px=920)

    _conv_tag = "converged" if _converged else "**not converged**"
    _summary_rows = []
    for _lbl in _labels:
        if _lbl in _summary:
            _s = _summary[_lbl]
            _row = {
                "Parameter": _lbl,
                "Mean": f"{_s['mean']:.4f}",
                "Std": f"{_s['std']:.4f}",
                "Median": f"{_s['median']:.4f}",
                "HDI 3%": f"{_s['hdi_3']:.4f}",
                "HDI 97%": f"{_s['hdi_97']:.4f}",
            }
            _gt_key = "Inclination" if "Inclination" in _lbl else _lbl
            if _gt_key in _gt:
                _row["Truth"] = f"{_gt[_gt_key]:.4f}"
                _row["Δ/σ"] = f"{(_s['mean'] - _gt[_gt_key]) / max(_s['std'], 1e-10):.2f}"
            else:
                _row["Truth"] = "—"
                _row["Δ/σ"] = "—"
            _summary_rows.append(_row)

    mo.vstack([
        mo.md(f"### {_filename} @ {_inc:.0f}° — {_conv_tag}"),
        mo.ui.table(_summary_rows, label="Posterior Summary", selection=None),
        _corner_display,
    ])
    return


@app.cell(hide_code=True)
def _(get_tier2_posteriors, mo, np, plt, render_fixed_matplotlib, t2_spectrum_slider):
    # ── Tier 2 Chain Trace Plot ──
    # Visualises the full MCMC walker chains for the selected spectrum,
    # mirroring the trace-plot diagnostic from the inference tool.
    _posteriors = get_tier2_posteriors()
    mo.stop(_posteriors is None or len(_posteriors) == 0)

    _idx = t2_spectrum_slider.value
    _post = _posteriors[_idx]
    _full_chain = _post.get("full_chain")
    mo.stop(
        _full_chain is None,
        mo.md("*Chain data not available for this spectrum (older benchmark run).*"),
    )

    _full_chain = np.asarray(_full_chain)  # (nsteps, nwalkers, ndim)
    _labels = _post.get("labels", [])
    _burnin = _post.get("burnin_used", 0)
    _gt = _post.get("truths", {})
    _nsteps, _nwalkers, _ndim = _full_chain.shape
    _filename = _post.get("filename", f"run{_post['run']}.spec")
    _inc = _post.get("inclination", 55.0)

    _fig, _axes = plt.subplots(
        _ndim, 1,
        figsize=(10, 2.5 * _ndim),
        sharex=True,
        squeeze=False,
    )
    _fig.patch.set_facecolor("black")

    for _j in range(_ndim):
        _ax = _axes[_j][0]
        _ax.set_facecolor("black")
        for _w in range(_nwalkers):
            _ax.plot(
                _full_chain[:, _w, _j],
                alpha=0.25, linewidth=0.5, color="#4c78a8",
            )
        # Mark burn-in boundary
        if _burnin > 0:
            _ax.axvline(_burnin, color="#f58518", linestyle="--",
                        linewidth=1, alpha=0.8, label="burn-in" if _j == 0 else None)
        # Overlay ground truth
        _lbl = _labels[_j] if _j < len(_labels) else f"param{_j}"
        _gt_key = "Inclination" if "Inclination" in _lbl else _lbl
        if _gt_key in _gt:
            _ax.axhline(_gt[_gt_key], color="#ff4444", linestyle="-",
                        linewidth=1.5, alpha=0.8, label="truth" if _j == 0 else None)
        _ax.set_ylabel(_lbl, color="white", fontsize=9)
        _ax.tick_params(colors="white", labelsize=8)
        for _spine in _ax.spines.values():
            _spine.set_color("white")

    _axes[-1][0].set_xlabel("Step", color="white", fontsize=10)
    # Show legend if any labelled artists were drawn (burn-in or truth lines)
    _handles, _leg_labels = _axes[0][0].get_legend_handles_labels()
    if _handles:
        _axes[0][0].legend(
            fontsize=8, loc="upper right",
            facecolor="black", edgecolor="white", labelcolor="white",
        )
    _fig.suptitle(
        f"MCMC Chains — {_filename} @ {_inc:.0f}°",
        color="white", fontsize=11, y=1.01,
    )
    _fig.tight_layout()

    mo.vstack([
        mo.md("#### Chain Trace Plot"),
        mo.md(f"Full walker chains ({_nwalkers} walkers × {_nsteps} steps). "
               f"Orange dashed line marks burn-in at step {_burnin}. "
               "Red line marks ground truth."),
        render_fixed_matplotlib(mo, _fig, width_px=960),
    ])
    return


@app.cell(hide_code=True)
def _(alt, build_bestfit_spectrum_altair, get_tier2_posteriors, mo, t2_spectrum_slider):
    # ── Tier 2 Best-Fit Spectrum Plot ──
    # Reconstructs the Starfish-style 3-panel plot (data vs model, residuals,
    # relative error) from spectral arrays stored during the benchmark run.
    _posteriors = get_tier2_posteriors()
    mo.stop(_posteriors is None or len(_posteriors) == 0)

    _idx = t2_spectrum_slider.value
    _post = _posteriors[_idx]
    _filename = _post.get("filename", f"run{_post['run']}.spec")
    _inc = _post.get("inclination", 55.0)
    _converged = _post.get("converged", False)
    _conv_tag = "converged" if _converged else "not converged"
    _bf = _post.get("bestfit_spec")

    mo.stop(
        not _bf,
        mo.md("*Best-fit spectrum data not available for this spectrum. Older reports and runs produced before the best-fit export fix will not have it.*"),
    )

    _fig = build_bestfit_spectrum_altair(
        alt,
        wavelength=_bf["wavelength"],
        data_flux=_bf["data_flux"],
        model_flux=_bf["model_flux"],
        model_cov_diag=_bf["model_cov_diag"],
        title=f"Best-Fit Model — {_filename} @ {_inc:.0f}° ({_conv_tag})",
        zoom_name="tier2_bestfit_zoom",
    )

    mo.vstack([
        mo.md("#### Best-Fit Spectrum (MCMC Posterior Mean)"),
        _fig,
    ])
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
    _orig = _arrs.get("display_original_flux", _arrs["original_flux"])[_idx]
    _pca = _arrs.get("display_pca_recon_flux", _arrs["pca_recon_flux"])[_idx]
    _loo = _arrs.get("display_loo_recon_flux", _arrs["loo_recon_flux"])[_idx]
    _gp = _arrs["grid_points"][_idx]
    _pnames = recon_param_names
    _flux_axis_title = _arrs.get("flux_axis_title", "Flux")

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
    _loo_recon_var = _arrs.get("display_loo_recon_var", _arrs.get("loo_recon_var"))
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
                y=alt.Y("Lower (2σ):Q", scale=alt.Scale(zero=False)),
                y2=alt.Y2("Upper (2σ):Q"),
            )
        )

    _spec_lines = (
        alt.Chart(_df_spec)
        .mark_line(strokeWidth=1.5, opacity=0.85)
        .encode(
            x=alt.X("Wavelength:Q", title="Wavelength (Å)"),
            y=alt.Y("Flux:Q", title=_flux_axis_title, scale=alt.Scale(zero=False)),
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
def _(alt, get_comparison_reports, mo, np, os):
    _comp_reports = get_comparison_reports()
    mo.stop(
        not _comp_reports or len(_comp_reports) < 2,
        mo.md("*Select 2+ reports above to compare.*"),
    )

    _comp_items = [mo.md("## Report Comparison")]

    def _report_full_label(_rep, _idx):
        _emu_path = _rep.get("config", {}).get("emulator", "")
        if _emu_path:
            return os.path.splitext(os.path.basename(_emu_path))[0]
        _tag = _rep.get("config", {}).get("tag")
        if _tag:
            return str(_tag)
        return f"report_{_idx}"

    def _report_short_label(_idx):
        return f"Emulator {_idx}"

    _report_meta = []
    for _i, _rep in enumerate(_comp_reports):
        _report_meta.append({
            "Emulator": _report_short_label(_i),
            "Emulator File": _report_full_label(_rep, _i),
        })

    _comp_items.append(mo.md("### Compared Emulators"))
    _comp_items.append(mo.ui.table(_report_meta, label="Emulator Labels", selection=None))

    # Comparison reduces each report to a small shared schema so reports from
    # different runs can still be compared even if some optional fields differ.
    _t1_data = {}
    for _i, _rep in enumerate(_comp_reports):
        _tag = _report_short_label(_i)
        _full = _report_full_label(_rep, _i)
        _t1 = _rep.get("tier1", {})
        _t1_data[_tag] = {
            "emulator_file": _full,
            "eas": _t1.get("emulator_accuracy_score"),
            "loo_rmse": _t1.get("loo_flux_rmse_median"),
            "pca_ev": _t1.get("pca_explained_variance"),
            "nlpd": _t1.get("nlpd_mean"),
            "q2": _t1.get("q2_aggregate"),
        }

    if _t1_data:
        _rows = [
            {
                "Emulator": _k,
                "Emulator File": _v["emulator_file"],
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
        _report_order = []
        _rmse_rows = []
        for _i, _rep in enumerate(_comp_reports):
            _tag = _report_short_label(_i)
            _full = _report_full_label(_rep, _i)
            _report_order.append(_tag)
            _agg = _rep.get("tier2", {}).get("aggregate", {})
            for _p in _params:
                _rmse_rows.append({
                    "Parameter": _p,
                    "RMSE": float(_agg.get(_p, {}).get("rmse", float("nan"))),
                    "Emulator": _tag,
                    "Emulator File": _full,
                })

        _rmse_chart = alt.Chart(alt.Data(values=_rmse_rows)).mark_bar().encode(
            x=alt.X(
                "Parameter:N",
                sort=_params,
                axis=alt.Axis(labelAngle=-40, labelLimit=280),
            ),
            xOffset=alt.XOffset("Emulator:N", sort=_report_order),
            y=alt.Y("RMSE:Q", title="RMSE"),
            color=alt.Color("Emulator:N", sort=_report_order, legend=alt.Legend(title="Emulator")),
            tooltip=[
                alt.Tooltip("Emulator:N", title="Emulator"),
                alt.Tooltip("Emulator File:N", title="Emulator File"),
                alt.Tooltip("Parameter:N", title="Parameter"),
                alt.Tooltip("RMSE:Q", title="RMSE", format=".4g"),
            ],
        ).properties(
            width=max(520, 120 * len(_params)),
            height=360,
            title="Tier 2 RMSE — Side by Side",
        )
        _comp_items.append(_rmse_chart)

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
        start=100, stop=5000, value=2500, step=100, show_value=True,
        label="MCMC steps",
    )
    mle_restarts_slider = mo.ui.slider(
        start=1, stop=10, value=3, step=1, show_value=True,
        label="MLE restarts",
    )

    # Inclination selector for Tier 2: which viewing angle column to read from
    # each .spec file.  "Random" assigns a reproducibly random inclination per
    # spectrum (seeded by run index).
    inclination_picker = mo.ui.dropdown(
        options={
            "30°": "30", "35°": "35", "40°": "40", "45°": "45",
            "50°": "50", "55°": "55", "60°": "60", "65°": "65",
            "70°": "70", "75°": "75", "80°": "80", "85°": "85",
            "Random": "random",
        },
        value="55°",
        label="Inclination (Tier 2)",
    )

    return (
        emu_picker,
        inclination_picker,
        max_spectra_slider,
        mcmc_steps_slider,
        mle_restarts_slider,
        obs_picker,
        tier_picker,
    )


@app.cell(hide_code=True)
def _(emu_picker, glob, mo, os, re):
    """Auto-populate grid and test-grid paths from the emulator selection."""
    _emu_val = emu_picker.value or ""
    _emu_base = os.path.basename(_emu_val)

    # Flux scale selector for Tier 2/3 — auto-detected from the emulator
    # filename when possible, but user can override.  Created here (not in
    # the picker cell) because marimo forbids reading .value in the same
    # cell that creates a UIElement.
    _detected_scale = "linear"
    if _emu_val:
        _emu_name = _emu_base.lower()
        if '_log_' in _emu_name:
            _detected_scale = "log"
        elif '_continuum-normalised_' in _emu_name:
            _detected_scale = "continuum-normalised"

    flux_scale_picker = mo.ui.dropdown(
        options=["linear", "log", "continuum-normalised"],
        value=_detected_scale,
        label="Flux Transform",
    )

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

        # Extract the wavelength range tag (e.g. "850-1850AA") from the emulator
        # filename so we find the grid that was processed with the same range.
        _wl_tag_match = re.search(r"(\d+-\d+AA)", _after_emu)
        _wl_tag = _wl_tag_match.group(1) if _wl_tag_match else "*"

        # Tier 1 consumes the processed NPZ grid, while Tier 2 consumes the raw
        # test-grid directory that contains individual .spec files.
        _grid_pattern = f"Grid-Emulator_Files/{_grid_stem}_grid_{_param_tag}_{_wl_tag}.npz"
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
    return emu_grid_info, flux_scale_picker, matched_grid_path, matched_testgrid_path


@app.cell(hide_code=True)
def _(
    emu_grid_info,
    emu_picker,
    flux_scale_picker,
    inclination_picker,
    max_spectra_slider,
    mcmc_steps_slider,
    mle_restarts_slider,
    mo,
    obs_picker,
    tier_picker,
):
    mo.vstack([
        mo.md("### Run Benchmark"),
        mo.hstack([emu_picker, flux_scale_picker], gap=1),
        emu_grid_info,
        mo.hstack([obs_picker], gap=1),
        mo.hstack([tier_picker, max_spectra_slider, mcmc_steps_slider, mle_restarts_slider, inclination_picker], gap=1),
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
    flux_scale_picker,
    inclination_picker,
    matched_grid_path,
    matched_testgrid_path,
    max_spectra_slider,
    mcmc_steps_slider,
    mle_restarts_slider,
    mo,
    obs_picker,
    os,
    run_btn,
    set_emulator,
    set_report,
    set_status_msg,
    set_tier1_arrays,
    set_tier2_posteriors,
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
        # Clear tier-specific live notebook state before the new run starts so
        # partial reruns cannot inherit stale Tier 1 or Tier 2 diagnostics.
        set_tier1_arrays(None)
        set_tier2_posteriors(None)

        # The picker stores numeric tier ids; keep the selected run isolated to
        # this execution so later reactive reruns do not reuse stale results.
        _tiers = [_v for _v in (tier_picker.value or [])]
        # Resolve the shared flux-scaling mode once so both Tier 2 and Tier 3
        # use the same setting even if only one tier is selected.
        _flux_scale = flux_scale_picker.value
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
            import json as _json
            import random as _random
            _t2_t0 = time.time()
            _test_path = _Path(matched_testgrid_path)
            # Tier 2 works spectrum-by-spectrum from the decompressed test-grid files.
            _spec_files = sorted(_test_path.glob("run*.spec"))
            if max_spectra_slider.value:
                _spec_files = _spec_files[: max_spectra_slider.value]
            _n_t2 = len(_spec_files)

            # Resolve inclination setting from UI
            _inc_raw = inclination_picker.value
            _inc_is_random = (_inc_raw == "random")
            _inc_fixed = float(_inc_raw) if not _inc_is_random else 55.0
            _VALID_INCS = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

            # Resolve wavelength range from emulator
            _wl_range = (float(_emu.wl.min()) + 10, float(_emu.wl.max()) - 10)

            # The lookup table is optional metadata used to compare recovered
            # parameters against the known test-grid inputs.
            _parquet = _ensure_lookup(_test_path)
            _friendly = _to_friendly(_emu.param_names)
            _n_params = len(_emu.param_names)

            # ---- Checkpoint / Resume ----
            # Each completed spectrum is appended as a JSON line to a partial
            # results file.  If a matching partial exists from a previous
            # interrupted run, we load completed results and skip those spectra.
            _emu_stem = os.path.basename(emu_picker.value or "").replace(".npz", "")
            _checkpoint_dir = "benchmark_results"
            os.makedirs(_checkpoint_dir, exist_ok=True)
            _checkpoint_path = os.path.join(
                _checkpoint_dir, f"benchmark_partial_{_emu_stem}.jsonl"
            )

            _completed_runs = set()
            _per_spectrum = []
            _tier2_posteriors = []  # full MCMC posteriors for corner-plot explorer
            _all_samples = {n: [] for n in _friendly}
            _all_truths = {n: [] for n in _friendly}
            _failures = 0
            _n_not_converged = 0
            _failure_log = []
            _mcmc_steps_val = mcmc_steps_slider.value

            # Resume: load any previously checkpointed results
            if os.path.exists(_checkpoint_path):
                try:
                    with open(_checkpoint_path, "r") as _cpf:
                        for _line in _cpf:
                            _line = _line.strip()
                            if not _line:
                                continue
                            _entry = _json.loads(_line)
                            # Legacy checkpoint entries created before the
                            # best-fit spectrum export fix are incomplete for
                            # the Tier 2 viewer, so rerun those spectra rather
                            # than treating them as finished.
                            if not _entry.get("bestfit_spec"):
                                continue
                            _completed_runs.add(_entry["run"])
                            _per_spectrum.append(_entry["spec_result"])
                            for _fn in _friendly:
                                if f"{_fn}_samples" in _entry:
                                    _all_samples[_fn].append(
                                        _np.array(_entry[f"{_fn}_samples"])
                                    )
                                if f"{_fn}_truth" in _entry:
                                    _all_truths[_fn].append(_entry[f"{_fn}_truth"])
                            if not _entry.get("mcmc_converged", True):
                                _n_not_converged += 1
                            # Reconstruct posterior entry for corner-plot explorer.
                            # If the checkpoint has the full ndim samples array,
                            # use it directly; otherwise fall back to grid-param
                            # columns only (older checkpoint format).
                            if "full_samples" in _entry:
                                _post_entry = {
                                    "run": _entry["run"],
                                    "filename": _entry.get("filename", f"run{_entry['run']}.spec"),
                                    "inclination": _entry.get("inclination", 55.0),
                                    "samples": _np.array(_entry["full_samples"]),
                                    "labels": _entry.get("full_labels", _friendly),
                                    "summary": _entry.get("full_summary", {}),
                                    "converged": _entry.get("mcmc_converged", False),
                                    "truths": _entry.get("truths", {}),
                                }
                                if "full_chain" in _entry:
                                    _post_entry["full_chain"] = _np.array(_entry["full_chain"])
                                    _post_entry["burnin_used"] = _entry.get("burnin_used", 500)
                                if "bestfit_spec" in _entry:
                                    _post_entry["bestfit_spec"] = _entry["bestfit_spec"]
                                if "prior_ranges" in _entry:
                                    _post_entry["prior_ranges"] = _entry["prior_ranges"]
                                _tier2_posteriors.append(_post_entry)
                            else:
                                # Legacy checkpoint: reconstruct from per-param samples
                                _cp_cols = []
                                _cp_labels = []
                                _cp_summary = {}
                                for _fn in _friendly:
                                    if f"{_fn}_samples" in _entry:
                                        _cp_cols.append(_np.array(_entry[f"{_fn}_samples"]))
                                        _cp_labels.append(_fn)
                                        _sr = _entry.get("spec_result", {})
                                        _cp_summary[_fn] = {
                                            "mean": _sr.get(f"{_fn}_mean", float("nan")),
                                            "std": _sr.get(f"{_fn}_std", float("nan")),
                                            "median": float("nan"),
                                            "hdi_3": float("nan"),
                                            "hdi_97": float("nan"),
                                        }
                                if _cp_cols:
                                    _cp_truths = {}
                                    for _fn in _friendly:
                                        if f"{_fn}_truth" in _entry:
                                            _cp_truths[_fn] = _entry[f"{_fn}_truth"]
                                    _tier2_posteriors.append({
                                        "run": _entry["run"],
                                        "filename": _entry.get("filename", f"run{_entry['run']}.spec"),
                                        "inclination": _entry.get("inclination", 55.0),
                                        "samples": _np.column_stack(_cp_cols),
                                        "labels": _cp_labels,
                                        "summary": _cp_summary,
                                        "converged": _entry.get("mcmc_converged", False),
                                        "truths": _cp_truths,
                                    })
                except Exception:
                    pass  # corrupted partial — start fresh

            _n_resumed = len(_completed_runs)

            with mo.status.progress_bar(
                total=_n_t2,
                title="Tier 2 — Parameter Recovery",
                subtitle=f"Resuming from {_n_resumed} completed…" if _n_resumed else "Starting…",
                completion_title="Tier 2 — Parameter Recovery",
                completion_subtitle="Complete ✓",
                show_rate=True,
                show_eta=True,
                remove_on_exit=True,
            ) as _t2_bar:
                # Advance the progress bar past already-completed spectra
                if _n_resumed > 0:
                    _t2_bar.update(increment=_n_resumed, subtitle=f"Resumed {_n_resumed} spectra")

                for _si, _sf in enumerate(_spec_files):
                    # Spectrum filenames encode the run number used by the lookup table.
                    _run_idx = int(_sf.stem.replace("run", ""))
                    _sname = _sf.name

                    # Skip already-completed spectra (resume mode)
                    if _run_idx in _completed_runs:
                        continue

                    # Resolve inclination for this spectrum
                    if _inc_is_random:
                        _rng = _random.Random(_run_idx)
                        _inc = float(_rng.choice(_VALID_INCS))
                    else:
                        _inc = _inc_fixed

                    # -- Load spectrum --
                    _t2_bar.update(increment=0, subtitle=f"Loading {_sname} (i={_inc:.0f}°)…")
                    try:
                        _wl, _flux, _sigma = _load_spec(str(_sf), _inc, _wl_range)
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

                    # Always record the true inclination used to extract this
                    # spectrum so the corner plot can overlay it.  This matters
                    # when inclination is an MCMC-sampled parameter (param9-11)
                    # but the parquet lookup didn't resolve it.
                    _gt.setdefault("Inclination", _inc)

                    # -- MLE (spinner with iteration updates) --
                    with mo.status.spinner(
                        title=f"MLE — {_sname}",
                        subtitle=f"Optimising ({_si + 1}/{_n_t2})…",
                        remove_on_exit=True,
                    ) as _mle_spin:
                        def _mle_cb(it, mx, best_nll, elapsed, restart=1, n_rst=1):
                            _rst_info = f"Restart {restart}/{n_rst} | " if n_rst > 1 else ""
                            _mle_spin.update(
                                subtitle=(
                                    f"{_rst_info}"
                                    f"Eval {it}/{mx} | "
                                    f"Best NLL: {best_nll:.2f} | "
                                    f"Time: {elapsed:.1f}s"
                                )
                            )
                        try:
                            _mle = _run_mle(
                                _emu, _wl, _flux, _sigma, flux_scale=_flux_scale,
                                max_iter=10_000, iteration_callback=_mle_cb,
                                n_restarts=mle_restarts_slider.value,
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
                                nwalkers=64,
                                nsteps=_mcmc_steps_val,
                                burnin=500,
                                iteration_callback=_mcmc_cb,
                                freeze_nuisance=True,
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

                    # Build per-spectrum summary
                    _spec_res = {
                        "run": _run_idx,
                        "filename": _sf.name,
                        "inclination": _inc,
                        "mle_success": _mle["success"],
                        "mcmc_converged": _mcmc["converged"],
                        "n_effective": _mcmc["n_effective"],
                        "mle_grid_params": _mle["grid_params"],
                    }

                    # Checkpoint entry: includes samples for resume
                    _cp_entry = {
                        "run": _run_idx,
                        "filename": _sf.name,
                        "inclination": _inc,
                        "mcmc_converged": _mcmc["converged"],
                        "spec_result": _spec_res,
                        # Full posterior for corner-plot explorer on resume
                        "full_samples": _mcmc["samples"].tolist(),
                        "full_chain": _mcmc["full_chain"].tolist(),
                        "burnin_used": _mcmc.get("burnin_used", 500),
                        "full_labels": _mcmc.get("labels", []),
                        "full_summary": {
                            k: {sk: sv for sk, sv in v.items()}
                            for k, v in _mcmc["summary"].items()
                        },
                        "truths": dict(_gt),
                        "bestfit_spec": _mcmc.get("bestfit_spec", {}),
                    }

                    # Build prior ranges dict from emulator grid bounds (used for
                    # the corner-plot full-prior-range view).
                    _prior_ranges_dict = {}
                    for _pi, _pn in enumerate(_emu.param_names):
                        _fn = _friendly[_pi] if _pi < len(_friendly) else _pn
                        _prior_ranges_dict[_fn] = [
                            float(_emu.min_params[_pi]),
                            float(_emu.max_params[_pi]),
                        ]
                    _cp_entry["prior_ranges"] = _prior_ranges_dict

                    # Map friendly names to their column index in the flat
                    # MCMC samples array. Columns follow model.labels order
                    # (nuisance params first), so we cannot assume grid param
                    # i maps to column i.
                    _mcmc_labels = _mcmc.get("labels", [])
                    for _pi, _fn in enumerate(_friendly):
                        if _fn in _mcmc["summary"]:
                            _spec_res[f"{_fn}_mean"] = _mcmc["summary"][_fn]["mean"]
                            _spec_res[f"{_fn}_std"] = _mcmc["summary"][_fn]["std"]
                            # Find correct column for this parameter
                            _col = _mcmc_labels.index(_fn) if _fn in _mcmc_labels else _pi
                            _samples_i = _mcmc["samples"][:, _col]
                            _all_samples[_fn].append(_samples_i)
                            _cp_entry[f"{_fn}_samples"] = _samples_i.tolist()
                            if _fn in _gt:
                                _all_truths[_fn].append(_gt[_fn])
                                _spec_res[f"{_fn}_truth"] = _gt[_fn]
                                _cp_entry[f"{_fn}_truth"] = _gt[_fn]
                                _spec_res[f"{_fn}_delta_sigma"] = (
                                    _mcmc["summary"][_fn]["mean"] - _gt[_fn]
                                ) / max(_mcmc["summary"][_fn]["std"], 1e-10)

                    # Update the spec_result in the checkpoint entry (it now has _mean/_std keys)
                    _cp_entry["spec_result"] = _spec_res
                    _per_spectrum.append(_spec_res)

                    # Store full posterior for corner-plot explorer
                    _tier2_posteriors.append({
                        "run": _run_idx,
                        "filename": _sf.name,
                        "inclination": _inc,
                        "samples": _mcmc["samples"],  # (N, ndim) burnt+thinned flat samples
                        "full_chain": _mcmc["full_chain"],  # (nsteps, nwalkers, ndim) full chain
                        "burnin_used": _mcmc.get("burnin_used", 500),
                        "labels": _mcmc_labels,
                        "summary": _mcmc["summary"],
                        "converged": _mcmc["converged"],
                        "truths": dict(_gt),
                        "bestfit_spec": _mcmc.get("bestfit_spec", {}),
                        "prior_ranges": _prior_ranges_dict,
                    })

                    # Append checkpoint line (crash-safe: one write per spectrum)
                    try:
                        with open(_checkpoint_path, "a") as _cpf:
                            _cpf.write(_json.dumps(_cp_entry) + "\n")
                    except Exception:
                        pass

                    _status_lbl = (
                        f"✓ {_sname} (i={_inc:.0f}°) complete"
                        if _mcmc["converged"]
                        else f"⚠ {_sname} (i={_inc:.0f}°, not converged)"
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
                mcmc_walkers=64,
                mcmc_steps=_mcmc_steps_val,
                mcmc_burnin=500,
                elapsed=time.time() - _t2_t0,
                n_not_converged=_n_not_converged,
            )

            # Clean up checkpoint file on successful completion
            if os.path.exists(_checkpoint_path):
                try:
                    os.remove(_checkpoint_path)
                except Exception:
                    pass

            # Persist the full per-spectrum posteriors so the interactive
            # corner-plot explorer can render any spectrum after the run.
            set_tier2_posteriors(_tier2_posteriors if _tier2_posteriors else None)

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
                        flux_scale=_flux_scale,
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
            "mle_restarts": mle_restarts_slider.value,
            "max_spectra": max_spectra_slider.value,
        }
        _report = _build_report_card(
            _tier1_result, _tier2_result, _tier3_results, _config,
            tier2_posteriors=_tier2_posteriors if 2 in _tiers and matched_testgrid_path else None,
        )

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
