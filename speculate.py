# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="full",
    app_title="Speculate - Spectral Emulator Suite",
    layout_file="layouts/speculate.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # Add logo using marimo's video display
    import pathlib

    logo_path = pathlib.Path("assets/logos/Speculate_logo4.mp4")

    logo = mo.video(src=str(logo_path), muted=True, autoplay=True, loop=False, controls=False, rounded=False, width=800)
    link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    mo.vstack([mo.md("---"), logo, link], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Welcome to the Speculate interface!

    Here, you can find all the tools for analysing spectra against sirocco simulations through the sidebar menu on the left. This suite provides:
    - **Grid Downloader** -- An intuitive interface for downloading and decompressing Sirocco spectral files (.spec).
    - **Grid Inspector** -- A plotting display interface to browse through our pre-made Sirocco spectra datasets.
    - **Training Tool** -- A training interface for the emulator. This tool requires GPU access, so it should only be run locally.
    - **Inference Tool** -- An analysis interface to uncover the best-matching sirocco emulated spectra to **YOUR** observational spectrum.
    - **Quick Fit** -- A lightweight fitting interface for quick-look analysis of spectra. This tool is designed to run on HuggingFace Spaces with limited resources.
    - **Benchmark Suite** -- A collection of benchmark results comparing different emulator models and training configurations.
    - **Documentation** -- Hosted on Speculate's GitHub repository wiki page. Links to Sirocco's repository and documentation are also available in the sidebar.

    🤗 HuggingFace Spaces are resource-limited and hence, only lightweight tools/models should be used online.

    For full access, perform a local install on a more powerful local machine or HPC interactive node. GPU access is HIGHLY RECOMMENDED.
    """)
    return


@app.cell
def _(mo):
    import os

    # Hard-detect if running on a real HuggingFace Space
    is_hf_space = os.environ.get("SPACE_ID") is not None

    # On a real HF Space: no switch, mode is locked
    # On local: provide a switch to simulate HF mode for development
    if is_hf_space:
        hf_mode_switch = None
    else:
        hf_mode_switch = mo.ui.switch(
            value=False,
            label="🤗 HuggingFace Space Mode",
        )
    return hf_mode_switch, is_hf_space


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
def _(hf_mode_switch, is_hf_space, mo, usage_bars):
    # Resolve the current mode (must be in a separate cell from switch creation)
    if is_hf_space:
        current_mode = "HuggingFace Space"
    elif hf_mode_switch is not None and hf_mode_switch.value:
        current_mode = "HuggingFace Space"
    else:
        current_mode = "Local Machine"

    # Build sidebar based on mode
    sidebar_items = [
        mo.md(f"#Speculate {mo.icon('lucide:telescope')}"),
        mo.md(" "),
        mo.md("---"),
        mo.md("---"),
        mo.md(" "),
    ]

    if current_mode == "HuggingFace Space":
        sidebar_items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
        }, orientation="vertical"))
        sidebar_items.extend([
            mo.md(" "),
            mo.md("---"),
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
        sidebar_items.append(mo.nav_menu({
            "/": f"###{mo.icon('lucide:home')} Home",
            "/downloader": f"###{mo.icon('lucide:download')} Grid Downloader",
            "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
            "/training": f"###{mo.icon('lucide:brain')} Training Tool",
            "/inference": f"###{mo.icon('lucide:sparkles')} Inference Tool",
            "/quickfit": f"###{mo.icon('lucide:zap')} Quick Fit",
            "/benchmark": f"###{mo.icon('lucide:test-tubes')} Benchmark Suite",
        }, orientation="vertical"))

    sidebar_items.extend([
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
            "https://sirocco-rt.readthedocs.io/en/latest/": f"###{mo.icon('lucide:wind')} Sirocco Docs",
        }, orientation="vertical"),
        mo.md("---"),
        mo.md("---"),
    ])

    # Show the dev-mode switch on local installs (after documentation)
    if hf_mode_switch is not None:
        sidebar_items.append(hf_mode_switch)

    sidebar_items.extend([mo.md("---"), usage_bars])

    mo.sidebar(mo.vstack(sidebar_items))
    return (current_mode,)


@app.cell
def _(current_mode, mo):
    if current_mode == "HuggingFace Space":
        mode_warning = mo.callout(
            mo.md(f"""**🤗 HuggingFace Space Mode Active** 

            Only the following tools will operate as expected due to resource limitations: 
            - **Grid Inspector**
            - **Quick Fit**

            {mo.icon('lucide:triangle-alert')} **Note:** For full features, install Speculate locally."""),
            kind="warn"
        )
    else:
        mode_warning = mo.callout(
            mo.md(f"""**{mo.icon('lucide:monitor')} Local Mode Active** 

            All tools available! Please ensure you have adequate local CPU and GPU provisions."""),
            kind="success"
        )

    mode_warning
    return


if __name__ == "__main__":
    app.run()
