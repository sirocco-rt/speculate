# /// script
# [tool.marimo.display]
# theme = "dark"
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(
    width="full",
    app_title="Speculate - Spectral Emulator Suite",
    layout_file="layouts/speculate.grid.json",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Welcome to
    """)
    return


@app.cell
def _(mo):
    # Add logo using marimo's image display
    import pathlib

    logo_path = pathlib.Path("assets/logos/Speculate_logo4.mp4")

    logo = mo.Html(f'<video autoplay muted playsinline style="border-radius: 8px; width: 800px; max-width: 100%;"><source src="{logo_path}" type="video/mp4"></video>')
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

    Documentation is hosted on Speculate's GitHub repository wiki page. Links to Sirocco's repository and documentation are also linked in the sidebar.

    HuggingFace Spaces are resource-limited and hence, only lightweight tools/models should be used online. For full access, perform a local install on a more powerful local machine or HPC interactive node. GPU access is HIGHLY RECOMMENDED for heavy-weight models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 🚀 Please select your correct working environment:
    """)
    return


@app.cell
def _(mo):
    import os

    # Auto-detect if running on HuggingFace Space
    is_hf_space = os.environ.get("SPACE_ID") is not None
    default_mode = "🤗 HuggingFace Space" if is_hf_space else "💻 Local Computer"

    # Environment mode selector
    environment_mode = mo.ui.dropdown(
        options={
            "🤗 HuggingFace Space": "HuggingFace Space",
            "💻 Local Computer": "Local Machine"
        },
        value=default_mode,
        label="",
        full_width=True
    )
    environment_mode
    return (environment_mode,)


@app.cell
def _(environment_mode):
    print(environment_mode.value)
    return


@app.cell
def _(environment_mode, mo):
    # Show warning about which tools are available based on mode
    current_mode = environment_mode.value

    if current_mode == "HuggingFace Space":
        mode_warning = mo.callout(
            mo.md("""**🤗 HuggingFace Space Mode Active** 

            Only the following tools will operate as expected due to resource limitations: 
            - **Grid Inspector**
            - **Inference Tool**

            ⚠️ **Note:** For full features, install Speculate locally and select "Local Machine".
            """),
            kind="warn"
        )
    else:
        mode_warning = mo.callout(
            mo.md("""**💻 Local Mode Active** 

            All tools available! Please ensure you have adequate local CPU and GPU provisions."""),
            kind="success"
        )

    mode_warning
    return (current_mode,)


@app.cell
def _(current_mode, mo):
    # Side bar menu adjusted based on environment mode
    if current_mode == "HuggingFace Space":
        # Show all items but mark unavailable ones
        menu = mo.vstack([
            mo.md("# 🔭 Speculate"),
            mo.md(" "),
            mo.md(" "),
            mo.md("---"),
            mo.md("---"),
            mo.md(" "),
            mo.md(" "),
            mo.nav_menu({
                "/": f"###{mo.icon('lucide:home')} Home",
                "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
                "/inference": f"###{mo.icon('lucide:sparkles')} Inference Tool",
            }, orientation="vertical"),
            mo.md(" "),
            mo.md("---"),
            mo.md("### 🔒 Locked Tools:"),
            mo.md("Install Speculate Locally"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:download')} Grid Downloader"),
            mo.md(" "),
            mo.md(f"###{mo.icon('lucide:brain')} Training Tool"),
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
    else: 
        # Show all items as available
        menu = mo.vstack([
            mo.md("# 🔭 Speculate"),
            mo.md(" "),
            mo.md(" "),
            mo.md("---"),
            mo.md("---"),
            mo.md(" "),
            mo.md(" "),
            mo.nav_menu({
                "/": f"###{mo.icon('lucide:home')} Home",
                "/downloader": f"###{mo.icon('lucide:download')} Grid Downloader",
                "/inspector": f"###{mo.icon('lucide:chart-spline')} Grid Inspector",
                "/training": f"###{mo.icon('lucide:brain')} Training Tool",
                "/inference": f"###{mo.icon('lucide:sparkles')} Inference Tool",
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

    mo.sidebar(menu)
    return


if __name__ == "__main__":
    app.run()
