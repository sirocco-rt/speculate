import marimo

__generated_with = "0.17.8"
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
    # Static sidebar - always shows all options
    mo.sidebar(
        mo.vstack([
            mo.md("# 🔭 Speculate"),
            mo.md(" "),
            mo.md(" "),
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
        ])
    )
    return


@app.cell
def _(mo):
    # Add logo using marimo's image display
    import pathlib

    logo_path = pathlib.Path("assets/logos/Speculate_logo3.png")

    logo = mo.image(src=str(logo_path), width=800)
    link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
    mo.vstack([mo.md("---"), logo, link], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### A Sirocco Emulator for Astrophysical Spectra

    Welcome to the Speculate web interface! This suite provides tools for downloading,
    analyzing, training, and using spectral emulators based on Sirocco radiative transfer grids.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🚀 Available Tools

    Select your working environment to see available features:
    """)
    return


@app.cell
def _(mo):
    # Environment mode selector
    environment_mode = mo.ui.dropdown(
        options={
            "🤗 Space": "HuggingFace Space",
            "Local Computer": "Local Machine"
        },
        value="🤗 Space",
        label="Select your environment:"
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
            mo.md("""
            **🤗 HuggingFace Space Mode Active**

            Some tools are disabled in HuggingFace Space mode. 
            The following will show "Not Available" if you navigate to them:
            - Grid Downloader (use local installation)
            - Training Tool (use local installation)
            """),
            kind="warn"
        )
    else:
        mode_warning = mo.callout(
            mo.md("**💻 Local Mode Active** - All tools available!"),
            kind="success"
        )

    mode_warning
    return


@app.cell
def _(environment_mode, mo):
    # Display feature availability based on environment
    if environment_mode.value == "HuggingFace Space":
        features_info = mo.md("""
        ### 🤗 HuggingFace Space Mode

        **Available Features:**
        - ❌ **Grid Downloader** - Browse datasets, view parameters, get download links
        - ✅ **Grid Inspector** - Preview single spectra (streaming)
        - ❌ **Training** - Not available (requires local installation)
        - ✅ **Inference Tool** - Use pre-trained emulators (upload your observations!)

        **Limitations:**
        - Single spectrum preview only (memory: 16GB limit)
        - Cannot download/store full grids
        - Pre-trained models only

        ⚠️ **Note:** For full features, select "Local Machine" and install Speculate locally.
        """)
    else:  # local mode
        features_info = mo.md("""
        ### 💻 Local Machine Mode

        **All Features Available:**
        - ✅ **Grid Downloader** - Full download & decompression
        - ✅ **Grid Inspector** - Analyze entire grids
        - ✅ **Training Tool** - Train custom emulators on full grids
        - ✅ **Inference Tool** - Use any trained emulator (local or HuggingFace)

        **Additional Capabilities:**
        - Bulk operations across multiple spectra
        - Custom emulator architectures
        - Upload trained models to HuggingFace
        - No memory limitations
        """)

    features_info
    return


if __name__ == "__main__":
    app.run()
