# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Speculate Inference Tool",
)


@app.cell(hide_code=True)
def __():
    import marimo as mo
    mo.md(
        """
        # Speculate Inference Tool 🔮
        
        Compare observations with emulated spectral models.
        """
    )
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


@app.cell
def __(mo):
    # Add logo
    import pathlib
    
    logo_path = pathlib.Path("assets/logos/Speculate_logo2.png")
    
    if logo_path.exists():
        logo = mo.image(src=str(logo_path), width=300)
        link = mo.md('<p style="text-align: center;">Powered by <a href="https://github.com/sirocco-rt" target="_blank">Sirocco-rt</a></p>')
        mo.vstack([logo, link], align="center")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("""
    ---
    ## 🚧 Coming Soon!
    
    This tool is currently under development and will include:
    
    ### Planned Features:
    - **Observation Upload**: Drag-and-drop your spectral data
      - CSV format support (wavelength, flux, error)
      - Multiple file formats
      - Data validation
    - **Model Selection**: Choose from available emulators
      - Pre-trained models from HuggingFace
      - Locally trained models
      - Model comparison
    - **Parameter Estimation**: Infer physical parameters from observations
      - Maximum likelihood estimation
      - Bayesian inference
      - Confidence intervals
    - **Visualization**: Compare observations with model predictions
      - Interactive plots
      - Residual analysis
      - Corner plots for parameter posteriors
    - **Export Results**: Save analysis results and plots
    
    ### ✅ Fully Functional on HuggingFace Space!
    This tool will work on the free Space tier because:
    - Pre-trained models are small (~MBs)
    - Inference is computationally light
    - User observations are typically small files
    - No grid data needed!
    """)
    return


if __name__ == "__main__":
    app.run()
