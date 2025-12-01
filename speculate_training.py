# /// script
# [tool.marimo.display]
# theme = "dark"
# ///
import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Speculate Training Tool",
)


@app.cell(hide_code=True)
def __():
    import marimo as mo
    mo.md(
        """
        # Speculate Training Tool 🧠
        
        Train custom emulator models on spectral grids.
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
    - **Model Selection**: Choose from multiple emulator architectures
      - Gaussian Processes
      - Neural Networks
      - Polynomial interpolation
      - Custom architectures
    - **Training Pipeline**: Automated training workflow
      - Data loading from local grids
      - Cross-validation
      - Hyperparameter tuning
    - **Model Evaluation**: Performance metrics and diagnostics
    - **Export to HuggingFace**: Upload trained models to Model Hub
    - **Model Versioning**: Track and manage different model versions
    
    ### ⚠️ Local Machine Only
    This tool requires:
    - Access to full spectral grids (1.8-18 GB)
    - Computational resources for training
    - Local installation of Speculate
    
    Training on HuggingFace Space is not supported due to memory and storage limitations.
    """)
    return


if __name__ == "__main__":
    app.run()
