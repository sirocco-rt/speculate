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
def __(mo):
    # Navigation - back to home only
    back_home = mo.nav_menu(
        {
            "/": "## <ins>⬅︎ Back to Home</ins>e",
        },
        orientation="horizontal"
    )
    back_home
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
