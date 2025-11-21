import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Speculate Grid Inspector",
)


@app.cell(hide_code=True)
def __():
    import marimo as mo
    mo.md(
        """
        # Speculate Grid Inspector 📊
        
        Visualize and analyze spectral grid data.
        """
    )
    return (mo,)


@app.cell
def __(mo):
    # Navigation - back to home only
    back_home = mo.nav_menu(
        {
            "/": "## <ins>⬅︎ Back to Home</ins>",
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
    - **Single Spectrum Viewer**: Load and visualize individual spectra from HuggingFace datasets
    - **Interactive Plots**: Explore spectral features with zoom, pan, and selection tools
    - **Multi-Spectrum Comparison**: Compare spectra across different parameter values
    - **Parameter Space Exploration**: Visualize how spectra vary across the grid
    - **Export Capabilities**: Save plots and data for further analysis
    
    ### Space vs Local Mode:
    - **HuggingFace Space**: Stream and view single spectra
    - **Local Machine**: Full grid analysis and bulk operations
    """)
    return


if __name__ == "__main__":
    app.run()
