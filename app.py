import marimo

__generated_with = "0.17.7"

# Create marimo ASGI app with multiple routes
server = (
    marimo.create_asgi_app()
    .with_app(path="/", root="speculate.py")
    .with_app(path="/downloader", root="speculate_grid_downloader.py")
    .with_app(path="/inspector", root="speculate_grid_inspector.py")
    .with_app(path="/training", root="speculate_training.py")
    .with_app(path="/inference", root="speculate_inference.py")
)

# Build and export the ASGI application
app = server.build()