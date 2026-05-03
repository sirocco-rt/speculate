#!/usr/bin/env python
"""
Simple runner script for Speculate marimo app.

Usage:
    python run.py              # Runs on default port 8080
    python run.py --port 7860  # Custom port (HuggingFace default)
"""
import sys
import socket
import webbrowser
import threading
import time
import numpy as np

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def open_browser(port):
    """Open browser after a short delay to ensure server is ready"""
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{port}")

if __name__ == "__main__":
    port = 8080  # Default port
    # if port in use, choose random port between 8000 and 8800
    if is_port_in_use(port):
        port = np.random.randint(8000, 8800)  # Default random port
    
    # Check if port is specified
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    
    # Open browser in background thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    # Import and run directly with uvicorn
    import uvicorn
    from app import app
    
    print(f"Starting Speculate on http://localhost:{port}")
    print("Browser will open automatically...")
    uvicorn.run(app, host="0.0.0.0", port=port)
