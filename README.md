---
title: Speculate
emoji: 💨
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: true
license: gpl-3.0
short_description: A sirocco emulator for astrophysical spectra
---

<img width="1659" height="779" alt="Screenshot 2025-11-06 at 20 17 28" src="https://github.com/user-attachments/assets/7ffd7afb-90de-460a-8211-58ecc64877d4" />

An emulator for Sirocco for faster inference of an observational spectrum's outflow parameters
## Local Installation

### Option A: Conda
This method automatically handles system dependencies (Git LFS) and NVIDIA drivers.

1. Create the environment:
    ```bash
    conda env create -f environment.yml
    conda activate speculate_env
    git lfs install && git lfs pull
    ```

### Option B: Standard Python (pip)

If you are not using Conda, you must point to the NVIDIA server and handle LFS manually.


1. Set up virtual environment: 
    ```bash
    python -m venv speculate_env
    source speculate_env/bin/activate
    ```

2. Install python libraries:
    ```bash
    pip install --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) -r requirements.txt
    ```
3. Fetch assets (images/videos) - 🚨 Ensure you have git-lfs installed on your system first. You can verify this with 
    ```bash 
    git lfs version
    ``` 
    If not, ensure you download the correct setup for your system until the command returns a verison number. 
    Afterwards, run:
    ```bash
    git lfs install && git lfs pull
    ```

##### NOTES

LFS isn't critical for functionality. It is use to load Speculate's image/video assets. If you can live without them (i.e it doesn't look very pretty), the conda/virtual environment with install packages from the requirements.txt file is all you need. 

## Running Speculate
### 🤗 On Huggingface Space (Lightweight Model Inference)
Simply follow the link: https://huggingface.co/Sirocco-rt 
Then click on the Speculate Space.
Hint: The Huggingface Space may go to sleep if unused for a while. If so, just restart the space when prompted!

### 💻 On a local machine (GPU recommended)
To get started, navigate to speculate's root directory, activate the speculate environment created when installing, and then simply run: 
```bash
python run.py
```
Speculate doesn't require a GPU to run, however, for the larger models, CPU computational time may be prohibitive.

### 🔌 On an GPU HPC system (Training Models)
This requires a couple extra steps as you will have to port forward the interface to your local browser. Also, as good practice, DON'T run speculate on login nodes!

First boot up an interactive compute node:
TODO

Check out marimo at <https://github.com/marimo-team/marimo>
Check out the configuration reference at <https://huggingface.co/docs/hub/spaces-config-reference>