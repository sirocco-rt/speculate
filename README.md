<img width="1659" height="779" alt="Screenshot 2025-11-06 at 20 17 28" src="https://github.com/user-attachments/assets/7ffd7afb-90de-460a-8211-58ecc64877d4" />

An emulator for Sirocco for faster inference of an observational spectrum's outflow parameters
## Local Installation

### Option A: Conda [Slower]
This method automatically handles system dependencies (Git LFS) and NVIDIA drivers.

1. Create the environment:
    ```bash
    cd <path/to/speculate>
    conda env create -f environment.yml
    conda activate speculate_env
    git lfs install && git lfs pull
    ```

### Option B: UV Python (uv pip) [Faster]

If you are not using Conda, you must point to the NVIDIA server and handle LFS manually.

1. Install uv and git-lfs on your computer. Follow instructions: 

- uv:  https://docs.astral.sh/uv/getting-started/installation/ Verify with: ```uv --version```. The command returns a version number

- git-lfs: https://git-lfs.com Verify with: 
    ```git lfs version```. The command returns a verison number.  


2. Change directory to speculate and set up virtual environment: 
    ```bash
    cd <path/to/speculate>
    uv venv speculate_env --python 3.12
    source speculate_env/bin/activate
    ```

3. Install python libraries:
    ```bash
    uv pip install --extra-index-url https://pypi.nvidia.com -r requirements.txt
    ```
4. Fetch assets (images/videos) - 🚨 Ensure you have git-lfs installed on your system first. Run:
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

We leave these specifics to the user as HPC systems vary greatly, but VS Code (server) as port forwarding functionality that from our experience, smoothens the process. 

Check out marimo at <https://github.com/marimo-team/marimo>
Check out the configuration reference at <https://huggingface.co/docs/hub/spaces-config-reference>