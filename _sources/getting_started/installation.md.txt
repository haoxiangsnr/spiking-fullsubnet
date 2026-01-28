# Prerequisites and Installation

## Prerequisites

Spiking-FullSubNet is built on top of PyTorch and provides standard audio signal processing and deep learning tools.

- **Python**: >= 3.10
- **uv**: We recommend using [uv](https://docs.astral.sh/uv/) as the package manager for faster and more reliable dependency management.

## Installation

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtual environments and dependencies efficiently.

1. **Install uv** (if not already installed):
    ```shell
    # On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Or using pip
    pip install uv
    ```

2. **Clone the repository**:
    ```shell
    git clone https://github.com/haoxiangsnr/spiking-fullsubnet.git
    cd spiking-fullsubnet
    ```

3. **Install PyTorch** (CUDA version):
    ```shell
    # Install PyTorch with CUDA support first
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    ```

4. **Sync dependencies and install the project**:
    ```shell
    # Install all dependencies (creates .venv automatically)
    uv sync

    # Or with GPU support (includes onnxruntime-gpu)
    uv sync --extra gpu

    # Or with all optional dependencies (gpu, test, docs, build)
    uv sync --all-extras
    ```

    This will:
    - Create a virtual environment in `.venv`
    - Install all dependencies from `uv.lock`
    - Install `audiozen` in editable mode

5. **Activate the environment** (optional, uv commands auto-detect):
    ```shell
    source .venv/bin/activate
    ```

### Option 2: Using Conda + pip

If you prefer Conda, you can still use the traditional approach:

1. **Create a Conda environment**:
    ```shell
    conda create --name spiking-fullsubnet python=3.10
    conda activate spiking-fullsubnet
    ```

2. **Install PyTorch**:
    ```shell
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3. **Install the project**:
    ```shell
    git clone https://github.com/haoxiangsnr/spiking-fullsubnet.git
    cd spiking-fullsubnet
    pip install -e .
    ```

## Common uv Commands

```shell
uv sync                    # Sync dependencies from lockfile
uv sync --extra gpu        # Include GPU dependencies
uv sync --all-extras       # Include all optional dependencies
uv add <package>           # Add a new dependency
uv lock --upgrade          # Upgrade all dependencies
uv run <command>           # Run a command in the virtual environment
uv build                   # Build the package
```

```{tip}
- uv is significantly faster (10~100x) than pip and handles dependency resolution more reliably.
- The `uv.lock` file ensures reproducible installations across different machines.
- Use `uv run python script.py` to run scripts without manually activating the environment.
```