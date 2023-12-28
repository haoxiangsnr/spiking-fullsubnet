# Prerequisites and Installation

## Prerequisites

Spiking-FullSubNet is built on top of PyTorch and provides standard audio signal processing and deep learning tools.
To install the PyTorch binaries, we recommend [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) as a Python distribution.

## Installation

1. First, create a Conda virtual environment with Python. In our project, `python=3.10` is tested.
    ```shell
    # Create a virtual environment named `spiking-fullsubnet`
    conda create --name spiking-fullsubnet python=3.10

    # Activate the environment
    conda activate spiking-fullsubnet
    ```
    The following steps will assume you have activated the `spiking-fullsubnet` environment.

2. Install Conda dependencies. Some dependencies of Spiking-FullSubNet, e.g., PyTorch and Tensorboard, are recommended to be installed using Conda instead of PyPI. First, we install a CUDA-capable PyTorch. Although `pytorch=2.1.1` has been tested, you may also [use other versions](https://pytorch.org/get-started/previous-versions/):
    ```shell
    # Install PyTorch
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    # Install other Conda dependencies
    conda install tensorboard joblib matplotlib

    # (Optional) If you have "mp3" format audio data in your dataset, install ffmpeg first.
    conda install ffmpeg -c conda-forge
    ```

3. Install PyPI dependencies. Clone the repository and install PyPI dependencies via `pip -r requirements.txt`. Check `requirements.txt` for more details.
    ```shell
    git clone git@github.com:haoxiangsnr/spiking-fullsubnet.git

    cd spiking-fullsubnet

    pip install -r requirements.txt
    ```

4. We integrated all the audio signal processing tools into a package named `audiozen`. We will install the `audiozen` package in editable mode. By installing in editable mode, we can call `audiozen` package in everywhere of code, e.g, in `recipes` and `tools` folders. In addition, we are able to modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your conda environment.
    ```shell
    cd audiozen

    pip install --editable . # or for short: pip install -e .
    ```

Ok, all installations have done. You may speed up the installation by the following tips.

```{tip}
- [Speed up your Conda installs with Mamba](https://pythonspeed.com/articles/faster-conda-install/)
- Use the [THU Anaconda mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to speed up the Conda installation.
- Use the [THU PyPi mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to speed up the PyPI installation.
```