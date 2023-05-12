# Installation

## Prerequisites

AudioZEN package is built on top of PyTorch and provides standard audio signal processing and deep learning tools.
To install the PyTorch binaries, you will need to use at least one of two supported package managers: [Anaconda](https://www.anaconda.com/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) and pip. Anaconda (or Miniconda) is the recommended package manager as it will provide you all of the PyTorch dependencies in one, sandboxed install, including Python and pip. For GPU parallel training, CUDA (version 10.2 or higher) and the corresponding CuDNN acceleration library must be installed.

## Installation

### Create virtual environment

First, create a Conda virtual environment with Python. Here, `python=3.10` is tested, but you may also use other versions.

```shell
conda create --name audiozen python=3.10
conda activate audiozen
```

The following steps will assume you have activated the `audiozen` environment.

### Install Conda dependencies

Some dependencies of AudioZEN, like PyTorch, Lava, and Tensorboard, are recommended to be installed using Conda instead of PyPI. First, we install a CUDA-capable PyTorch. Although `pytorch=1.12.1` has been tested, you may also [use other versions](https://pytorch.org/get-started/previous-versions/). We use `CUDA 11.6` as an example:

```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Install Lava and Lava-dl, which are used for neuromorphic computing.
Check [Lava](https://github.com/lava-nc/lava) and [Lava-dl](https://github.com/lava-nc/lava-dl) for more details.

```shell
conda install lava lava-dl -c conda-forge
```

Install other Conda dependencies:

```shell
conda install tensorboard joblib matplotlib

# (Optional) If you have "mp3" format data in your dataset, install ffmpeg first.
conda install -c conda-forge ffmpeg
```

### Install PyPI dependencies

Clone the repository and install PyPI dependencies via `pip -r requirements.txt`. Check `requirements.txt` for more details.

```shell
git clone git@github.com:haoxiangsnr/audiozen.git

cd audiozen

pip install -r requirements.txt
```

## Install AudioZEN package in editable mode

Finally, we will install the AudioZEN package in editable mode (a.k.a. development mode). By installing in editable mode, we can call `audiozen` package in everywhere of code, e.g, in `recipes` and `tools` folders. In addition, we can modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your environment.

```shell
pip install --editable .
```

Ok, all installations have done.

## References

- [Speed up your Conda installs with Mamba](https://pythonspeed.com/articles/faster-conda-install/)
- Use the [THU Anaconda mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to speed up the Conda installation.
- Use the [THU PyPi mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to install PyPI dependencies.
