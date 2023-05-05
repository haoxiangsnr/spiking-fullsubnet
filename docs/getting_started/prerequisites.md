# Prerequisites

To conduct experiments, a Linux-based operating system is required as it does not support the Windows platform. Additionally, Anaconda or Miniconda must be installed to create a virtual environment and install dependencies. For GPU parallel training, CUDA (version 10.2 or higher) and the corresponding CuDNN acceleration library must be installed.



## Create virtual environment

First, create a conda virtual environment:

```shell
# python=3.10 is tested, but you may also use other versions.
conda create --name audiozen python=3.10

conda activate audiozen
```

## Install conda dependencies

Some dependencies, like PyTorch, are recommended to be installed using conda instead of PyPI. You may use the [THU mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to speed up the installation.

```shell
# PyTorch. Although pytorch=1.12.1 has been tested, you may also use other versions.
# Use CUDA 10.2 as an example:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

# Install other dependencies, which are recommended to be installed using conda instead of pypi
conda install tensorboard joblib matplotlib

# (Optional) If there are "mp3" format data in your dataset, you need to install ffmpeg
conda install -c conda-forge ffmpeg
```

## Install PyPI dependencies

Clone the repository and install PyPI dependencies. You may use the [THU mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to install PyPI dependencies.

```shell
# Clone the repository
git clone git@github.com:haoxiangsnr/audiozen.git

cd audiozen

# Install PyPI dependencies
pip install -r requirements.txt
```

## Install audiozen package in editable mode

```shell
# Install audiozen package in editable mode and other PyPI dependencies
pip install -e .
```

After installing audiozen package, we can call `audiozen` package in everywhere of code.
In addition, we can modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your environment.
