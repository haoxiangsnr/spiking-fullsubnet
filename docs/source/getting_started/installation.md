# Getting Started

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

4. Install Spiking-FullSubNet package in editable mode. Finally, we will install the Spiking-FullSubNet package in editable mode (a.k.a. development mode). By installing in editable mode, we can call `spiking_fullsubnet` package in everywhere of code, e.g, in `recipes` and `tools` folders. In addition, we are able to modify the source code of `spiking_fullsubnet` package directly. Any changes to the original package would reflect directly in your conda environment.
    ```shell
    pip install --editable . # or for short: pip install -e .
    ```

Ok, all installations have done. You may speed up the installation by the following tips.

```{tip}
- [Speed up your Conda installs with Mamba](https://pythonspeed.com/articles/faster-conda-install/)
- Use the [THU Anaconda mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to speed up the Conda installation.
- Use the [THU PyPi mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to speed up the PyPI installation.
```

## Running an experiment

In Spiking-FullSubNet, we adopt a `recipes/<dataset>/<model>` directory structure. For example, let us entry to the directory `recipes/intel_ndns/`. The corresponding dataset is the Intel Neuromorphic DNS Challenge dataset. Please refer to [Intel Neuromorphic DNS Challenge Datasets](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset) for preparing the dataset.

To run an experiment of a model, we first go to a model directory. For example, we can entry to the directory `recipes/intel_ndns/spiking_fullsubnet/` to run an experiment of the `spiking_fullsubnet` model.

```shell
cd recipes/intel_ndns/spiking_fullsubnet/
```

In this `<model>` directory, we have an entry file `run.py`, dataloaders, and some model directories. We use HugoingFace Accelerate to start an experiment. Don't worry if you are not familiar with Accelerate. It will help you to run an parallel experiment easily. Please refer to [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/) for more details.

First, we need to configuration the GPU usage. Accelerate provides a CLI tool that unifies all launchers, so you only have to remember one command. To use it, run a quick configuration setup first on your machine and answer the questions:

```shell
accelerate config
```

Then, we can use the following command to train the `spiking_fullsubnet` model using configurations in `baseline_m_cumulative_laplace_norm.toml`:

```shell
accelerate launch run.py -C baseline_m_cumulative_laplace_norm.toml -M train
```