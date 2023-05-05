# Prerequisites

To run experiments, you must use a Linux-based operating system that does not support the Windows platform. In addition, you need to install Anaconda or Miniconda, which will be used to create a virtual environment and install the dependencies. To use GPU parallel training, you must install CUDA (10.2+) and the corresponding version of the CuDNN acceleration library.

Create a conda virtual environment and install the dependencies:

```shell
# Create a conda virtual environment
conda create --name audiozen python=3.10
conda activate audiozen

# Install PyTorch. Although pytorch=1.12.1 has been tested, you may also use other versions. Use CUDA 10.2 as an example:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

# Install other dependencies, which are recommended to be installed using conda
conda install tensorboard joblib matplotlib

# (Optional) If there are "mp3" format data in your dataset, you need to install ffmpeg
conda install -c conda-forge ffmpeg
```

If your network is not stable, you may use the [THU mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) to install conda dependencies.

Clone the repository and install PyPI dependencies:

```shell
# Clone the repository
git clone git@github.com:haoxiangsnr/audiozen.git && cd audiozen

# Install PyPI dependencies
pip install -r requirements.txt

# Install audiozen package in editable mode and other PyPI dependencies
pip install -e .
```

If your network is unstable, you may use the [THU mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to install PyPI dependencies.
