# AudioZEN

AudioZEN is a toolkit for audio signal processing and deep learning. It is designed to be a lightweight and flexible audio signal processing and deep learning framework. It is built on top of PyTorch and provides standard audio signal processing and deep learning tools.

## Prerequisites

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
git clone git@github.com:haoxiangsnr/audiozen.git
cd audiozen

# Install PyPI dependencies
pip install -r requirements.txt

# Install audiozen package in editable mode and other PyPI dependencies
pip install -e .
```

If your network is unstable, you may use the [THU mirror site](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) to install PyPI dependencies.


## Project Structure

The project directory structure is as follows:

```shell
.
├── audiozen
│   ├── acoustics
│   ├── dataset
│   ├── model
│   │   ├── module
│   └── trainer
├── docs
│   └── audiozen
│       ├── acoustics
│       ├── dataset
│       └── trainer
├── notebooks
├── recipes
│   └── dns_icassp_2020
│       ├── cirm_lstm
│       ├── data
└── tools
```

- `audiozen/`: The core code of the project. It contains the following subdirectories:
  - `acoustics/`: The directory contains the code for audio signal processing.
  - `dataset/`: The directory contains the data loading and processing code.
  - `model/`: The directory contains the code for model definition and training.
  - `trainer/`: The directory contains the code for training and evaluation.
  - ...
- `docs/`: The directory contains the project's documentation.
- `recipes/`: The directory contains the code for experiments. Name the subdirectory after the dataset and create a subdirectory for the dataset named after the model. For example, `dns_icassp_2020/` represents the dataset `dns_icassp_2020`, and this directory contains data loading classes, training, and inference scripts for this dataset. `cirm_lstm/` contains the model for this dataset, including the structure and trainers for each model.
- `tools/`: The directory contains the code for additional tools, such as data preprocessing, model conversion, etc.

For the dataset directory, take `recipes/dns_icassp_2020` as an example. Its directory structure is as follows:

```shell
.
├── cirm_lstm
│   ├── baseline.toml
│   ├── model.py
│   └── trainer.py
├── dataset_train.py
├── dataset_validation_dns_1.py
├── dataset_validation_dns_4_non_personalized.py
└── run.py
```

The scripts in the `recipes/dns_icassp_2020` directory are common to all models in this dataset, covering entry files, data loading classes, etc.:
- `run.py`: The entry of the entire project, which can be used to train all models in the `dns_icassp_2020` directory.
- `dataset_train.py`: The construction class of the training dataset.
- `dataset_validation_dns_1.py`: The construction class of the first validation dataset.
- `dataset_validation_dns_4_non_personalized.py`: The construction class of the second validation dataset.

In addition, the `recipes/dns_icassp_2020` directory can contain multiple model directories, each corresponding to a model. Each model directory contains:

- `<exp_id>.toml`: The training parameters for this model.
- `trainer.py`: The trainer for this model, which contains the operations and operations to be executed in each training, validation and test round.
- `model.py`: The structure of the current model.
- `run.py` (optional): The entry of the current model, which can be used to train the current model. If this file is not present, the `run.py` file in the `recipes/dns_icassp_2020` directory will be used.

## Usage

First, we need to enter a data directory, such as `recipes/dns_icassp_2020` directory, and call the `run.py` script:

```shell
cd recipes/dns_icassp_2020

# Use baseline.toml to train cirm_lstm with 2 GPUs on a single machine
CUDA_VISIABLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py -C cirm_lstm/baseline.toml -M train

# Use baseline.toml to train cirm_lstm with 1 GPU on a single machine
CUDA_VISIABLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 run.py -C cirm_lstm/baseline.toml -M train

# Use baseline.toml to train cirm_lstm with 2 GPUs on a single machine, and resume training from the last checkpoint
CUDA_VISIABLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py -C cirm_lstm/baseline.toml -M train -R

# In the case of running multiple experiments on a single machine, since the first experiment has occupied the default DistributedDataParallel (DDP) listening port 29500, we can use need to make sure that each instance (job) is setup on different ports to avoid port conflicts
# rdzv-endpoint=localhost:0 means to select a random unused port
CUDA_VISIABLE_DEVICES=0,1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 run.py -C cirm_lstm/baseline.toml -M train

# Test the model performance on the test dataset
CUDA_VISIABLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py -C cirm_lstm/baseline.toml -M test --ckpt_path best

# First, use the baseline.toml to train cirm_lstm with 2 GPUs on a single machine, and then test the model performance on the test dataset
CUDA_VISIABLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py -C cirm_lstm/baseline.toml -M train test --ckpt_path best
```

Note that `torchrun` isn't a magic wrapper, its just a python console_entrypoint added for convenience (check [torchrun versus python -m torch.distributed.run](https://pytorch.org/docs/stable/elastic/run.html)).

`run.py` supports the following parameters:

```shell
usage: run.py [-h] -C CONFIGURATION [-M {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...]] [-R] [--ckpt_path CKPT_PATH]

Audio-ZEN

options:
  -h, --help            show this help message and exit
  -C CONFIGURATION, --configuration CONFIGURATION
                        Configuration (*.toml).
  -M {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...], --mode {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...]
                        Mode of the experiment.
  -R, --resume          Resume the experiment from latest checkpoint.
  --ckpt_path CKPT_PATH
                        Checkpoint path for test. It can be 'best', 'latest', or a path to a checkpoint.
usage: run.py [-h] -C CONFIGURATION [-M {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...]] [-R] [--ckpt_path CKPT_PATH]

Audio-ZEN

options:
  -h, --help            show this help message and exit
  -C CONFIGURATION, --configuration CONFIGURATION
                        Configuration (*.toml).
  -M {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...], --mode {train,validate,test,predict,finetune} [{train,validate,test,predict,finetune} ...]
                        Mode of the experiment.
  -R, --resume          Resume the experiment from latest checkpoint.
  --ckpt_path CKPT_PATH
                        Checkpoint path for test. It can be 'best', 'latest', or a path to a checkpoint.
```

See more details in `recipes/dns_icassp_2020/run.py` and the configuration file `recipes/dns_icassp_2020/cirm_lstm/baseline.toml`.

### Logging and Visualization

The log information generated during the training process will be stored. Assuming that:

- The file name of the training configuration file is: `baseline.toml`
- The value of the `save_dir` parameter in the training configuration file `basline.toml` is `~/exp`

Then the log information will be stored in the `~/exp/baseline` directory, which contains the following information:

```shell
.
├── baseline.log
├── checkpoints
├── config__2023_01_13--10_27_42.toml
├── enhanced
└── tb_log
    └── events.out.tfevents.1673576862.VM-97-67-ubuntu.3747605.0
```

- `baseline.log`: log information
- `checkpoints/`: model checkpoints
- `config__2023_04_13--10_27_42.toml`: training configuration file
- `enhanced`: enhanced audio files when running in test mode
- `tb_log/`: TensorBoard log information, we can visualize it through TensorBoard

Currently, we only support TensorBoard for visualization. Assuming that the value of the `save_dir` parameter in the training configuration file `basline.toml` is `~/exp`, then we can use the following command to visualize the log information:

```shell
tensorboard --logdir ~/exp
```
