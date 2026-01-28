# Spiking-FullSubNet

Spiking-FullSubNet is the winner solution of Intel N-DNS Challenge Track 1 (Algorithmic). This repository serves as the official home of the Spiking-FullSubNet implementation. Here, you will find:

- A PyTorch-based implementation of the Spiking-FullSubNet model described in our paper "Toward Ultralow-Power Neuromorphic Speech Enhancement With Spiking-FullSubNet".
- Scripts for training the model and evaluating its performance.
- The pre-trained models in the `model_zoo` directory, ready to be further fine-tuned on the other datasets.
- The frozen version of the solution used in the Intel N-DNS Challenge in the `38fe020` commit.

## Updates

- 2026-01-27: The `main` branch shows the implementation of the Spiking-FullSubNet model as described in our published paper "Toward Ultralow-Power Neuromorphic Speech Enhancement With Spiking-FullSubNet", which includes several improvements and optimizations over the challenge version. We recommend using this branch for the citation and further research.
- 2024-02-26: The **frozen version**, which serves as a backup for the submitted solution used in the Intel N-DNS Challenge. This solution has been checked and verified by Intel during the challenge. If you need to check the experimental results from that time, please refer to this specific commit: [38fe020](https://github.com/haoxiangsnr/spiking-fullsubnet/tree/38fe020cdb803d2fdc76a0df4b06311879c8e370). There you will find everything you need. After switching to this commit, you can place the checkpoints from the `model_zoo` into the `exp` directory and use `-M test` for inference or `-M train` to retrain the model. After challenge, we made some improvements and optimizations to the solution and published a paper (IEEE TNNLS) based on these improvements. Please check the `main` branch for the published paper version.

## Quick Start

You can either clone the repository, setup an environment and start with the scripts, or directly open in Colab (under construction).

## Environment Setup

We really like [uv](https://docs.astral.sh/uv/) and recommend using it as your package manager. But feel free to use whichever you prefer.

> [!TIP]
> uv is significantly faster (10~100x) than pip and handles dependency resolution more reliably.
> The `uv.lock` file ensures reproducible installations across different machines.


```bash
# Clone the repository
git clone git@github.com:haoxiangsnr/spiking-fullsubnet.git && cd spiking-fullsubnet

# [Optional] Install uv
# Check https://docs.astral.sh/uv/ for other installation methods
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
# This will:
# - Create a virtual environment in `.venv`
# - Install all dependencies from `uv.lock`
# - Install `audiozen` folder in editable mode so you can import it everywhere
uv sync --all-extras

# Activate the virtual environment
source .venv/bin/activate
```

If you prefer Conda/pip, you can still use the traditional approach:

```bash
git clone git@github.com:haoxiangsnr/spiking-fullsubnet.git && cd spiking-fullsubnet

conda create --name spiking-fullsubnet python=3.10
conda activate spiking-fullsubnet

# torch==2.1.1 and torch==2.10 have been tested to work well with this codebase
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies and the `audiozen` folder in editable mode
# Please check the `pyproject.toml` for the full list of dependencies
pip install -e .
```

## Inference on Validation Set

Since the official test set requires a long time (10 hours+) for inference on a single GPU,
we provide a mini-validation set to quickly verify. The validation set contains 341 noisy-clean pairs generated in the same way as the official test set. In our experience, performance gains on this set are highly positively correlated with the official test set.

```bash
# Download validation set from Github Releases
cd <your_project_root>
mkdir data && cd data

# Download and extract validation set
wget https://github.com/haoxiangsnr/spiking-fullsubnet/releases/download/data/validation_set.tar.gz
tar -xzvf validation_set.tar.gz

# folder structure:
.
└── data
    ├── validation_set
    │   ├── clean
    │   │   ├── clean_fileid_119.wav
    │   │   ├── clean_fileid_165.wav
    │   │   └── clean_fileid_7.wav
    │   ├── noise
    │   │   ├── noise_fileid_27.wav
    │   │   ├── noise_fileid_312.wav
    │   │   └── noise_fileid_4.wav
    │   └── noisy
    │       ├── book_00588_chp_0003_..._fileid_115.wav
    │       ├── book_09739_chp_0003_..._fileid_275.wav
    │       └── German_Wikiped_..._fileid_246.wav
    └── validation_set.tar.gz
```

To run inference on the validation set using the pre-trained model, use the following command:

```bash
cd <your_project_root>

# Download pre-trained model from Github Releases
wget https://github.com/haoxiangsnr/spiking-fullsubnet/releases/download/ckpt-epoch-188/epoch_0188.zip

# Unzip the pre-trained model to the correct directory
mkdir -p recipes/intel_ndns/spiking_fullsubnet_v2/exp/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8/checkpoints/

unzip epoch_0188.zip -d recipes/intel_ndns/spiking_fullsubnet_v2/exp/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8/checkpoints/epoch_0188

# Inference on validation set
accelerate launch --multi_gpu \
    --num_processes=4 \
    --gpu_ids 0,1,2,3 \
    --main_process_port 46601 \
    run.py \
    --config_path conf/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8.yaml \
    --eval_batch_size 4 \
    --resume_from_checkpoint /home/xhao/proj/spiking-fullsubnet/recipes/intel_ndns/spiking_fullsubnet_v2/exp/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8/checkpoints/epoch_0188 \
    --do_eval true 
```

Depending on your software environment, you may have the results like below:

|        set |  si_sdr |    P808 |    OVRL |     SIG |     BAK |
| ---------: | ------: | ------: | ------: | ------: | ------: |
| validation | 15.0127 | 3.61135 | 3.01281 | 3.33227 | 3.93021 |

## Citation

If you find this repository useful for your research, please consider citing the following papers:

```bibtex
@ARTICLE{hao2025toward,
  author={Hao, Xiang and Ma, Chenxiang and Yang, Qu and Wu, Jibin and Tan, Kay Chen},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Toward Ultralow-Power Neuromorphic Speech Enhancement With Spiking-FullSubNet}, 
  year={2025},
  volume={36},
  number={9},
  pages={17350-17364},
  doi={10.1109/TNNLS.2025.3566021}}

@INPROCEEDINGS{hao2024when,
  author={Hao, Xiang and Ma, Chenxiang and Yang, Qu and Tan, Kay Chen and Wu, Jibin},
  booktitle={2024 IEEE Conference on Artificial Intelligence (CAI)}, 
  title={When Audio Denoising Meets Spiking Neural Network}, 
  year={2024},
  volume={},
  number={},
  pages={1524-1527},
  doi={10.1109/CAI59869.2024.00275}}
```

## License

This project is licensed under the [MIT License](LICENSE).