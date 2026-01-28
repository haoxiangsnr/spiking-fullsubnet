# Spiking-FullSubNet

Spiking-FullSubNet is the 1st Place Winner solution of the Intel N-DNS Challenge (Track 1: Algorithmic). This repository serves as the official implementation of our paper: "Toward Ultralow-Power Neuromorphic Speech Enhancement With Spiking-FullSubNet" (IEEE TNNLS, 2025).

- **SOTA Performance:** Outperforms standard baselines on the Intel N-DNS benchmark.
- **Ultra-low Power:** Designed with Spiking Neural Networks (SNN) for neuromorphic hardware efficiency.
- **Reproducible:** We provide pre-trained models and a carefully selected validation set for quick verification.

## Updates

- [2026.01] **Paper Accepted:** Our paper has been accepted by IEEE TNNLS. The main branch now hosts the improved, research-friendly version of the model (recommended for citation & research). The current codebase has been updated to reflect the changes in the published paper.
- [2024-02] **Frozen Version:** This serves as a backup of the submitted solution used in the Intel N-DNS Challenge. This solution was checked and verified by Intel during the challenge. If you need to check the experimental results from that time, please refer to this specific commit: [38fe020](https://github.com/haoxiangsnr/spiking-fullsubnet/tree/38fe020cdb803d2fdc76a0df4b06311879c8e370). There you will find everything you need. After switching to this commit, you can place the checkpoints from the `model_zoo` into the `exp` directory and use `-M test` for inference or `-M train` to retrain the model. After the challenge, we made improvements and optimizations to the solution and published a paper (IEEE TNNLS) based on these improvements. Please check the `main` branch for the published paper version.

## Quick Start

You can either clone the repository, setup an environment and start with the scripts, or directly open in Colab (under construction...).

## Environment Setup

We really like [uv](https://docs.astral.sh/uv/) and recommend using it as your package manager, but feel free to use whichever you prefer.

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

## Dataset Preparation

Since the official test set requires a long time (10+ hours) for inference on a single GPU,
we provide a mini-validation set to quickly verify. The validation set contains 341 noisy-clean pairs generated in the same way as the official test set. In our experience, performance gains on this set are highly positively correlated with the official test set.

### 1. Mini-Validation Set (Recommended)

Running the full official test set typically requires 8 GPUs and over 10 hours. **For quick verification**, we provide a **Mini-Validation Set (341 samples)**.

> **Note:** In our experiments, performance trends on this mini-set are highly positively correlated with the official test set.

```bash
cd <your_project_root>
mkdir -p data
cd data

# Download validation set (hosted on GitHub Releases)
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

cd ..
```

### 2. Official Test Set (Optional)

If you require the full Intel N-DNS test set for comparison, we have hosted a backup copy here:  https://github.com/haoxiangsnr/IntelNeuromorphicDNSChallenge/releases.

The official test set is large (about 37 GB, 12000 files). Please ensure you have sufficient disk space and a stable internet connection before downloading.

The method for downloading and extracting the official test set is similar to the mini Validation Set.

## Inference

To run inference on the validation set using the pre-trained model, use the following commands:

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

On the Mini Validation Set, you should obtain results close to:

|        set |  si_sdr |    P808 |    OVRL |     SIG |     BAK |
| ---------: | ------: | ------: | ------: | ------: | ------: |
| validation | 15.0127 | 3.61135 | 3.01281 | 3.33227 | 3.93021 |

On the Official Test Set, you should obtain results close to:

|  set |  si_sdr |    P808 |    OVRL |     SIG |     BAK |
| ---: | ------: | ------: | ------: | ------: | ------: |
| test | 15.2008 | 3.62772 | 3.03368 | 3.35132 | 3.94128 |


## Project Structure

You don't need to understand project structure to run experiments. However, if you want to modify the code or add new models, this section may help you.

<details>

<summary>Click to expand</summary>

Let's take a look at the overall structure of the project. You may familiar with this project structure (`recipes/<dataset>/<model>`) if you have used [ESPNet](https://github.com/espnet/espnet) and [SpeechBrain](https://github.com/speechbrain/speechbrain) before.
This project is inspired by them, but it is more simpler. This project includes a core package (`audiozen/`) and a series of training recipes (`recipes/`).The core package is named `audiozen`, which provides common audio signal processing tools and deep learning trainers. As we have installed `audiozen` in editable mode, we can call `audiozen` package everywhere in the project. In addition, we can modify the source code of `audiozen` package directly. Any changes to the original package would reflect directly in your environment. For example, we can call `audiozen` package in `recipes` folder to train models on specific datasets and call `audiozen` package in `tools` folder to preprocess data. The recipes in the `recipes` folder are used to research the audio/speech signal processing. The recipe concept was introduced by [Kaldi](https://kaldi-asr.org/doc/about.html) first, providing a convenient and reproducible way to organize and save the deep learning training pipelines.


The directory structure is as follows:

```shell
├── audiozen/
│   ├── acoustics/
│   ├── dataset/
│   ├── model/
│   │   ├── module/
│   └── trainer/
├── docs/
├── notebooks/
├── recipes/
│   └── intel_ndns/
│       ├── sdnn_delays/
│       │   ├── baseline.toml
│       │   ├── model.py
│       │   └── trainer.py
│       ├── dataloader.py
│       ├── loss.py
│       └── run.py
└── tools/
```


- `audiozen/`: The core of the project. After installing `audiozen` in the editable mode, we can call `audiozen` package everywhere in the project.
    - `acoustics/`: Contain the code for audio signal processing.
    - `dataset/`: Contain the data loading and processing code.
    - `model/`: Contain the code for model definition and training.
    - `trainer/`: Contain the code for training and evaluation.
    - ...
- `docs/`: Contains the project's documentation. We use [Sphinx Documentation Generator](https://www.sphinx-doc.org/en/master/) to build the documentation.
- `recipes/`: Contains the recipes for specific experiments. It follows a `<dataset_name>/<model_name>` structure.
- `tools/`: Contains the code for additional tools, such as data preprocessing, model conversion, etc.

In the `recipes` folder, we name the subdirectory after the dataset. create a subdirectory for the dataset named after the model.
For example, `recipes/intel_ndns/` saves the models trained on the Intel Neuromorphic DNS Challenge dataset. It contains commonly-used data loading classes, training, and inference scripts.

- `run.py`: The entry of the entire project, which can be used to train and evaluate all models in the `intel_ndns` directory.
- `dataloader.py`: The data loading and processing code for the Intel Neuromorphic DNS Challenge dataset.
- `loss.py`: The loss function commonly used in the Intel Neuromorphic DNS Challenge dataset.

</details>

## Logging and Visualization

You don't need to understand logging and visualization to run experiments. However, if you want to monitor the training process, this section may help you.

<details>
<summary>Click to expand</summary>

After the training process has been completed, the log information will be stored in the `save_dir` directory. Assuming that:


- The filename of the training configuration file is: `baseline.toml`
- The value of the `save_dir` parameter in the `baseline.toml` is `sdnn_delays/exp`

Then, the log information will be stored in the `sdnn_delays/exp/baseline` directory, which contains the following information:

```shell
.
├── baseline.log
├── checkpoints
├── config__2023_01_13--10_27_42.toml
├── enhanced
└── tb_log
    └── events.out.tfevents.1673576862.VM-97-67-ubuntu.3747605.0
```

- `baseline.log`: the log information.
- `checkpoints/`: model checkpoints.
- `config__2023_04_13--10_27_42.toml`: a backup of the training configuration file.
- `enhanced`: the enhanced audio files when running in test mode
- `tb_log/`: `tensorBoard` log information, we can visualize it through TensorBoard

Currently, we only support TensorBoard for visualization. Assuming that the value of the `save_dir` parameter in the `basline.toml` is `sdnn_delays/exp`, then we can use the following command to visualize the log information:

```shell
tensorboard --logdir sdnn_delays/exp --bind_all
```
</details>


## Citation

If you find this repository useful, please consider citing our work:

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