# Running an experiment

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

Then, we can use the following command to train the `spiking_fullsubnet` model using configurations in `baseline_m.toml`:

```shell
accelerate launch run.py -C baseline_m.toml -M train
```

```{note}
Alternatively, if you don't want to use the CLI tool, you may use explicit arguments to specify the GPU usage (https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-env). For example: `accelerate launch --multi_gpu --num_processes=6 --gpu_ids 0,1,2,3,4,5 --main_process_port 46524 --main_process_ip 127.0.0.1 run.py -C baseline_m.toml`
```


## Entry file `run.py`

In this `<model>` directory, we have an entry file `run.py`, dataloaders, and some model directories. We use HuggingFace Accelerate to start an experiment. Please refer to [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/) for more details.

`run.py` supports the following parameters:

| Parameter                | Description                                                                                                           | Default  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- | -------- |
| `-C` / `--configuration` | The configuration file (`*.toml`) for the experiment.                                                                 | `None`   |
| `-M` / `--mode`          | The mode of the experiment. It can be `train`, `validate`, `test`, `predict`, or `finetune` or a combination of them. | `train`  |
| `-R` / `--resume`        | Resume the experiment from the latest checkpoint.                                                                     | `False`  |
| `--ckpt_path`            | The checkpoint path for test. It can be `best`, `latest`, or a path to a checkpoint file.                             | `latest` |

See more details in `recipes/intel_ndns/spiking_fullsubnet/run.py`.


After a suspended experiment, we can resume training (using `-R` or `--resume`) from the last checkpoint:

```shell
accelerate launch run.py
    -C baseline_m_cumulative_laplace_norm.toml
    --M train
    -R
```

Using "best" epoch to test the model performance on the test dataset:

```shell
accelerate launch run.py
    -C baseline_m_cumulative_laplace_norm.toml
    --M test
    --ckpt_path best
```