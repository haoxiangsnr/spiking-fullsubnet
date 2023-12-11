# Running an experiment using HuggingFace Accelerate

As mentioned in the previous section, AudioZEN adopts a `recipes/<dataset>/<model>` directory structure.
For example, let us entry to the directory `recipes/intel_ndns/`. The corresponding dataset is the Intel Neuromorphic DNS Challenge dataset. Please refer to [Intel Neuromorphic DNS Challenge Datasets](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset) for preparing the dataset.

To run an experiment of a model, we first go to a model directory. For example, we can entry to the directory `recipes/intel_ndns/spiking_fullsubnet/` to run an experiment of the `spiking_fullsubnet` model.

```shell
cd recipes/intel_ndns/spiking_fullsubnet/
```

## Entry file `run.py`

In this `<model>` directory, we have an entry file `run.py`, dataloaders, and some model directories. We use HugoingFace Accelerate to start an experiment. Please refer to [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/) for more details.

First, we need to configuration the GPU usage.  Accelerate provides a CLI tool that unifies all launchers, so you only have to remember one command. To use it, run a quick configuration setup first on your machine and answer the questions:

```shell
accelerate config
```

Then, we can use the following command to train the `spiking_fullsubnet` model using configurations in `baseline_m_cumulative_laplace_norm.toml`:

```shell
accelerate launch run.py -C baseline_m_cumulative_laplace_norm.toml -M train
```

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