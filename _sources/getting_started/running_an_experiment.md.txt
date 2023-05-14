# Running an experiment

As mentioned in the previous section, AudioZEN adopts a `recipes/<dataset>/<model>` directory structure.
To run an experiment of a model, we first go to a dataset directory, which will include an entry file `run.py` and some dataloaders dedicated to this dataset. For example, let us entry to the directory `recipes/intel_ndns/`. The corresponding dataset is the Intel Neuromorphic DNS Challenge dataset.
```shell
cd recipes/intel_ndns/
```

## Entry file `run.py`

In each `<dataset>` directory, we have an entry file `run.py`, dataloaders, and some model directories. We call the `run.py` script to run an experiment.
For example, we can use the following command to train the `sdnn_delays` model using configurations in `baseline.toml`:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=1
    run.py
    -C sdnn_delays/baseline.toml
    -M train
```

Here, we use `torchrun` to start the experiment.
`torchrun` isn't magic. It is a superset of `torch.distributed.launch` and is provided by PyTorch officials, helping us to start multi-GPU training conveniently. Its just a python `console_entrypoint` added for convenience (check [torchrun versus python -m torch.distributed.run](https://pytorch.org/docs/stable/elastic/run.html)). Check [Torchrun (Elastic Training)](https://pytorch.org/docs/stable/elastic/run.html) for more details.

`run.py` supports the following parameters:

| Parameter                | Description                                                                                                           | Default  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------- | -------- |
| `-C` / `--configuration` | The configuration file (`*.toml`) for the experiment.                                                                 | `None`   |
| `-M` / `--mode`          | The mode of the experiment. It can be `train`, `validate`, `test`, `predict`, or `finetune` or a combination of them. | `train`  |
| `-R` / `--resume`        | Resume the experiment from the latest checkpoint.                                                                     | `False`  |
| `--ckpt_path`            | The checkpoint path for test. It can be `best`, `latest`, or a path to a checkpoint file.                             | `latest` |

See more details in `recipes/intel_ndns/run.py` and `recipes/intel_ndns/sdnn_delays/baseline.toml`.

## Single-machine multi-GPU training

In most cases, we want to start an experiment on a single machine with multiple GPUs. Here, we show some examples for how to.

First, let us use `baseline.toml` to train `sdnn_delays` with two GPUs on a single machine:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    --configuration sdnn_delays/baseline.toml
    --mode train
```

`--nnodes=1` means that we will start the experiment on a single machine. `--nproc_per_node=2` means that we will use two GPUs on the single machine.

:::{attention}
The model `sdnn_delays` based on Lava-dl package, which actually does not support multi-GPU training. Here, we just use it as an example to show how to start an experiment on a single machine with multiple GPUs using `torchrun`.
:::

After a suspended experiment, we can resume training (using `-R` or `--resume`) from the last checkpoint:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    --C sdnn_delays/baseline.toml
    --M train
    -R
```

In the case of running multiple experiments on a single machine, since the first experiment has occupied the default `DistributedDataParallel` (DDP) listening port `29500`, we need to make sure that each instance (job) is setup on different ports to avoid port conflicts. Or you may directly use `rdzv_endpoint=localhost:0`, meaning to select a random unused port:

```shell
torchrun
    --rdzv_backend=c10d
    --rdzv_endpoint=localhost:0
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C sdnn_delays/baseline.toml
    -M train
```

Using "best" epoch to test the model performance on the test dataset:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C sdnn_delays/baseline.toml
    -M test
    --ckpt_path best
```

First to train the model on the training dataset. Then test the model performance on the test dataset:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C sdnn_delays/baseline.toml
    -M train test
    --ckpt_path best
```

:::{attention}
Before use `torchrun`, don't forget to use a environment variable `CUDA_VISIBLE_DEVICES` to control the GPU usage. For example, the following command will use the first and second GPUs:

```shell
export CUDA_VISIABLE_DEVICES=0,1
```
:::