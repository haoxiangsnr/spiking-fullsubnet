# Running an experiment

AudioZEN adopts a `recipes/<dataset>/<model>` direcotry structure.
To run an experiment of a model, we first need to enter a dataset direcotry, which will include a entry file `run.py` and some dataloaders dedicated to this dataset. For example, let us entry to the directory `recipes/dns_icassp_2020/`. The correspoding dataset is the ICASSP 2020 DNS Challenge dataset:

```shell
cd recipes/dns_icassp_2020
```

## Entry file `run.py`

In each `<dataset>` directory, we have a entry file `run.py`, dataloaders, and some model direcotries.
Then, we call this `run.py` script to run the experiment. For example, we can use the following command to train the `cirm_lstm` model using configurations in `baseline.toml`:

```shell
torchrun run.py -C cirm_lstm/baseline.toml -M train
```

Here, `torchrun` helps us to start multi-GPU training conveniently. `torchrun` isn't a magic, its just a python `console_entrypoint` added for convenience (check [torchrun versus python -m torch.distributed.run](https://pytorch.org/docs/stable/elastic/run.html)).

`run.py` supports the following parameters:

| Parameter                 | Description                                                                                                           | Default  |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------- |
| `-C` or `--configuration` | The configuration file (*.toml) for the experiment.                                                                   | `None`   |
| `-M` or `--mode`          | The mode of the experiment. It can be `train`, `validate`, `test`, `predict`, or `finetune` or a combination of them. | `train`  |
| `-R` or `--resume`        | Resume the experiment from the latest checkpoint.                                                                     | `False`  |
| `--ckpt_path`             | The checkpoint path for test. It can be `best`, `latest`, or a path to a checkpoint.                                  | `latest` |

See more details in `recipes/dns_icassp_2020/run.py` and the configuration file `recipes/dns_icassp_2020/cirm_lstm/baseline.toml`.

## Single-machine multi-GPU training

In most cases, we want to start an experiment on a single machine with multiple GPUs. Here, we show some examples for how to. First, let us use `baseline.toml` to train `cirm_lstm` with 2 GPUs on a single machine

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    --configuration cirm_lstm/baseline.toml
    --mode train
```

Use baseline.toml to train cirm_lstm with 1 GPU on a single machine

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=1
    run.py
    --configuration cirm_lstm/baseline.toml
    --mode train
```

Use `baseline.toml` to train cirm_lstm with 2 GPUs on a single machine, and resume training (using `-R` or `--resume`) from the last checkpoint:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C cirm_lstm/baseline.toml
    -M train
    -R
```

In the case of running multiple experiments on a single machine, since the first experiment has occupied the default DistributedDataParallel (DDP) listening port 29500, we need to make sure that each instance (job) is setup on different ports to avoid port conflicts. Use `rdzv_endpoint=localhost:0` means to select a random unused port:

```shell
torchrun
    --rdzv_backend=c10d
    --rdzv_endpoint=localhost:0
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C cirm_lstm/baseline.toml
    -M train
```

Using "best" epoch to test the model performance on the test dataset:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C cirm_lstm/baseline.toml
    -M test
    --ckpt_path best
```

Sequential train the model on the training dataset and test the model performance on the test dataset:

```shell
torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=2
    run.py
    -C cirm_lstm/baseline.toml
    -M train test
    --ckpt_path best
```

:::{attention}
Before use `torchrun`, don't forget to use the environment variable `CUDA_VISIBLE_DEVICES` to control the GPU usage. For example, the following command will use the first and second GPUs:

```shell
export CUDA_VISIABLE_DEVICES=0,1
```
:::