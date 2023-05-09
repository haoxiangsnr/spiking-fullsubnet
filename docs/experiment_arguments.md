# Experiment arguments

AudioZEN uses TOML configuration files to configure the experiment. Check any experiment configuration file in the `recipes` directory for more details.

In the audiozen configuration file, we must contain the following sections:

- `meta`: Configure the experiment meta information, such as `name`, `save_dir`, `seed`, etc.
- `trainer`: Configure the trainer.
- `loss_function`: Configure the loss function.
- `lr_scheduler`: Configure the learning rate scheduler.
- `optimizer`: Configure the optimizer.
- `model`: Configure the model.
- `dataset`: Configure the dataset.
- `acoustics`: Configure the acoustic features.

## `meta`

The `meta` section is used to configure the experiment meta information, such as `name`, `save_dir`, `seed`, etc.

| Item                           | Description                                                                                                                                     |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `save_dir`                     | The directory where the experiment is saved. The log information, model checkpoints, and enhanced audio files will be stored in this directory. |
| `seed`                         | The random seed used to initialize the random number generator.                                                                                 |
| `use_amp`                      | Whether to use automatic mixed precision (AMP) to accelerate the training.                                                                      |
| `use_deterministic_algorithms` | Whether to use nondeterministic algorithms to accelerate the training. If it is True, the training will be slower but more reproducible.        |

## `trainer`

The `trainer` section is used to configure the trainer. The following configurations are used to configure the `Trainer`:

```toml
[trainer]
path = "trainer.Trainer"
[trainer.args]
max_epochs = 100
clip_grad_norm_value = 5
...
```

Using this configuration, AudioZEN will load the `Trainer` class from the current model directory and initialize it with the arguments in the `[trainer.args]` section.
`Trainer` class must be a subclass of `audiozen.trainer.base_trainer.BaseTrainer`. It supports the following arguments:

| Item                       | Description                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `max_epochs`               | The maximum number of epochs to train.                                                               |
| `clip_grad_norm_value`     | The maximum norm of the gradients used for clipping.                                                 |
| `save_max_score`           | Whether to find the best model by the maximum score.                                                 |
| `save_checkpoint_interval` | The interval of saving checkpoints.                                                                  |
| `patience`                 | The number of epochs with no improvement after which the training will be stopped.                   |
| `validation_interval`      | The interval of validation.                                                                          |
| `max_num_checkpoints`      | The maximum number of checkpoints to keep. Saving too many checkpoints causes disk space to run out. |

### Finding modules by `path` argument

We support multiple ways to find modules by the `path` in the configuration file. For example, we have the following directory structure:

```text
recipes/intel_ndns_challenge
├── README.md
├── run.py
└── sdnn_intel_ndns_challengedelays
    ├── baseline.toml
    ├── exp
    │   └── baseline
    │       └── baseline.log
    ├── model.py
    └── trainer.py
```

```py
sys.path = [
    '/path/to/audiozen/recipes/intel_ndns_challenge/sdnn_delays',
    '/path/to/audiozen/recipes/intel_ndns_challenge',
    ...
]
```

In `recipes/dns_1/baseline.toml`, the `path` of the `trainer` is set to:

```toml
[trainer]
path = "trainer.Trainer"
```

In this case, we will try to find the `Trainer` class in `recipes/dns_1/trainer`. If we set the `path` to:

```toml
[trainer]
path = "audiozen.trainer.custom_trainer.CustomTrainer"
```

We will try to find the `CustomTrainer` class in `audiozen/trainer/custom_trainer.py`.

!!! note

    If you want to call `Trainer` in `audiozen` package, you must install it in editable way by `pip install -e .` first.

## `loss_function`, `lr_scheduler`, `optimizer`, `model`, and `dataset`

`loss_function`, `lr_scheduler`, `optimizer`, `model`, `dataset` sections are used to configure the loss function, learning rate scheduler, optimizer, model, and dataset, respectively.
They have the same logic as the `trainer` section.

```toml
[loss_function|lr_scheduler|optimizer|model|dataset]
path = "..."
[loss_function|lr_scheduler|optimizer|model|dataset.args]
...
```

For example, you may use the loss function provided by PyTorch or implement your own loss function. For example, the following configuration is used to configure the `MSELoss`:

```toml
[loss_function]
path = "torch.nn.MSELoss"
[loss_function.args]
```

Configuration of a custom loss function:

```toml
[loss_function]
path = "audiozen.loss.MyLoss"
[loss_function.args]
weights = [1.0, 1.0]
...
```

!!! note

    You must keep the `[loss_function.args]` section even this loss function does not need any arguments.

You may use the learning rate scheduler provided by PyTorch or implement your own learning rate scheduler. For example, the following configuration is used to configure the `StepLR`:

```toml
[lr_scheduler]
path = "torch.optim.lr_scheduler.StepLR"
[lr_scheduler.args]
step_size = 100
gamma = 0.5
```

## `acoustics`

The `acoustics` section is used to configure the acoustic features.
These configurations are used for the whole project, like visualization, except for the `dataloader` and `model` sections.

| Item         | Description                                      |
| ------------ | ------------------------------------------------ |
| `sr`         | The sample rate of the audio.                    |
| `n_fft`      | The number of FFT points.                        |
| `hop_length` | The number of samples between successive frames. |
| `win_length` | The length of the STFT window.                   |
