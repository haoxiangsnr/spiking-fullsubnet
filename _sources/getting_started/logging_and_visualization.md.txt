# Logging and Visualization

Now, the training process has been completed. The log information will be stored in the `save_dir` directory. Assuming that:

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
