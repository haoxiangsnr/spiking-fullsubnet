# Logging and Visualization

After the training process is completed, the log information will be stored in the `save_dir` directory. Assuming that:

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
