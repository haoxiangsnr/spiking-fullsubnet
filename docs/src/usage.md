# Usage

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

## Logging and Visualization

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
