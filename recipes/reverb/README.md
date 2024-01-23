# REVERB Challenge dataset

REVERB challenge is to inspire and evaluate diverse ideas for speech enhancement and robust automatic speech recognition in reverberant environments.

The REVERB challenge data is currently available only through LDC. For more details, please visit the link: https://reverb2014.dereverberation.com/instructions.html.

## Inference

If you want to use a pretrained model, please refer to the following example command:

```shell
accelerate launch --multi_gpu --num_processes=4 --gpu_ids 0,1,2,3 --main_process_port 46599 run.py -C default.toml -M predict --ckpt_path /home/xhao/proj/spiking-fullsubnet/recipes/reverb/spiking_fullsubnet/exp/default/checkpoints/epoch_0155
```

Some notes:
- As the metrics on the test set only must be evaluated by MatLab, you should run the command using the `predict` mode, which will output the audio samples of the model.
- The `--ckpt_path` is the path of the pretrained model.
- Using `predict` mode, the output audio samples of the model will be saved in a evaluation directory, which is corresponding to the shell script of the REVERB Challenge dataset. For example. we output the audio samples in the `/nfs/xhao/data/reverb_challenge/kaldi/egs/reverb/s5/wav/spiking_fullsubnet` directory.