# FullSubNet with MetricGAN

## Recipe overview

- `baseline_onlyGen`: Baseline model with only generator using time-domain and magnitude loss.
  - **9M** parameters
  - loss: time-domain loss + magnitude loss
  - trainer: `trainer_accelerate_gan_onlyGen`
  - base_trainer: `audiozen/trainer/base_trainer_gan_accelerate.py`
- `baseline_onlyGen_freq_MAE_mag_MAE`: only changed loss function to MAE in frequency domain and MAE in magnitude domain
  - trainer: `trainer_accelerate_gan_onlyGen_freq_MAE_mag_MAE` which changed the loss function
- `baseline_onlyGen_freq_MAE_mag_MSE_dynamicDataloader`
  - dataloader: `dataloader_dynamic` which mixing data using dynamic mixing

## Dataloader

```shell
Total number of files: 60000.
Total duration: 500 hr 0 min 0.00 s.
Average duration: 30.00 s.
```

- `dataloader`: load data following the official repo. It contains 60000 samples with 30s length. Total 500 hours. However, for each sample, we only use 6s length for speed up training. So, we have 83 hours data for training at each epoch.
- `dataloader_dynamic`: mixing data using dynamic mixing. We set the sample length to 6s and total samples to 60000.


## Trainer

- `trainer_accelerate_gan_onlyGen`: trainer only for generator