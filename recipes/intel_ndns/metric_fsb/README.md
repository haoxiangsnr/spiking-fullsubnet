# FullSubNet with MetricGAN

## Recipe overview

### Only Generator

- `baseline_onlyGen`: Baseline model with only generator using time-domain and magnitude loss.
  - **9M** parameters
  - loss: time-domain loss + magnitude loss
  - trainer: `trainer_accelerate_gan_onlyGen`
  - base_trainer: `audiozen/trainer/base_trainer_gan_accelerate.py`
- `baseline_onlyGen_freq_MAE_mag_MAE`
  - trainer: `trainer_accelerate_gan_onlyGen_freq_MAE_mag_MAE` which changed the loss function to freqMAE + magMAE
- `baseline_onlyGen_freq_MAE_mag_MAE_SDR`
  - trainer: `trainer_accelerate_gan_onlyGen_freq_MAE_mag_MAE_SDR` which changed the loss function to freqMAE + magMAE + (0.01 * SDR)
- `baseline_onlyGen_freq_MAE_mag_MAE_lowSNR`
  - trainer: use 0.001 * SNR
- `baseline_onlyGen_freq_MAE_mag_MAE_multiframe`
  - 20 to 80 frequency bins use 3 frame size
- `baseline_onlyGen_freq_MAE_mag_MAE_dynamicDataloader`
  - dataloader: `dataloader_dynamic` which mixing data using dynamic mixing
- `baseline_onlyGen_freq_MAE_mag_MAE_dynamicDataloader_mos4_len1`
  - dataloader: `dataloader_dynamic` use `/datasets/datasets_fullband/datasets_fullband_16k/clean_fullband_mos4`


### Generator and Discriminator

- `baseline_freq_MAE_mag_MAE`
  - trainer: `trainer_accelerate_gan_freq_MAE_mag_MAE` which changed the loss function to freqMAE + magMAE


## Dataloader

```shell
Total number of files: 60000.
Total duration: 500 hr 0 min 0.00 s.
Average duration: 30.00 s.
```

- `dataloader`: load data following the official repo. It contains 60000 samples with 30s length. Total 500 hours. However, for each sample, we only use 6s length for speed up training. So, we have 83 hours data for training at each epoch.
- `dataloader_v2`:
  - support load segmental audio data directly using `subsample` function.
- `dataloader_dynamic`: mixing data using dynamic mixing. We set the sample length to 6s and total samples to 60000.

Note:

The data in `/datasets/datasets_fullband/datasets_fullband_16k/clean_fullband_mos4`
is with 16k sample rate, mos > 4.0, and length > 1.

## Trainer

- `trainer_accelerate_gan_onlyGen`: trainer only for generator.