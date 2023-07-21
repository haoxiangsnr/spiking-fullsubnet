# Intel N-DNS Challenge Track 1

## Dataset

### Training set

Please follow the [instructions](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset) provided by the Intel N-DNS Challenge organizers to download the training set. If you can access servers (`10.21.4.69` and `10.21.4.65`) in PolyU, you may directly use the simulated data at `/datasets/datasets_fullband/training_set`.

Note that the origianl dataset is very large (about 700 GB). We recommend you to download the training set as early as possible.

### Validation set

Please download the validation set from the release page.
If you can access servers (`10.21.4.69` and `10.21.4.65`) in PolyU, you may directly use the simulated data at `/datasets/datasets_fullband/validation_set`.

## Reference

|                Entry                | <sub>$\text{SI-SNR}$ <sup>(dB) | <sub>$\text{SI-SNRi}$ <sup>data (dB) | <sub>$\text{SI-SNRi}$ <sup>enc+dec (dB) | <sub>$\text{MOS}$ <sup>(ovrl) | <sub>$\text{MOS}$ <sup>(sig) | <sub>$\text{MOS}$ <sup>(bak) | <sub>$\text{latency}$ <sup>enc+dec (ms) | <sub>$\text{latency}$ <sup>total (ms) | <sub>$\text{Power}$ $\text{proxy}$ <sup>(M-Ops/s) | <sub>$\text{PDP}$ $\text{proxy}$ <sup>(M-Ops) | <sub>$\text{Params}$ <sup>($\times 10^3$) | <sub>$\text{Size}$ <sup>(KB) |
| :---------------------------------: | -----------------------------: | -----------------------------------: | --------------------------------------: | ----------------------------: | ---------------------------: | ---------------------------: | --------------------------------------: | ------------------------------------: | ------------------------------------------------: | --------------------------------------------: | ----------------------------------------: | ---------------------------: |
|           Validation set            |                           7.62 |                                    - |                                       - |                          2.45 |                         3.19 |                         2.72 |                                       - |                                     - |                                                 - |                                             - |                                         - |                            - |
|    Microsoft NsNet2 (02/20/2023)    |                          11.89 |                                 4.26 |                                    4.26 |                          2.95 |                         3.27 |                         3.94 |                                   0.024 |                                20.024 |                                            136.13 |                                          2.72 |                                     2,681 |                       10,500 |
| Intel proprietary DNS (02/28/2023)  |                          12.71 |                                 5.09 |                                    5.09 |                          3.09 |                         3.35 |                         4.08 |                                   0.036 |                                 8.036 |                                                 - |                                             - |                                     1,901 |                        3,802 |
| Baseline SDNN solution (02/20/2023) |                          12.50 |                                 4.88 |                                    4.88 |                          2.71 |                         3.21 |                         3.46 |                                   0.036 |                                 8.036 |                                             11.59 |                                          0.09 |                                       525 |                          465 |

## Results

|                Entry                | <sub>$\text{SI-SNR}$ <sup>(dB) | <sub>$\text{SI-SNRi}$ <sup>data (dB) | <sub>$\text{SI-SNRi}$ <sup>enc+dec (dB) | <sub>$\text{MOS}$ <sup>(ovrl) | <sub>$\text{MOS}$ <sup>(sig) | <sub>$\text{MOS}$ <sup>(bak) | <sub>$\text{latency}$ <sup>enc+dec (ms) | <sub>$\text{latency}$ <sup>total (ms) | <sub>$\text{Power}$ $\text{proxy}$ <sup>(M-Ops/s) | <sub>$\text{PDP}$ $\text{proxy}$ <sup>(M-Ops) | <sub>$\text{Params}$ <sup>($\times 10^3$) | <sub>$\text{Size}$ <sup>(KB) |
| :---------------------------------: | -----------------------------: | -----------------------------------: | --------------------------------------: | ----------------------------: | ---------------------------: | ---------------------------: | --------------------------------------: | ------------------------------------: | ------------------------------------------------: | --------------------------------------------: | ----------------------------------------: | ---------------------------: |
|           Validation set            |                           6.89 |                                    - |                                       - |                          2.54 |                         3.40 |                         2.49 |                                       - |                                     - |                                                 - |                                             - |                                         - |                            - |
| Baseline SDNN (audiozen on epoch 7) |                          11.17 |                                 4.88 |                                    4.88 |                          2.81 |                         3.43 |                         3.09 |                                   0.036 |                                 8.036 |                                             11.59 |                                          0.09 |                                       525 |                          465 |
|                 FSB                 |                          17.84 |                                 4.88 |                                    4.88 |                          3.60 |                         3.85 |                         4.27 |                                   0.036 |                                 8.036 |                                             22680 |                                        182.26 |                                      9301 |                              |
|             FSB (tiny)              |                          16.24 |                                 4.88 |                                    4.88 |                          3.48 |                         3.72 |                         4.23 |                                   0.036 |                                 8.036 |                                            299.84 |                                          2.41 |                                      2451 |                              | 84 |

## Results (polyfit)

|           |    P808 |    OVRL |    SIG |     BAK |  si_sdr |
| --------: | ------: | ------: | -----: | ------: | ------: |
|     Noisy | 3.04809 | 2.39574 | 3.0967 | 2.66386 | 6.89017 |
| FSB + GAN | 3.11288 | 2.65288 | 3.0765 | 3.61572 |   -0.38 |
