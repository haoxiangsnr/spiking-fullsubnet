# Microsoft Deep Noise Suppression Challenge - Interspeech 2020

This directory contains the code for the experiments in the [Microsoft Deep Noise Suppression Challenge - Interspeech 2020](https://www.microsoft.com/en-us/research/event/deep-noise-suppression-challenge-interspeech-2020/).

## Results

|      |     stoi | pesq_wb | pesq_nb |   si_sdr |    sr |  len |    P808 |     SIG |     BAK |    OVRL |    pSIG |    pBAK |   pOVRL |
| ---: | -------: | ------: | ------: | -------: | ----: | ---: | ------: | ------: | ------: | ------: | ------: | ------: | ------: |
|    0 | 0.713908 | 1.18611 | 1.78374 | -7.95876 | 16000 |   10 | 2.35056 | 1.04576 | 1.05276 | 1.01393 | 3.05355 | 1.66791 | 1.72709 |

### `dns_16k_fsb_v1_mag_causal`

|                 With Reverb |    stoi | pesq_wb | pesq_nb |  si_sdr |    P808 |     SIG |     BAK |   OVRL |    pSIG |    pBAK |  pOVRL |
| --------------------------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -----: | ------: | ------: | -----: |
| `dns_16k_fsb_v1_mag_causal` | 0.93003 | 3.02291 | 3.52653 | 16.5924 | 3.23176 | 2.57365 | 2.53904 | 2.0986 | 3.28428 | 2.56574 | 2.2806 |

|                   No Reverb |     stoi | pesq_wb | pesq_nb |  si_sdr |    P808 |     SIG |     BAK |    OVRL |    pSIG |    pBAK |   pOVRL |
| --------------------------: | -------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: |
| `dns_16k_fsb_v1_mag_causal` | 0.964011 | 2.79921 | 3.32055 | 17.8226 | 3.95829 | 3.91861 | 4.29599 | 3.68653 | 3.87756 | 4.06713 | 3.48979 |
