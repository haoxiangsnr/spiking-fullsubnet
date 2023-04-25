import numpy as np
from numpy.typing import NDArray


def hz_to_bark(hz: float | NDArray) -> float | NDArray:
    return 26.81 / (1 + 1960.0 / hz) - 0.53


def bark_to_hz(bark: float | NDArray) -> float | NDArray:
    return 1960.0 / (26.81 / (0.53 + bark) - 1)


def bark_filter_bank(
    num_filters: int,
    n_fft: int,
    sr: int,
    low_freq: float,
    high_freq: float,
) -> NDArray:
    high_freq = high_freq or sr / 2
    assert high_freq <= sr / 2, "highfreq is greater than samplerate/2"

    low_bark = hz_to_bark(low_freq)
    high_bark = hz_to_bark(high_freq)
    barkpoints = np.linspace(low_bark, high_bark, num_filters + 2)
    bin = np.floor((n_fft + 1) * bark_to_hz(barkpoints) / sr)
    # bin = np.array(
    #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36, 38, 40, 42,
    #      44, 46, 48, 56, 64, 72, 80, 92, 104, 116, 128, 144, 160, 176, 192, 208, 232, 256])

    print(bin.shape)

    fbank = np.zeros([num_filters, n_fft // 2 + 1])
    for j in range(0, num_filters):
        print(j)
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return fbank
