import random

import h5py
import librosa
import numpy as np
import torch
from scipy import signal
from torch.utils.data import Dataset


def tailor_dB_FS(y, target_dB_FS=-25, eps=1e-6):
    rms = np.sqrt(np.mean(y**2))
    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(y, clipping_threshold=0.999):
    return any(np.abs(y) > clipping_threshold)


def norm_amplitude(y, scalar=None, eps=1e-6):
    if not scalar:
        scalar = np.max(np.abs(y)) + eps

    return y / scalar, scalar


class DNS_OTF(Dataset):
    def __init__(
        self,
        speech_path,
        noise_path,
        rir_path,
        input_sr,
        target_sr,
        snr_range,
        reverb_proportion,
        target_dB_FS,
        target_dB_FS_floating_value,
        num_iter_per_epoch,
        session_length,
    ):
        super(DNS_OTF, self).__init__()

        print("Loading Speech data...")
        self.speech_path = open(speech_path).readlines()
        print("Loading Noise data...")
        self.noise_path = open(noise_path).readlines()
        print("Loading rir_path data...")
        self.rir_path = open(rir_path).readlines()

        self.speech_dataset = []
        for i in range(len(self.speech_path)):
            print(i)
            self.speech_dataset.append(h5py.File(self.speech_path[i].strip(), "r"))

        self.noise_dataset = []
        for i in range(len(self.noise_path)):
            self.noise_dataset.append(h5py.File(self.noise_path[i].strip(), "r"))

        self.rir_dataset = []
        for i in range(len(self.rir_path)):
            self.rir_dataset.append(h5py.File(self.rir_path[i].strip(), "r"))

        self.num_iter_per_epoch = num_iter_per_epoch
        self.session_length = session_length

        # for simulator
        self.input_sr = input_sr
        self.target_sr = target_sr
        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list
        self.reverb_proportion = reverb_proportion
        # self.noise_proportion = conf.noise_proportion
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        # if self.target_sr != self.input_sr:
        #     self.resample = Resample(self.input_sr, self.target_sr)

    def __getitem__(self, index):
        speech, noise, rir = self.sampling()

        snr = self._random_select_from(self.snr_list)
        use_reverb = bool(np.random.random(1) < self.reverb_proportion)
        # use_noise = bool(np.random.random(1) < self.noise_proportion)
        # noise if use_noise else noise*0,

        mixture, target = self.simulator(
            clean_y=speech,
            noise_y=noise,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            rir=rir if use_reverb else None,
        )

        key = index
        wav_len = mixture.shape[-1]
        if self.target_sr != self.input_sr:
            mixture = librosa.resample(
                mixture, orig_sr=self.input_sr, target_sr=self.target_sr
            )
            target = librosa.resample(
                target, orig_sr=self.input_sr, target_sr=self.target_sr
            )

        wav_sr = self.target_sr
        # mixture = torch.from_numpy(mixture).unsqueeze(0).float(),
        # target_reverb = torch.from_numpy(target).unsqueeze(0).float()
        # target_direct = target_reverb.clone()
        # sf.write('mixture.wav', mixture, 16000)
        # sf.write('target.wav', target, 16000)

        return (
            torch.from_numpy(mixture).unsqueeze(0).float(),
            torch.from_numpy(target).unsqueeze(0).float(),
        )

    def __len__(self):
        return self.num_iter_per_epoch

    def sampling(
        self,
    ):
        # sample a speech segment
        speech_idx = torch.randint(len(self.speech_path), (1,))
        speech_seg_idx = torch.randint(
            self.speech_dataset[speech_idx]["data"].shape[0], (1,)
        )
        speech = self.speech_dataset[speech_idx]["data"][speech_seg_idx].astype(float)

        s_idx = torch.randint(
            speech.shape[-1] - self.input_sr * self.session_length, (1,)
        )[0]
        speech = speech[s_idx : s_idx + self.input_sr * self.session_length]

        # sample a noise segment
        noise_idx = torch.randint(len(self.noise_path), (1,))
        noise_seg_idx = torch.randint(
            self.noise_dataset[noise_idx]["data"].shape[0], (1,)
        )
        noise = self.noise_dataset[noise_idx]["data"][noise_seg_idx].astype(float)
        s_idx = torch.randint(
            noise.shape[-1] - self.input_sr * self.session_length, (1,)
        )[0]
        noise = noise[s_idx : s_idx + self.input_sr * self.session_length]

        # sample a rir segment
        rir_idx = torch.randint(len(self.rir_path), (1,))
        rir_seg_idx = torch.randint(self.rir_dataset[rir_idx]["idx"].shape[0], (1,))
        s, e = self.rir_dataset[rir_idx]["idx"][rir_seg_idx].astype(int)
        rir = self.rir_dataset[rir_idx]["data"][s:e]

        return speech, noise, rir

    @staticmethod
    def simulator(
        clean_y,
        noise_y,
        snr,
        target_dB_FS,
        target_dB_FS_floating_value,
        rir=None,
        eps=1e-6,
    ):
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            # Is there a alignment issue?
            clean_y = signal.fftconvolve(clean_y, rir)[: len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y**2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y**2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value,
        )

        # Use the same RMS value of dBFS for noisy speech
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # The mixed speech is clipped if the RMS value of noisy speech is too large.
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    @staticmethod
    def _parse_snr_range(snr_range):
        assert (
            len(snr_range) == 2
        ), f"The range of SNR should be [low, high], not {snr_range}."
        assert (
            snr_range[0] <= snr_range[-1]
        ), f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list
