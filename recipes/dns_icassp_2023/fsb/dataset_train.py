import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from audiozen.acoustics.audio_feature import (
    is_clipped,
    loudness_max_norm,
    loudness_rms_norm,
    sxr2gain,
)
from audiozen.dataset.base_dataset import BaseDataset
from audiozen.utils import expand_path


class Dataset(BaseDataset):
    def __init__(
        self,
        sr,
        clean_dataset,
        real_rir_dataset,
        simulated_rir_dataset,
        point_noise_dataset,
        iso_noise_dataset,
        enrollment_dataset,
        pt_noise_snr_range,
        iso_noise_snr_range,
        sir_range,
        enroll_snr_range,
        loudness_lvl,
        loudness_floating_value,
        silence_length,
        sample_length,
        learning_target,
        len_early_rir,
        prob_use_real_rir,
        prob_add_itf_spks,
        max_num_itf_spks,
        prob_add_point_noise,
        max_num_pt_noises,
        enroll_sample_length,
        prob_add_iso_noise,
    ):
        """Dynamic generate mixing data for training"""
        super().__init__()
        # acoustics args
        self.sr = sr

        # data files
        self.clean_path_list = self._load_dataset_in_txt(clean_dataset)
        self.real_rir_path_list = self._load_dataset_in_txt(real_rir_dataset)
        self.simulated_rir_path_list = self._load_dataset_in_txt(simulated_rir_dataset)
        self.pt_noise_path_list = self._load_dataset_in_txt(point_noise_dataset)
        self.iso_noise_path_list = self._load_dataset_in_txt(iso_noise_dataset)
        self.enrollment_path_list = self._load_dataset_in_txt(enrollment_dataset)

        self.pt_noise_snr_list = self._parse_snr_range(pt_noise_snr_range)
        self.iso_noise_snr_list = self._parse_snr_range(iso_noise_snr_range)
        self.enroll_snr_list = self._parse_snr_range(enroll_snr_range)
        self.sir_list = self._parse_snr_range(sir_range)

        # Probabilities
        self.prob_use_real_rir = prob_use_real_rir
        self.prob_add_itf_spks = prob_add_itf_spks
        self.prob_add_point_noises = prob_add_point_noise
        self.prob_add_iso_noise = prob_add_iso_noise
        self.max_num_itf_spks = max_num_itf_spks
        self.max_num_pt_noises = max_num_pt_noises

        self.silence_length = silence_length
        self.enroll_sample_length = enroll_sample_length
        self.loudness_lvl = loudness_lvl
        self.loudness_floating_value = loudness_floating_value
        self.sample_length = sample_length
        self.learning_target = learning_target
        self.len_early_rir = len_early_rir

        if self.prob_add_itf_spks:
            if max_num_itf_spks <= 0:
                raise ValueError("The 'max_num_itf_spks' should be greater than 0.")

        if self.prob_add_point_noises:
            if max_num_pt_noises <= 0:
                raise ValueError("The 'max_num_point_noise' should be greater than 0.")

        self.length = len(self.clean_path_list)

    def __len__(self):
        return self.length

    @staticmethod
    def _load_wav(path, duration=None, sr=None):
        if isinstance(path, Path):
            path = path.as_posix()

        with sf.SoundFile(path) as sf_desc:
            orig_sr = sf_desc.samplerate

            if duration is not None:
                frame_orig_duration = sf_desc.frames
                frame_duration = int(duration * orig_sr)
                if frame_duration < frame_orig_duration:
                    # Randomly select a segment
                    offset = np.random.randint(frame_orig_duration - frame_duration)
                    sf_desc.seek(offset)
                    y = sf_desc.read(
                        frames=frame_duration, dtype=np.float32, always_2d=True
                    ).T
                else:
                    y = sf_desc.read(dtype=np.float32, always_2d=True).T  # [C, T]
                    y = np.pad(
                        y, ((0, 0), (0, frame_duration - frame_orig_duration)), "wrap"
                    )
            else:
                y = sf_desc.read(dtype=np.float32, always_2d=True).T

        if y.shape[0] == 1:
            y = y[0]

        if sr is not None:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        else:
            sr = orig_sr

        return y, sr

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    @staticmethod
    def find_peak_idx(rir):
        """Find the peak index of the RIR.

        Args:
            rir: room impulse response with the shape of [T] or [C, T], where C is the number of channels.

        Returns:
            peak index
        """
        return np.argmax(np.abs(rir))

    @staticmethod
    def random_select_channel(rir):
        """Randomly select a channel of the RIR.

        Args:
            rir: room impulse response with the shape of [C, T], where C is the number of channels.

        Returns:
            selected channel of the RIR
        """
        return rir[np.random.randint(0, rir.shape[0])]

    def conv_rir(self, clean_y, rir, learning_target="full"):
        """Convolve clean_y with a RIR.

        Args:
            clean_y: clean signal.
            rir: room impulse response with the shape of [T] or [C, T], where C is the number of channels.
            learning_target: Which can be 'direct_path', 'early', or 'full'.

        Returns:
            convolved signal
        """
        assert rir.ndim in [1, 2], "The dimension of the RIR should be 1 or 2."
        assert learning_target in [
            "direct_path",
            "early",
            "full",
        ], "The learning target should be 'direct_path' or 'early', or 'full'."

        if rir.ndim == 2:
            rir = self.random_select_channel(rir)

        # Find the RIR of the learning target
        dp_idx = self.find_peak_idx(rir)

        if learning_target == "direct_path":
            rir = rir[:dp_idx]
        elif learning_target == "early":
            assert self.len_early_rir is not None, "The 'len_early_rir' should be set."
            len_early_rir = int(self.sr * self.len_early_rir)
            rir = rir[: dp_idx + len_early_rir]
        elif learning_target == "full":
            rir = rir
        else:
            raise ValueError(
                "The learning target should be 'direct_path' or 'early', or 'full'."
            )

        # audio with full-length RIR
        clean_y_rvb = signal.fftconvolve(clean_y, rir)
        clean_y_rvb = clean_y_rvb[: len(clean_y)]

        return clean_y_rvb

    def _mix_all_sources(self, sources, sxr_list):
        """Mix all sources with the given SNRs or SIRs."""
        assert len(sources) - len(sxr_list) == 1

        tgt_y = sources[0]
        mixture = [tgt_y]

        if len(sxr_list) > 0:
            for y, sxr in zip(sources[1:], sxr_list):
                gain = sxr2gain(tgt_y, y, sxr)
                mixture.append(y * gain)

        mixture = np.sum(mixture, axis=0)

        return mixture

    def _norm_all_sources(self, sources):
        out_sources = []
        for y in sources:
            y, _ = loudness_max_norm(y)
            y, _ = loudness_rms_norm(y, lvl=self.loudness_lvl)
            out_sources.append(y)

        return out_sources

    def _get_spk_id(self, fpath: Path):
        if not isinstance(fpath, Path):
            fpath = Path(fpath)

        return "_".join(fpath.stem.split("_")[:-1])

    def _find_enrollment(self, s0_path):
        # Find the path of enrollment speech
        path = Path(str(s0_path).replace("seg/clean", "enrollment_wav"))
        new_stem = "_".join(path.stem.split("_")[:-1])
        filename = f"enrol_{new_stem}.wav"  # e.g., enrol_complete_french_mix_sous_les_mers_1_01_seg1
        enroll_path = Path(path).parent / filename

        return enroll_path

    def _simulate_mixture(
        self,
        target_path,
        add_itf_spks,
        itf_path_list,
        max_num_itf_spks,
        sir_list,
        add_pt_noises,
        pt_noise_path_list,
        max_num_pt_noises,
        pt_noise_snr_list,
        add_iso_noise,
        iso_noise_path_list,
        iso_noise_snr_list,
        use_real_rir,
        target_rir_type,
    ):
        collected_point_sources = []
        collected_sxr_list = []
        collected_all_sources = []

        # Load point_source target speech
        target_y, _ = self._load_wav(target_path, self.sample_length, self.sr)
        collected_point_sources.append(target_y)

        # Load point_source interfering speech
        if add_itf_spks:
            collected_spk_ids = [self._get_spk_id(target_path)]
            num_itf_spks = random.randint(1, max_num_itf_spks)
            for _ in range(num_itf_spks):
                path = random.choice(itf_path_list)
                spk_id = self._get_spk_id(path)
                while spk_id in collected_spk_ids:  # Avoid using the same speaker
                    path = random.choice(self.clean_path_list)
                    spk_id = self._get_spk_id(path)
                y, _ = self._load_wav(path, self.sample_length, self.sr)
                collected_spk_ids.append(spk_id)

                collected_point_sources.append(y)
                collected_sxr_list.append(random.choice(sir_list))

        # Load point-source noises
        if add_pt_noises:
            num_pt_noises = random.randint(1, max_num_pt_noises)
            for _ in range(num_pt_noises):
                path = random.choice(pt_noise_path_list)
                y, _ = self._load_wav(path, self.sample_length, self.sr)
                collected_point_sources.append(y)
                collected_sxr_list.append(random.choice(pt_noise_snr_list))

        # Add RIRs to point sources
        num_pt_sources = len(collected_point_sources)
        if use_real_rir:
            # Randomly select RIRs from real RIR dataset
            rir_path_list = random.choices(self.real_rir_path_list, k=num_pt_sources)
            rir_list = [
                self._load_wav(rir_path, sr=self.sr)[0] for rir_path in rir_path_list
            ]
        else:
            # Randomly select a multichannel RIR from simulated RIR dataset
            # Simulated RIRs will keep the sources within a same room
            rir_path = random.choice(self.simulated_rir_path_list)
            mc_rir, _ = self._load_wav(rir_path, sr=self.sr)
            rir_list = mc_rir[:num_pt_sources, :]

        for idx, (y, rir) in enumerate(zip(collected_point_sources, rir_list)):
            if idx == 0:
                target_y_rvb = self.conv_rir(y, rir, target_rir_type)
            y_rvb = self.conv_rir(y, rir, "full")
            collected_all_sources.append(y_rvb)

        # Load isotropic noise
        if add_iso_noise:
            path = random.choice(iso_noise_path_list)
            y, _ = self._load_wav(path, self.sample_length, sr=self.sr)
            collected_sxr_list.append(random.choice(iso_noise_snr_list))
            collected_all_sources.append(y)

        # mixing sources according to the collected_sxr_list
        target_y_rvb, _ = loudness_max_norm(target_y_rvb)
        target_y_rvb, _ = loudness_rms_norm(target_y_rvb, lvl=self.loudness_lvl)

        collected_all_sources = self._norm_all_sources(collected_all_sources)
        mixture = self._mix_all_sources(collected_all_sources, collected_sxr_list)

        mixture_lvl = random.randint(
            self.loudness_lvl - self.loudness_floating_value,
            self.loudness_lvl + self.loudness_floating_value,
        )
        mixture, _ = loudness_rms_norm(mixture, lvl=mixture_lvl)

        if is_clipped(mixture):
            mixture, _ = loudness_max_norm(mixture)

        # Do we need to normalize the target speech?
        return mixture, target_y_rvb

    def __getitem__(self, item):
        # Target speech simulation
        target_path = self.clean_path_list[item]
        add_itf_spks = bool(np.random.random(1) <= self.prob_add_itf_spks)
        add_pt_noises = bool(np.random.random(1) <= self.prob_add_point_noises)
        add_iso_noise = bool(np.random.random(1) <= self.prob_add_iso_noise)
        use_real_rir = bool(np.random.random(1) <= self.prob_use_real_rir)
        mixture_y, target_y = self._simulate_mixture(
            target_path=target_path,
            add_itf_spks=add_itf_spks,
            itf_path_list=self.clean_path_list,
            max_num_itf_spks=self.max_num_itf_spks,
            sir_list=self.sir_list,
            add_pt_noises=add_pt_noises,
            pt_noise_path_list=self.pt_noise_path_list,
            max_num_pt_noises=self.max_num_pt_noises,
            pt_noise_snr_list=self.pt_noise_snr_list,
            add_iso_noise=add_iso_noise,
            iso_noise_path_list=self.iso_noise_path_list,
            iso_noise_snr_list=self.iso_noise_snr_list,
            use_real_rir=use_real_rir,
            target_rir_type=self.learning_target,
        )

        # Enrollment speech
        # enroll_path = self._find_enrollment(target_path)
        # add_pt_noises = bool(np.random.random(1) <= self.prob_add_point_noises)
        # add_iso_noise = bool(np.random.random(1) <= self.prob_add_iso_noise)
        # enroll_mixture_y, enroll_target_y = self._simulate_mixture(
        #     target_path=enroll_path,
        #     add_itf_spks=False,
        #     itf_path_list=self.clean_path_list,
        #     max_num_itf_spks=0,
        #     sir_list=None,
        #     add_pt_noises=False,
        #     pt_noise_path_list=self.pt_noise_path_list,
        #     max_num_pt_noises=self.max_num_pt_noises,
        #     pt_noise_snr_list=self.pt_noise_snr_list,
        #     add_iso_noise=False,
        #     iso_noise_path_list=self.iso_noise_path_list,
        #     iso_noise_snr_list=self.iso_noise_snr_list,
        #     use_real_rir=True,
        #     target_rir_type="full",
        # )

        return mixture_y, target_y  # , enroll_mixture_y


if __name__ == "__main__":
    dataset = Dataset(
        sr=16000,
        clean_dataset="/data/ssp/public/data/dns/dns4/track2/subset/all_train.txt",
        point_noise_dataset="/data/ssp/public/data/dns/dns4/noise_rir/noise.txt",
        simulated_rir_dataset="/data/ssp/public/data/dns/dns4/noise_rir/rir.txt",
        real_rir_dataset="/data/ssp/public/data/dns/dns4/noise_rir/rir.txt",
        iso_noise_dataset="/data/ssp/public/data/dns/dns4/noise_rir/noise.txt",
        enrollment_dataset="/data/ssp/public/data/dns/dns4/track2/enrollment_wav/enrollment_wav.txt",
        pt_noise_snr_range=[-5, 25],
        iso_noise_snr_range=[-5, 25],
        sir_range=[0, 25],
        enroll_snr_range=[0, 25],
        loudness_lvl=-26,
        loudness_floating_value=10,
        prob_use_real_rir=1.0,  # use real rir for all data
        prob_add_itf_spks=0.6,
        max_num_itf_spks=2,
        prob_add_point_noise=0.5,
        max_num_pt_noises=2,
        prob_add_iso_noise=0.5,
        enroll_sample_length=30,
        silence_length=0.2,
        sample_length=4,
        learning_target="early",
        len_early_rir=0.1,
    )

    data = next(iter(dataset))
    print(data)
