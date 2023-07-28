import logging
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import torch
from pesq import pesq as pesq_backend
from pystoi import stoi as stoi_backend
from torch import Tensor

from audiozen.utils import check_same_shape

logger = logging.getLogger(__name__)


def preprocessing(est, ref):
    if est.ndim != 1 or ref.ndim != 1:
        est = est.reshape(-1)
        ref = ref.reshape(-1)

    check_same_shape(est, ref)

    if torch.is_tensor(est) or torch.is_tensor(ref):
        est = est.detach().cpu().numpy()
        ref = ref.detach().cpu().numpy()

    return est, ref


class STOI:
    def __init__(self, sr=16000) -> None:
        self.sr = sr

    def __call__(self, est, ref, extended=False):
        est, ref = preprocessing(est, ref)
        stoi_val = stoi_backend(ref, est, self.sr, extended=extended)
        return {"stoi": stoi_val}


class PESQ:
    def __init__(self, sr=16000, mode="wb") -> None:
        self.sr = sr
        self.mode = mode

        if self.sr not in (8000, 16000):
            logging.warning(
                f"Unsupported sample rate: {sr}. Expected 8000 or 16000. Resampling will be applied later."
            )

        if self.mode not in ("wb", "nb"):
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'wb' or 'nb'.")

    def __call__(self, est, ref):
        est, ref = preprocessing(est, ref)
        if self.sr not in (8000, 16000):
            ref = librosa.resample(ref, orig_sr=self.sr, target_sr=16000)
            est = librosa.resample(est, orig_sr=self.sr, target_sr=16000)
            self.sr = 16000

        pesq_val = pesq_backend(self.sr, ref, est, self.mode)

        return {f"pesq_{self.mode}": pesq_val}


class SISDR:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, est: Tensor, ref: Tensor, zero_mean: bool = False) -> dict:
        """Scale-invariant signal-to-distortion ratio (SI-SDR).

        Args:
            est: float tensor with the shape of (B, T)
            ref: float tensor with shape of (B,T)
            zero_mean: If to zero mean ref and est or not

        Returns:
            Float tensor with shape of SDR values per sample

        Examples:
            >>> ref = torch.tensor([3.0, -0.5, 2.0, 7.0])
            >>> est = torch.tensor([2.5, 0.0, 2.0, 8.0])
            >>> si_sdr(est, ref)
            >>> tensor(18.4030)
        """
        est = est.reshape(-1)
        ref = ref.reshape(-1)

        check_same_shape(est, ref)

        if not torch.is_tensor(est) or not torch.is_tensor(ref):
            est = torch.tensor(est)
            ref = torch.tensor(ref)

        eps = torch.finfo(est.dtype).eps

        if zero_mean:
            ref = ref - torch.mean(ref, dim=-1, keepdim=True)
            est = est - torch.mean(est, dim=-1, keepdim=True)

        alpha = (torch.sum(est * ref, dim=-1, keepdim=True) + eps) / (
            torch.sum(ref**2, dim=-1, keepdim=True) + eps
        )
        ref_scaled = alpha * ref

        noise = ref_scaled - est

        val = (torch.sum(ref_scaled**2, dim=-1) + eps) / (
            torch.sum(noise**2, dim=-1) + eps
        )
        val = 10 * torch.log10(val)

        val = val.item()

        return {"si_sdr": val}


class pDNSMOS:
    def __init__(self, input_sr=16000) -> None:
        super().__init__()

        root_dir = Path(__file__).parent.absolute()

        self.p835_personal_sess = ort.InferenceSession(
            root_dir / "external" / "pDNSMOS" / "sig_bak_ovr.onnx",
            providers=["CPUExecutionProvider"],
        )

        self.input_sr = input_sr

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40  # type: ignore
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS=False):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio):
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()

        SAMPLERATE = 16000
        INPUT_LENGTH = 9.01

        if self.input_sr != 16000:
            audio = librosa.resample(audio, orig_sr=self.input_sr, target_sr=SAMPLERATE)

        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * SAMPLERATE)

        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLERATE) - INPUT_LENGTH) + 1

        hop_len_samples = SAMPLERATE
        predicted_p_mos_sig_seg_raw = []
        predicted_p_mos_bak_seg_raw = []
        predicted_p_mos_ovr_seg_raw = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p_mos_sig_raw, p_mos_bak_raw, p_mos_ovr_raw = self.p835_personal_sess.run(
                None, oi
            )[0][0]

            predicted_p_mos_ovr_seg_raw.append(p_mos_ovr_raw)
            predicted_p_mos_sig_seg_raw.append(p_mos_sig_raw)
            predicted_p_mos_bak_seg_raw.append(p_mos_bak_raw)

        clip_dict = {}
        # clip_dict["sr"] = SAMPLERATE
        # clip_dict["len"] = actual_audio_len / SAMPLERATE
        clip_dict["pSIG"] = np.mean(predicted_p_mos_sig_seg_raw)
        clip_dict["pBAK"] = np.mean(predicted_p_mos_bak_seg_raw)
        clip_dict["pOVRL"] = np.mean(predicted_p_mos_ovr_seg_raw)

        return clip_dict


class DNSMOS:
    def __init__(self, input_sr=16000, device=-1) -> None:
        super().__init__()

        root_dir = Path(__file__).parent.absolute()

        if device > -1:
            print(f"Using device: {device}")
            providers = [("CUDAExecutionProvider", {"device_id": device})]
        else:
            providers = ["CPUExecutionProvider"]

        self.p835_sess = ort.InferenceSession(
            (root_dir / "external" / "DNSMOS" / "sig_bak_ovr.onnx").as_posix(),
            providers=providers,
        )

        self.p808_sess = ort.InferenceSession(
            (root_dir / "external" / "DNSMOS" / "model_v8.onnx").as_posix(),
            providers=providers,
        )

        self.input_sr = input_sr

    def audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40  # type: ignore
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS=False):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, return_p808=True):
        if audio.ndim != 1:
            audio = audio.reshape(-1)

        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()

        SAMPLERATE = 16000
        INPUT_LENGTH = 9.01

        if self.input_sr != 16000:
            audio = librosa.resample(audio, orig_sr=self.input_sr, target_sr=SAMPLERATE)

        len_samples = int(INPUT_LENGTH * SAMPLERATE)

        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLERATE) - INPUT_LENGTH) + 1

        hop_len_samples = SAMPLERATE
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            oi = {"input_1": input_features}

            if return_p808:
                p808_input_features = np.array(
                    self.audio_melspec(audio=audio_seg[:-160])
                ).astype("float32")[np.newaxis, :, :]
                p808_oi = {"input_1": p808_input_features}
                p808_mos = self.p808_sess.run(None, p808_oi)[0][0][0]
                predicted_p808_mos.append(p808_mos)

            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.p835_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw
            )

            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)

        clip_dict = {}

        if return_p808:
            clip_dict["P808"] = np.mean(predicted_p808_mos)

        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)

        return clip_dict


def compute_synops(fb_all_layer_outputs, sb_all_layer_outputs):
    synops = 0.0
    for i in range(1, len(fb_all_layer_outputs) - 1):
        synops += (
            torch.gt(fb_all_layer_outputs[i], 0).float().mean()
            * fb_all_layer_outputs[i].size(-1)
            * (fb_all_layer_outputs[i + 1].size(-1) + fb_all_layer_outputs[i].size(-1))
        )
    for i in range(len(sb_all_layer_outputs)):
        for j in range(1, len(sb_all_layer_outputs[i]) - 1):
            # print(sb_all_layer_outputs[i][j].size())
            synops += (
                torch.gt(sb_all_layer_outputs[i][j], 0).float().mean()
                * sb_all_layer_outputs[i][j].size(-1)
                * (
                    sb_all_layer_outputs[i][j + 1].size(-1)
                    + sb_all_layer_outputs[i][j].size(-1)
                )
            )
    return synops.item()


def compute_neuronops(fb_all_layer_outputs, sb_all_layer_outputs):
    neuronops = 0.0
    for i in range(len(fb_all_layer_outputs)):
        neuronops += fb_all_layer_outputs[i].size(-1)
    for i in range(len(sb_all_layer_outputs)):
        for j in range(len(sb_all_layer_outputs[i])):
            neuronops += sb_all_layer_outputs[i][j].size(-1)
    return neuronops
