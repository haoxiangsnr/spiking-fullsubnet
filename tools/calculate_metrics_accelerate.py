from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator, PartialState
from regex import P
from tqdm import tqdm

from audiozen.metric import DNSMOS, SISDR, IntelSISNR


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, enh_wav_dir, clean_wav_dir):
        super(TestDataset, self).__init__()
        enh_wav_fpath_list = librosa.util.find_files(enh_wav_dir)
        clean_wav_fpath_list = librosa.util.find_files(clean_wav_dir)
        assert len(enh_wav_fpath_list) == len(
            clean_wav_fpath_list
        ), "The number of files in the two directories must be the same."
        print(f"Found {len(enh_wav_fpath_list)} files in {enh_wav_dir}.")

        self.enh_wav_fpath_list = enh_wav_fpath_list
        self.clean_wav_fpath_list = clean_wav_fpath_list
        self.enh_wav_dir = Path(enh_wav_dir)
        self.clean_wav_dir = Path(clean_wav_dir)

    def __len__(self):
        return len(self.enh_wav_fpath_list)

    def __getitem__(self, index):
        enh_wav_fpath = self.enh_wav_fpath_list[index]
        enh_stem = Path(enh_wav_fpath).stem
        clean_stem = "clean_" + "_".join(enh_stem.split("_")[-2:])

        clean_wav_fpath = self.clean_wav_dir / f"{clean_stem}.wav"

        enh_wav, _ = librosa.load(enh_wav_fpath, sr=16000)
        clean_wav, _ = librosa.load(clean_wav_fpath, sr=16000)
        num_samples = 30 * 16000

        if len(enh_wav) > num_samples:
            enh_wav = enh_wav[:num_samples]
        else:
            enh_wav = np.concatenate([enh_wav, np.zeros(num_samples - len(enh_wav))])
        if len(clean_wav) > num_samples:
            clean_wav = clean_wav[:num_samples]
        else:
            clean_wav = np.concatenate(
                [clean_wav, np.zeros(num_samples - len(clean_wav))]
            )

        return enh_wav, clean_wav


def calculate(enh_wav_dir, clean_wav_dir, batch_size=10, num_workers=0):
    accelerator = Accelerator()
    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    test_dataset = TestDataset(enh_wav_dir=enh_wav_dir, clean_wav_dir=clean_wav_dir)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    # Prepare the metrics
    intel_sisnr = IntelSISNR()
    dnsmos = DNSMOS(input_sr=16000, device=accelerator.process_index)

    intel_si_snr_output = []
    dnsmos_ovrl_output = []
    dnsmos_sig_output = []
    dnsmos_bak_output = []

    for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        enh_wav, clean_wav = batch

        intel_si_snr_list = []
        dnsmos_ovrl_score_list = []
        dnsmos_sig_score_list = []
        dnsmos_bak_score_list = []
        # Calculate the metrics
        for i in range(enh_wav.shape[0]):
            intel_si_snr = intel_sisnr(enh_wav[i], clean_wav[i])
            dnsmos_score = dnsmos(enh_wav[i])

            intel_si_snr_list.append(intel_si_snr["intel_si_snr"])
            dnsmos_ovrl_score_list.append(dnsmos_score["OVRL"])
            dnsmos_sig_score_list.append(dnsmos_score["SIG"])
            dnsmos_bak_score_list.append(dnsmos_score["BAK"])

        # to Tensor
        intel_si_snr_list = torch.from_numpy(np.array(intel_si_snr_list))
        dnsmos_ovrl_score_list = torch.from_numpy(np.array(dnsmos_ovrl_score_list))
        dnsmos_sig_score_list = torch.from_numpy(np.array(dnsmos_sig_score_list))
        dnsmos_bak_score_list = torch.from_numpy(np.array(dnsmos_bak_score_list))

        # Gather the results
        # [[64, 1], [64, 1], [64, 1], [64, 1], ...]
        intel_si_snr_list = intel_si_snr_list.to(accelerator.device)
        dnsmos_ovrl_score_list = dnsmos_ovrl_score_list.to(accelerator.device)
        dnsmos_sig_score_list = dnsmos_sig_score_list.to(accelerator.device)
        dnsmos_bak_score_list = dnsmos_bak_score_list.to(accelerator.device)

        intel_si_snr_list = accelerator.gather_for_metrics(intel_si_snr_list)
        dnsmos_ovrl_score_list = accelerator.gather_for_metrics(dnsmos_ovrl_score_list)
        dnsmos_sig_score_list = accelerator.gather_for_metrics(dnsmos_sig_score_list)
        dnsmos_bak_score_list = accelerator.gather_for_metrics(dnsmos_bak_score_list)

        intel_si_snr_output.append(intel_si_snr_list)
        dnsmos_ovrl_output.append(dnsmos_ovrl_score_list)
        dnsmos_sig_output.append(dnsmos_sig_score_list)
        dnsmos_bak_output.append(dnsmos_bak_score_list)

    mean_intel_si_snr = torch.mean(torch.cat(intel_si_snr_output))
    mean_dnsmos_ovrl = torch.mean(torch.cat(dnsmos_ovrl_output))
    mean_dnsmos_sig = torch.mean(torch.cat(dnsmos_sig_output))
    mean_dnsmos_bak = torch.mean(torch.cat(dnsmos_bak_output))

    if accelerator.is_local_main_process:
        print(f"Calculated metrics for {enh_wav_dir} ...")
        print(
            f"mean_intel_si_snr: {mean_intel_si_snr}, mean_dnsmos_ovrl: {mean_dnsmos_ovrl}, mean_dnsmos_sig: {mean_dnsmos_sig}, mean_dnsmos_bak: {mean_dnsmos_bak}"
        )


if __name__ == "__main__":
    enh_wav_dir = Path(
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_m/enhanced/best"
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_m/enhanced/latest"
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_l/enhanced/best"
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_l/enhanced/latest"
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_xl/enhanced/best"
        # "/datasets/datasets_fullband/testset_1_enh_20230917/baseline_xl/enhanced/latest"
        "/datasets/datasets_fullband/IntelNeuromorphicDNSChallenge/data/MicrosoftDNS_4_ICASSP/test_set_1/noisy"
    )
    clean_wav_dir = Path(
        "/datasets/datasets_fullband/IntelNeuromorphicDNSChallenge/data/MicrosoftDNS_4_ICASSP/test_set_1/clean"
    )

    calculate(enh_wav_dir, clean_wav_dir)
