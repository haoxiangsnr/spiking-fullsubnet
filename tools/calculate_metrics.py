import argparse
import os
from pathlib import Path

import librosa
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from audiozen.metric import DNSMOS, PESQ, SISDR, STOI

"""
Steps:
    1. Check alignment of reference and estimated wav files
    2. Compute metrics
"""


class MetricComputer:
    def __init__(self, sr=16000) -> None:
        self.dns_mos = DNSMOS(input_sr=sr)
        self.stoi = STOI(sr=sr)
        self.pesq_wb = PESQ(sr=sr, mode="wb")
        self.pesq_nb = PESQ(sr=sr, mode="nb")
        self.si_sdr = SISDR()
        self.sr = sr

    def compute_per_item(self, ref_wav_path, est_wav_path):
        ref, _ = librosa.load(ref_wav_path, sr=self.sr)
        est, _ = librosa.load(est_wav_path, sr=self.sr)

        basename = get_basename(ref_wav_path)

        if len(ref) != len(est):
            raise ValueError(
                f"{ref_wav_path} and {est_wav_path} are not in the same length."
            )

        pesq_wb = self.pesq_wb(est, ref)
        pesq_nb = self.pesq_nb(est, ref)
        stoi = self.stoi(est, ref)
        si_sdr = self.si_sdr(est, ref)
        dns_mos = self.dns_mos(est)

        return {"name": basename} | pesq_wb | pesq_nb | stoi | si_sdr  # | dns_mos

    def compute(self, reference_wav_paths, estimated_wav_paths, n_jobs=10):
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.compute_per_item)(ref, est)
            for ref, est in tqdm(zip(reference_wav_paths, estimated_wav_paths))
        )

        df_metrics = pd.DataFrame(rows)

        df_metrics_mean = df_metrics.mean(numeric_only=True)
        df_metrics_mean_df = df_metrics_mean.to_frame().T

        print(df_metrics_mean_df.to_markdown())


def load_wav_paths_from_text_file(scp_path, to_abs=True):
    wav_paths = [
        line.rstrip("\n")
        for line in open(os.path.abspath(os.path.expanduser(scp_path)), "r")
    ]
    if to_abs:
        tmp = []
        for path in wav_paths:
            tmp.append(os.path.abspath(os.path.expanduser(path)))
        wav_paths = tmp
    return wav_paths


def shrink_multi_channel_path(full_dataset_list: list, num_channels: int) -> list:
    """
    Args:
        full_dataset_list: [
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic1.wav
            ...
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic2.wav
        ]
        num_channels:

    Returns:

    """
    assert len(full_dataset_list) % num_channels == 0, "Num error"

    shrunk_dataset_list = []
    for index in range(0, len(full_dataset_list), num_channels):
        full_path = full_dataset_list[index]
        shrunk_path = f"{'_'.join(full_path.split('_')[:-1])}.wav"
        shrunk_dataset_list.append(shrunk_path)

    assert len(shrunk_dataset_list) == len(full_dataset_list) // num_channels
    return shrunk_dataset_list


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def pre_processing(est_list, ref_list, align_mode=None):
    """Merge and validate filepath"""
    reference_wav_paths = []
    estimated_wav_paths = []

    for est in est_list:
        est = Path(est).expanduser().absolute()
        if est.is_dir():
            estimated_wav_paths += librosa.util.find_files(est.as_posix(), ext="wav")
        else:
            estimated_wav_paths += load_wav_paths_from_text_file(est.as_posix())

    for ref in ref_list:
        ref = Path(ref).expanduser().absolute()
        if ref.is_dir():
            reference_wav_paths += librosa.util.find_files(ref.as_posix(), ext="wav")
        else:
            reference_wav_paths += load_wav_paths_from_text_file(ref.as_posix())

    reference_wav_paths.sort(key=lambda x: os.path.basename(x))
    estimated_wav_paths.sort(key=lambda x: os.path.basename(x))

    print(f"#Ref: {len(reference_wav_paths)}, #Est: {len(estimated_wav_paths)}")
    print("Checking whether files in two lists have the same basename...")

    # ================== Keep two lists aligned ==================
    if align_mode is None:
        check_two_aligned_list(reference_wav_paths, estimated_wav_paths)
    else:
        # Reorder estimated_wav_paths according to reference_wav_paths for specific dataset
        reordered_estimated_wav_paths = []
        if align_mode == "dns_1":
            # Recoder estimated_wav_paths according to the suffix of reference_wav_paths
            # ref:
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    est_basename = get_basename(est_path)
                    if "clean_" + "_".join(
                        est_basename.split("_")[-2:]
                    ) == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif align_mode == "dns_1_custom":
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    # e.g., clnsp74_fan_out_56236_0_snr9_tl-28_fileid_210.wav-denoise.wav
                    # clean_fileid_207.wav
                    est_basename = get_basename(est_path)[:-12]  # remove ".wav-denoise"
                    expected_basename = "clean_" + "_".join(
                        est_basename.split("_")[-2:]
                    )
                    if expected_basename == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif align_mode == "intel_ndns":
            for ref_path in reference_wav_paths:
                ref_basename = get_basename(ref_path)
                ref_file_id = ref_basename.split("_")[-1]
                for est_path in estimated_wav_paths:
                    # zweiplaneten..._snr4_tl-33_fileid_49151.wav
                    est_basename = get_basename(est_path)
                    est_file_id = est_basename.split("_")[-1]

                    if ref_file_id == est_file_id:
                        reordered_estimated_wav_paths.append(est_path)
        elif align_mode == "dns_2":
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    # synthetic_french_acejour_orleans_sb_64kb-01_jbq2HJt9QXw_snr14_tl-26_fileid_47
                    # synthetic_clean_fileid_47
                    est_basename = get_basename(est_path)
                    file_id = est_basename.split("_")[-1]
                    if f"synthetic_clean_fileid_{file_id}" == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif align_mode == "maxhub_noisy":
            # Reference_channel = 0
            # 寻找对应的干净语音
            reference_channel = 0
            print(f"Found #files: {len(reference_wav_paths)}")
            for est_path in estimated_wav_paths:
                # MC0604W0154_room4_rev_RT600.1_mic1_micpos1.5p0.5p1.84_srcpos4.507p1.5945p1.3_langle180_angle20_ds3.2_kesou_kesou_mic1.wav
                est_basename = get_basename(est_path)  # 带噪的
                for ref_path in reference_wav_paths:
                    ref_basename = get_basename(ref_path)
        else:
            raise NotImplementedError(f"Not supported specific dataset {align_mode}.")

        estimated_wav_paths = reordered_estimated_wav_paths

    print(
        f"After checking, #Ref: {len(reference_wav_paths)}, #Est: {len(estimated_wav_paths)}"
    )

    return reference_wav_paths, estimated_wav_paths


def check_two_aligned_list(a, b):
    assert len(a) == len(b), f"两个列表中的长度不等. {len(a)} {len(b)}"
    for z, (i, j) in enumerate(zip(a, b), start=1):
        assert get_basename(i) == get_basename(j), (
            f"两个列表中存在不相同的文件名，行数为: {z}" f"\n\t {i}" f"\n\t{j}"
        )


def main(args):
    sr = args.sr
    align_mode = args.align_mode.lower()

    reference_wav_paths, estimated_wav_paths = pre_processing(
        args.estimated, args.reference, align_mode
    )

    metric_computer = MetricComputer(sr=sr)
    metric_computer.compute(reference_wav_paths, estimated_wav_paths, n_jobs=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate speech enhancement metrics for estimated speech and reference speech.",
        epilog="python calculate_metrics.py -E 'est_dir' -R 'ref_dir' -M SI_SDR,STOI,WB_PESQ,NB_PESQ,SSNR,LSD,SRMR",
    )
    parser.add_argument(
        "-R",
        "--reference",
        required=True,
        help="It can be a list of dir paths seprated using comma or a list of scp text paths.",
        type=lambda s: [i for i in s.split(",")],
    )
    parser.add_argument(
        "-E",
        "--estimated",
        required=True,
        help="",
        type=lambda s: [i for i in s.split(",")],
    )

    parser.add_argument("--sr", type=int, default=16000, help="Sample rate.")
    parser.add_argument(
        "-A",
        "--align_mode",
        type=str,
        default=None,
        help="Use corresponding pre-processing method for specific dataset. It can be 'dns_1', 'dns_2'",
    )
    args = parser.parse_args()
    main(args)
