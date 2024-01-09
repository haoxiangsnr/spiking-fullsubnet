from pathlib import Path

import soundfile as sf
from torch.utils.data import Dataset

from audiozen.acoustics.io import load_audio, subsample

dists = ["far", "near"]
rooms = ["room1", "room2", "room3"]


class EvaluationRealDataset(Dataset):
    def __init__(self, scp_fpath):
        super().__init__()

        with open(scp_fpath) as f:
            self.fpath_list = f.read().splitlines()

    def __len__(self):
        return len(self.fpath_list)

    def __getitem__(self, index):
        id_and_fpath = self.fpath_list[index]
        id, fpath = id_and_fpath.split()

        noisy, sr = sf.read(fpath, dtype="float32")
        return noisy, fpath


class EvaluationSimDataset(Dataset):
    def __init__(self, scp_fpath):
        super().__init__()

        with open(scp_fpath) as f:
            self.fpath_list = f.read().splitlines()

    def __len__(self):
        return len(self.fpath_list)

    def __getitem__(self, index):
        id_and_fpath = self.fpath_list[index]
        id, fpath = id_and_fpath.split()

        noisy, sr = sf.read(fpath, dtype="float32")
        return noisy, fpath


class SimTrainDataset(Dataset):
    def __init__(self, rvb_scp_fpath, dry_scp_fpath, duration_in_seconds=4.0, sr=16000, limit=None, offset=0):
        super().__init__()

        with open(rvb_scp_fpath) as f:
            self.rvb_fpath_list = f.read().splitlines()

        with open(dry_scp_fpath) as f:
            self.ref_fpath_list = f.read().splitlines()

        if len(self.rvb_fpath_list) != len(self.ref_fpath_list):
            raise ValueError(
                f"len(self.rvb_fpath_list) != len(self.ref_fpath_list): {len(self.rvb_fpath_list)} != {len(self.ref_fpath_list)}"
            )

        if offset > 0:
            self.rvb_fpath_list = self.rvb_fpath_list[offset:]
            self.ref_fpath_list = self.ref_fpath_list[offset:]

        if limit is not None:
            self.rvb_fpath_list = self.rvb_fpath_list[:limit]
            self.ref_fpath_list = self.ref_fpath_list[:limit]
        self.duration_in_seconds = duration_in_seconds

    def __len__(self):
        return len(self.rvb_fpath_list)

    def __getitem__(self, index):
        id_and_rvb_fpath = self.rvb_fpath_list[index]
        id_and_ref_fpath = self.ref_fpath_list[index]
        id, rvb_fpath = id_and_rvb_fpath.split(" ")
        _, ref_fpath = id_and_ref_fpath.split(" ")

        rvb_y, sr = sf.read(rvb_fpath, dtype="float32")
        ref_y, sr = sf.read(ref_fpath, dtype="float32")

        if rvb_y.shape != ref_y.shape:
            raise ValueError(f"rvb_y.shape != ref_y.shape: {rvb_y.shape} != {ref_y.shape}")

        rvb_y, start_idx = subsample(rvb_y, int(self.duration_in_seconds * sr), return_start_idx=True)
        ref_y = subsample(ref_y, int(self.duration_in_seconds * sr), start_idx=start_idx)

        return rvb_y, ref_y, id


class SimDTDataset(Dataset):
    def __init__(self, rvb_scp_fpath, dry_scp_fpath, sr=16000, limit=None, offset=0):
        super().__init__()

        with open(rvb_scp_fpath) as f:
            self.rvb_fpath_list = f.read().splitlines()

        with open(dry_scp_fpath) as f:
            self.ref_fpath_list = f.read().splitlines()

        if len(self.rvb_fpath_list) != len(self.ref_fpath_list) * 2:
            raise ValueError(
                f"len(self.rvb_fpath_list) != len(self.ref_fpath_list) * 2: {len(self.rvb_fpath_list)} != {len(self.ref_fpath_list)} * 2"
            )

        if offset > 0:
            self.rvb_fpath_list = self.rvb_fpath_list[offset:]
            self.ref_fpath_list = self.ref_fpath_list[offset:]

        if limit is not None:
            self.rvb_fpath_list = self.rvb_fpath_list[:limit]
            self.ref_fpath_list = self.ref_fpath_list[:limit]

        self.sr = sr

    def __len__(self):
        return len(self.rvb_fpath_list)

    def __getitem__(self, index):
        id_and_est_fpath = self.rvb_fpath_list[index]
        id, rvb_fpath = id_and_est_fpath.split()

        # replace "far_test" and "near_test" with "clean_test"
        ref_fpath = rvb_fpath.replace("far_test", "cln_test").replace("near_test", "cln_test")
        # remove "_ch1" from ref_fpath
        ref_fpath = ref_fpath.replace("_ch1", "")

        rvb_y, _ = load_audio(rvb_fpath, sr=self.sr)
        ref_y, _ = load_audio(ref_fpath, sr=self.sr)

        rvb_y = rvb_y[: ref_y.shape[0]]  # trim to same length. this is only necessary for the dt/et set

        return rvb_y, ref_y, id


if __name__ == "__main__":
    # real_dataset = EvaluationRealDataset("/home/xhao/proj/spiking-fullsubnet/recipes/reverb/data/et_real_1ch.scp")
    real_dataset = EvaluationSimDataset("/home/xhao/proj/spiking-fullsubnet/recipes/reverb/data/et_simu_1ch.scp")
    print(next(iter(real_dataset)))

    # real ref
    old_path = Path(
        "/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL/MC_WSJ_AV_Eval/audio/stat/T21/array2/5k/AMI_WSJ21-Array2-2_T21c0206.wav"
    )
    old_root = Path("/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL")
    new_root = Path("/nfs/xhao/data/reverb_challenge/kaldi/egs/reverb/s5/wav/spiking_fullsubnet")
    new_path = new_root / old_path.relative_to(old_root)
    print(new_path)

    # sim ref
    old_path = Path(
        "/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL/REVERB_WSJCAM0_et/data/far_test/primary_microphone/si_et_1/c30/c30c0201_ch1.wav"
    )
    old_root = Path("/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL")
    new_root = Path("/nfs/xhao/data/reverb_challenge/kaldi/egs/reverb/s5/wav/spiking_fullsubnet")
    new_path = new_root / old_path.relative_to(old_root)
    print(new_path)
