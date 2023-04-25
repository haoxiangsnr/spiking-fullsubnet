import shutil
from importlib.resources import path
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm


def main(src_dir, dest_dir):
    path_list = librosa.util.find_files(src_dir, ext=["wav"])
    for path in tqdm(path_list):
        path = Path(path)

        dest_path = dest_dir / path.relative_to(src_dir)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        rt60 = float(path.stem.split("_")[-1])
        if rt60 <= 0.8:
            shutil.copy(path, dest_path)


if __name__ == "__main__":
    src_dir = "/data/ssp/public/data/dns/dns4_16k/noise_rir/RIR_simulated"
    dest_dir = "/data/ssp/public/data/dns/dns4_16k/noise_rir/RIR_simulated_low_rt60"
    main(src_dir, dest_dir)
