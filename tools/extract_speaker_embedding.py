from pathlib import Path

import librosa
import numpy as np
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


def main(src_dir, dest_dir, gpu_id):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": f"cuda:{gpu_id}"},
    )

    src_dir = Path(src_dir).expanduser().absolute()
    dest_dir = Path(dest_dir).expanduser().absolute()

    path_list = librosa.util.find_files(src_dir.as_posix(), ext=["wav"])
    print(f"Found {len(path_list)} files in {src_dir.as_posix()}.")

    for path in tqdm(path_list):
        signal, fs = torchaudio.load(path)
        assert fs == 16000, f"Sampling rate of {path} is not 16kHz."
        embeddings = classifier.encode_batch(signal)

        dest_fpath = dest_dir / Path(path).relative_to(src_dir)
        dest_fpath.parent.mkdir(parents=True, exist_ok=True)
        dest_fpath = dest_fpath.with_suffix(".npy")

        np.save(dest_fpath, embeddings.squeeze().cpu().numpy())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", type=str, required=True)
    parser.add_argument("-o", "--dest_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()

    main(args.src_dir, args.dest_dir, args.gpu_id)
