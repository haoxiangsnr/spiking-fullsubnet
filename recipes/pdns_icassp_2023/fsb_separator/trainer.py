import gc
import logging
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from audiozen.acoustics.audio_feature import save_wav, tune_dB_FS
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr)
        self.stoi = STOI(sr=self.sr)
        self.pesq = PESQ(sr=self.sr)
        self.si_sdr = SISDR()

    def training_step(self, batch, batch_idx):
        noisy, clean, embedding = batch
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        embedding = embedding.to(self.device)

        enhanced_real, enhanced_imag = self.model(noisy, embedding)
        enhanced = self.torch_istft(
            [enhanced_real, enhanced_imag],
            length=noisy.size(-1),
            input_type="real_imag",
        )

        loss = self.loss_function(enhanced, clean)

        return loss

    def training_epoch_end(self, training_epoch_output):
        loss_mean = torch.mean(torch.tensor(training_epoch_output))

        if self.rank == 0:
            logger.info(f"Training loss on epoch {self.current_epoch}: {loss_mean}")
            self.writer.add_scalar(f"Loss/Train", loss_mean, self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, fpath, embedding = batch
        noisy = noisy.to(self.device)
        embedding = embedding.to(self.device)

        enhanced_real, enhanced_imag = self.model(noisy, embedding)
        enhanced = self.torch_istft(
            [enhanced_real, enhanced_imag],
            length=noisy.size(-1),
            input_type="real_imag",
        )

        enhanced, *_ = tune_dB_FS(enhanced, target_dB_FS=-26)

        return enhanced, fpath

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=10, prefer="threads")(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            logger.info(f"\n {df_metrics.describe()}")

            df_metrics_mean = df_metrics.mean(numeric_only=True)

            score += df_metrics_mean["pOVRL"] + df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(
                    f"metrics_{dataloader_idx}/{metric}", value, self.current_epoch
                )

        gc.collect()
        torch.cuda.empty_cache()

        return score

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, fpath, embedding = batch
        noisy = noisy.to(self.device)
        embedding = embedding.to(self.device)

        enhanced_real, enhanced_imag = self.model(noisy, embedding)

        enhanced = self.torch_istft(
            [enhanced_real, enhanced_imag],
            length=noisy.size(-1),
            input_type="real_imag",
        )

        enhanced, *_ = tune_dB_FS(enhanced, target_dB_FS=-26)

        # save enhanced audio
        stem = Path(fpath[0]).stem
        enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
        enhanced_dir.mkdir(exist_ok=True, parents=True)
        enhanced_fpath = enhanced_dir / f"{stem}.wav"
        save_wav(enhanced, enhanced_fpath.as_posix(), self.sr)

        return enhanced, fpath

    def test_epoch_end(self, outputs):
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=10, prefer="threads")(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            logger.info(f"\n {df_metrics.describe()}")

    def compute_validation_metrics(self, dataloader_idx, step_out):
        enhanced, fpath = step_out
        dns_mos_dict = self.dns_mos(enhanced)
        dns_mos_dict["fpath"] = fpath
        return dns_mos_dict

    def compute_test_metrics(self, dataloader_idx, step_out):
        enhanced, fpath = step_out
        dns_mos_dict = self.dns_mos(enhanced)
        dns_mos_dict["fpath"] = fpath
        return dns_mos_dict