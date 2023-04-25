import logging

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from audiozen.acoustics.audio_feature import (
    build_complex_ideal_ratio_mask,
    decompress_cIRM,
    drop_band,
    tune_dB_FS,
)
from audiozen.metrics import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.si_sdr = SISDR()

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
        _, _, clean_real, clean_imag = self.torch_stft(clean)

        cIRM = build_complex_ideal_ratio_mask(
            noisy_real, noisy_imag, clean_real, clean_imag
        )  # [B, F, T, 2]

        cIRM = drop_band(
            cIRM.permute(0, 3, 1, 2),
            self.model.module.num_groups_in_drop_band,
        ).permute(0, 2, 3, 1)

        noisy_mag = noisy_mag.unsqueeze(1)
        cRM = self.model(noisy_mag)  # [B, 2, F ,T]
        cRM = cRM.permute(0, 2, 3, 1)
        loss = self.loss_function(cIRM, cRM)

        return loss

    def training_epoch_end(self, training_epoch_output):
        loss_mean = torch.mean(torch.tensor(training_epoch_output))

        if self.rank == 0:
            logger.info(f"Training loss on epoch {self.current_epoch}: {loss_mean}")
            self.writer.add_scalar(f"Loss/Train", loss_mean, self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean, fpath = batch
        clean = clean.to(self.device)
        noisy = noisy.to(self.device)

        noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
        _, _, clean_real, clean_imag = self.torch_stft(clean)
        cIRM = build_complex_ideal_ratio_mask(
            noisy_real, noisy_imag, clean_real, clean_imag
        )  # [B, F, T, 2]

        noisy_mag = noisy_mag.unsqueeze(1)
        cRM = self.model(noisy_mag)  # [B, 2, F ,T]
        cRM = cRM.permute(0, 2, 3, 1)

        cRM = decompress_cIRM(cRM)

        enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
        enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
        enhanced = self.torch_istft(
            (enhanced_real, enhanced_imag),
            length=noisy.size(-1),
            input_type="real_imag",
        )

        enhanced, *_ = tune_dB_FS(enhanced, target_dB_FS=-26)

        return noisy, clean, enhanced, fpath

    def compute_validation_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, fpath = step_out
        si_sdr = self.si_sdr(enhanced, clean)
        stoi = self.stoi(enhanced, clean)
        pesq_wb = self.pesq_wb(enhanced, clean)
        pesq_nb = self.pesq_nb(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)
        return {"fpath": fpath} | stoi | pesq_wb | pesq_nb | si_sdr | dns_mos

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=40, prefer="threads")(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            with pd.option_context("display.max_columns", 2000):
                logger.info(f"\n {df_metrics.describe()}")

            df_metrics_mean = df_metrics.mean(numeric_only=True)

            score += df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(
                    f"metrics_{dataloader_idx}/{metric}", value, self.current_epoch
                )

        return score

    def compute_test_metrics(self, step_out):
        enhanced, fpath = step_out
        dns_mos_dict = self.dns_mos(enhanced)
        dns_mos_dict["fpath"] = fpath
        return dns_mos_dict

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean, fpath = batch
        clean = clean.to(self.device)
        noisy = noisy.to(self.device)
        enhanced = self.model(noisy)
        enhanced, *_ = tune_dB_FS(enhanced, target_dB_FS=-26)
        return noisy, clean, enhanced, fpath

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
            logger.info(f"\n {df_metrics.describe()}")
