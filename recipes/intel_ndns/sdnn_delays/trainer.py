import logging

import pandas as pd
import torch
from joblib import Parallel, delayed
from lava.lib.dl import slayer
from tqdm import tqdm

from audiozen.metric import DNSMOS, SISDR
from audiozen.trainer_backup.base_trainer import BaseTrainer


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr)
        self.si_sdr = SISDR()

    def training_step(self, batch, batch_idx):
        noisy, clean, noise = batch

        out_delay = 0

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        noisy_mag, noisy_phase, *_ = self.torch_stft(noisy)  # [B, F, T]
        clean_mag, *_ = self.torch_stft(clean)

        enhanced_mag = self.model(noisy_mag)

        noisy_phase = slayer.axon.delay(noisy_phase, out_delay)
        clean_mag = slayer.axon.delay(clean_mag, out_delay)
        clean = slayer.axon.delay(clean, self.n_fft // 4 * out_delay)

        clean_rec = self.torch_istft([enhanced_mag, noisy_phase], length=noisy.shape[-1])

        loss = self.loss_function(clean_rec, clean)

        return loss

    @torch.no_grad()
    def training_epoch_end(self, training_epoch_output):
        loss_mean = torch.mean(torch.tensor(training_epoch_output))

        if self.rank == 0:
            logger.info(f"Training loss on epoch {self.current_epoch}: {loss_mean}")
            self.writer.add_scalar("Loss/Train", loss_mean, self.current_epoch)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        noisy, clean, noise = batch

        out_delay = 0

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)

        noisy_mag, noisy_phase, *_ = self.torch_stft(noisy)  # [B, F, T]
        clean_mag, *_ = self.torch_stft(clean)

        enhanced_mag = self.model.module(noisy_mag)

        noisy_phase = slayer.axon.delay(noisy_phase, out_delay)
        clean_mag = slayer.axon.delay(clean_mag, out_delay)
        clean = slayer.axon.delay(clean, self.n_fft // 4 * out_delay)

        clean_rec = self.torch_istft([enhanced_mag, noisy_phase])

        return noisy, clean, clean_rec

    def compute_validation_metrics(self, step_out):
        noisy, clean, clean_rec = step_out
        si_sdr = self.si_sdr(clean_rec, clean)
        dns_mos = self.dns_mos(clean_rec)
        return si_sdr | dns_mos

    def validation_epoch_end(self, outputs):
        score = 0.0
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}...")

            rows = Parallel(n_jobs=40, prefer="threads")(
                delayed(self.compute_validation_metrics)(step_out) for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            score += df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.current_epoch)

        return score
