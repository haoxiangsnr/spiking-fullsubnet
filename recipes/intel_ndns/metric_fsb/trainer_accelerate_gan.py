import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm

from audiozen.acoustics.audio_feature import save_wav
from audiozen.loss import freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer.base_trainer_gan_accelerate import BaseTrainer

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
        noisy_y, clean_y, _ = batch
        batch_size, *_ = noisy_y.shape

        one_label = torch.ones(batch_size, 1).float()
        clean_mag, *_ = self.torch_stft(clean_y)

        # ================== Train Generator ================== #
        self.optimizer_g.zero_grad()

        enhanced_y, enhanced_mag = self.model_g(noisy_y)
        pred_fake = self.model_d(clean_mag, enhanced_mag)
        loss_g_fake = F.mse_loss(pred_fake, one_label)
        loss_time = self.loss_function(enhanced_y, clean_y)
        loss_g = loss_time + loss_g_fake

        self.accelerator.backward(loss_g)
        self.optimizer_g.step()

        # ================== Train Discriminator ================== #
        self.optimizer_d.zero_grad()

        pred_real = self.model_d(clean_mag, clean_mag)
        pred_fake = self.model_d(clean_mag, enhanced_mag.detach())
        mos_score = self.model_d.batch_dns_mos(enhanced_y)
        loss_d_real = F.mse_loss(pred_real, one_label)
        loss_d_fake = F.mse_loss(pred_fake, mos_score)
        loss_d = loss_d_real + loss_d_fake

        self.accelerator.backward(loss_d)
        self.optimizer_d.step()

        return {
            "loss_g": loss_g,
            "loss_g_fake": loss_g_fake,
            "loss_time": loss_time,
            "loss_d": loss_d,
            "loss_d_real": loss_d_real,
            "loss_d_fake": loss_d_fake,
        }

    def training_epoch_end(self, training_epoch_output):
        # Compute mean loss on all loss items on epoch
        for key in training_epoch_output[0].keys():
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                logger.info(f"Loss {key} on epoch {self.current_epoch}: {loss_mean}")
                self.writer.add_scalar(
                    f"Train_Epoch/{key}", loss_mean, self.current_epoch
                )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean, _ = batch
        clean = clean.to(self.device)
        noisy = noisy.to(self.device)

        enhanced = self.model(noisy)

        return noisy, clean, enhanced

    def compute_validation_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced = step_out
        si_sdr = self.si_sdr(enhanced, clean)
        stoi = self.stoi(enhanced, clean)
        pesq_wb = self.pesq_wb(enhanced, clean)
        pesq_nb = self.pesq_nb(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)
        return stoi | pesq_wb | pesq_nb | si_sdr | dns_mos

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=40)(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            score += df_metrics_mean["pesq_wb"]

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

        # save enhanced audio
        stem = Path(fpath[0]).stem
        enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
        enhanced_dir.mkdir(exist_ok=True, parents=True)
        enhanced_fpath = enhanced_dir / f"{stem}.wav"
        save_wav(enhanced, enhanced_fpath.as_posix(), self.sr)

        return noisy, clean, enhanced, fpath

    def test_epoch_end(self, outputs):
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=10)(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
