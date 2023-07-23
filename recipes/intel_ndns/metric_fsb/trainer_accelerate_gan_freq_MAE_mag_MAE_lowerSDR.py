import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger

from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer.base_trainer_gan_accelerate import BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.si_sdr = SISDR()
        self.sisnr_loss = SISNRLoss()

    @torch.no_grad()
    def batch_dns_mos(self, x):
        """Calculate MOS score for batch of audio, [B, 1, T] => [B, 1]"""
        audio_list = list(x.squeeze(1).detach().cpu().numpy())

        scores = []
        for audio in audio_list:
            scores.append(self.dns_mos(audio, return_p808=False)["OVRL"])

        scores = np.array(scores)

        # Normalize
        scores = (scores - 1.0) / 4.0

        return torch.from_numpy(scores).float().to(x.device).unsqueeze(1)

    def training_step(self, batch, batch_idx):
        noisy_y, clean_y, _ = batch

        batch_size, *_ = noisy_y.shape

        one_labels = torch.ones(batch_size, 1, device=self.accelerator.device).float()
        clean_mag, *_ = self.torch_stft(clean_y)

        # ================== Train Generator ================== #
        self.optimizer_g.zero_grad()

        enhanced_y, enhanced_mag = self.model_g(noisy_y)

        pred_fake = self.model_d(clean_mag, enhanced_mag)  # [B, 1]
        loss_g_fake = 0.05 * F.mse_loss(pred_fake, one_labels)
        loss_freq_mae = freq_MAE(enhanced_y, clean_y)
        loss_mag_mae = mag_MAE(enhanced_y, clean_y)
        loss_sdr = 0.001 * (100 - self.sisnr_loss(enhanced_y, clean_y))
        loss_g = loss_freq_mae + loss_mag_mae + loss_g_fake + loss_sdr

        self.accelerator.backward(loss_g)
        self.optimizer_g.step()

        # ================== Train Discriminator ================== #
        self.optimizer_d.zero_grad()

        pred_real = self.model_d(clean_mag, clean_mag)
        pred_fake = self.model_d(clean_mag, enhanced_mag.detach())
        mos_score = self.batch_dns_mos(enhanced_y)
        loss_d_real = F.mse_loss(pred_real, one_labels)
        loss_d_fake = F.mse_loss(pred_fake, mos_score)
        loss_d = loss_d_real + loss_d_fake

        self.accelerator.backward(loss_d)
        self.optimizer_d.step()

        return {
            "loss_g": loss_g,
            "loss_freq_mae": loss_freq_mae,
            "loss_mag_mae": loss_mag_mae,
            "loss_sdr": loss_sdr,
            "loss_g_fake": loss_g_fake,
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
                logger.info(f"Loss '{key}' on epoch {self.current_epoch}: {loss_mean}")
                self.writer.add_scalar(
                    f"Train_Epoch/{key}", loss_mean, self.current_epoch
                )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy_y, clean_y, _ = batch
        noisy_y = noisy_y.to(self.accelerator.device)
        clean_y = clean_y.to(self.accelerator.device)

        enhanced_y, enhanced_mag = self.model_g(noisy_y)
        return noisy_y, clean_y, enhanced_y

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

            rows = []
            for step_out in dataloader_outputs:
                rows.append(self.compute_validation_metrics(dataloader_idx, step_out))

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            score += df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(
                    f"metrics_{dataloader_idx}/{metric}", value, self.current_epoch
                )

        return score
