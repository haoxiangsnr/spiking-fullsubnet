import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI, compute_neuronops, compute_synops
from audiozen.trainer.base_trainer_dualgan_accelerate_ddp_validate import BaseTrainer

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

        sig_scores = []
        bak_scores = []
        for audio in audio_list:
            metrics = self.dns_mos(audio, return_p808=False)
            sig_scores.append(metrics["SIG"])
            bak_scores.append(metrics["BAK"])

        # Average
        sig_scores = np.array(sig_scores)
        bak_scores = np.array(bak_scores)

        # Normalize
        sig_scores = (sig_scores - 1) / 4
        bak_scores = (bak_scores - 1) / 4

        return torch.from_numpy(sig_scores).float().to(
            self.accelerator.device
        ).unsqueeze(1), torch.from_numpy(bak_scores).float().to(
            self.accelerator.device
        ).unsqueeze(
            1
        )

    def training_step(self, batch, batch_idx):
        noisy_y, clean_y, _ = batch

        batch_size, *_ = noisy_y.shape

        one_labels = torch.ones(batch_size, 1, device=self.accelerator.device).float()
        clean_mag, *_ = self.torch_stft(clean_y)

        # ================== Train Generator ================== #
        self.optimizer_g.zero_grad()

        enhanced_y, enhanced_mag, *_ = self.model_g(noisy_y)

        pred_fake_sig = self.model_d_sig(clean_mag, enhanced_mag)  # [B, 1]
        loss_g_fake_sig = F.mse_loss(pred_fake_sig, one_labels)

        pred_fake_bak = self.model_d_bak(clean_mag, enhanced_mag)  # [B, 1]
        loss_g_fake_bak = 0.5 * F.mse_loss(pred_fake_bak, one_labels)

        loss_freq_mae = freq_MAE(enhanced_y, clean_y)
        loss_mag_mae = mag_MAE(enhanced_y, clean_y)
        loss_sdr = 0.001 * (100 - self.sisnr_loss(enhanced_y, clean_y))
        loss_g = (
            loss_freq_mae + loss_mag_mae + loss_sdr + loss_g_fake_sig + loss_g_fake_bak
        )

        self.accelerator.backward(loss_g)
        self.optimizer_g.step()

        # ================== Train First Discriminator ================== #
        sig_score, bak_score = self.batch_dns_mos(enhanced_y)

        self.optimizer_d_sig.zero_grad()
        pred_real = self.model_d_sig(clean_mag, clean_mag)
        pred_fake = self.model_d_sig(clean_mag, enhanced_mag.detach())
        loss_d_real = F.mse_loss(pred_real, one_labels)
        loss_d_fake = F.mse_loss(pred_fake, sig_score)
        loss_d_sig = loss_d_real + loss_d_fake

        self.accelerator.backward(loss_d_sig)
        self.optimizer_d_sig.step()

        # ================== Train Second Discriminator ================== #
        self.optimizer_d_bak.zero_grad()
        pred_real = self.model_d_bak(clean_mag, clean_mag)
        pred_fake = self.model_d_bak(clean_mag, enhanced_mag.detach())
        loss_d_real = F.mse_loss(pred_real, one_labels)
        loss_d_fake = F.mse_loss(pred_fake, bak_score)
        loss_d_bak = loss_d_real + loss_d_fake

        self.accelerator.backward(loss_d_bak)
        self.optimizer_d_bak.step()

        return {
            "loss_g": loss_g.item(),
            "loss_freq_mae": loss_freq_mae.item(),
            "loss_mag_mae": loss_mag_mae.item(),
            "loss_sdr": loss_sdr.item(),
            "loss_g_fake_sig": loss_g_fake_sig.item(),
            "loss_g_fake_bak": loss_g_fake_bak.item(),
            "loss_d_sig": loss_d_sig.item(),
            "loss_d_bak": loss_d_bak.item(),
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
        noisy_y, clean_y, filename = batch
        noisy_y = noisy_y.to(self.accelerator.device)
        clean_y = clean_y.to(self.accelerator.device)

        enhanced_y, enhanced_mag, fb_out, sb_out = self.model_g(noisy_y)

        # detach and move to cpu
        synops = compute_synops(fb_out, sb_out)
        neuron_ops = compute_neuronops(fb_out, sb_out)

        # to tensor
        synops = torch.tensor([synops], device=self.accelerator.device).unsqueeze(0)
        synops = synops.repeat(enhanced_y.shape[0], 1)
        neuron_ops = torch.tensor(
            [neuron_ops], device=self.accelerator.device
        ).unsqueeze(0)
        neuron_ops = neuron_ops.repeat(enhanced_y.shape[0], 1)

        return noisy_y, clean_y, enhanced_y, synops, neuron_ops

    def compute_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, synops, neuron_ops = step_out

        si_sdr = self.si_sdr(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)

        return (
            si_sdr
            | dns_mos
            | {"synops": synops.item()}
            | {"neuron_ops": neuron_ops.item()}
        )

    def compute_batch_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, synops, neuron_ops = step_out
        assert noisy.ndim == clean.ndim == enhanced.ndim == 2

        # [num_ranks * batch_size, num_samples]
        results = []
        for i in range(noisy.shape[0]):
            enhanced_i = enhanced[i, :]
            clean_i = clean[i, :]
            noisy_i = noisy[i, :]
            synops_i = synops[i, :]
            neuron_ops_i = neuron_ops[i, :]
            results.append(
                self.compute_metrics(
                    dataloader_idx,
                    (noisy_i, clean_i, enhanced_i, synops_i, neuron_ops_i),
                )
            )

        return results

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = []
            for step_out in tqdm(dataloader_outputs):
                rows += self.compute_batch_metrics(dataloader_idx, step_out)

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs):
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = []
            for step_out in tqdm(dataloader_outputs):
                rows += self.compute_batch_metrics(dataloader_idx, step_out)

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
