from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.acoustics.audio_feature import save_wav
from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI, IntelSISNR, compute_neuronops, compute_synops
from audiozen.trainer.base_trainer_gan_accelerate_ddp_validate import BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.si_sdr = SISDR()
        self.intel_si_snr = IntelSISNR()
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

        # Check data stats (for debugging)
        # logger.info(
        #     f"[Noisy Audio] device: {self.accelerator.device}, shape: {noisy_y.shape}, mean: {noisy_y.mean()}, std: {noisy_y.std()}",
        #     main_process_only=False,
        # )

        one_labels = torch.ones(batch_size, 1, device=self.accelerator.device).float()
        clean_mag, *_ = self.torch_stft(clean_y)

        # ================== Train Generator ================== #
        self.optimizer_g.zero_grad()

        enhanced_y, enhanced_mag, *_ = self.model_g(noisy_y)

        # check data stats (for debugging)
        # logger.info(
        #     f"[Enhanced Audio] device: {self.accelerator.device}, shape: {enhanced_y.shape}, mean: {enhanced_y.mean()}, std: {enhanced_y.std()}",
        #     main_process_only=False,
        # )

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
            "loss_g": loss_g.item(),
            "loss_freq_mae": loss_freq_mae.item(),
            "loss_mag_mae": loss_mag_mae.item(),
            "loss_sdr": loss_sdr.item(),
            "loss_g_fake": loss_g_fake.item(),
            "loss_d": loss_d.item(),
            "loss_d_real": loss_d_real.item(),
            "loss_d_fake": loss_d_fake.item(),
        }

    def training_epoch_end(self, training_epoch_output):
        # Compute mean loss on all loss items on epoch
        for key in training_epoch_output[0].keys():
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                logger.info(f"Loss '{key}' on epoch {self.current_epoch}: {loss_mean}")
                self.writer.add_scalar(f"Train_Epoch/{key}", loss_mean, self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy_y, clean_y, noisy_file = batch
        noisy_y = noisy_y.to(self.accelerator.device)
        clean_y = clean_y.to(self.accelerator.device)

        enhanced_y, enhanced_mag, fb_out, sb_out = self.model_g(noisy_y)

        # save enhanced audio
        # stem = Path(noisy_file[0]).stem
        # enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
        # enhanced_dir.mkdir(exist_ok=True, parents=True)
        # enhanced_fpath = enhanced_dir / f"{stem}.wav"
        # save_wav(enhanced_y, enhanced_fpath.as_posix(), self.sr)

        # detach and move to cpu
        synops = compute_synops(
            fb_out,
            sb_out,
            shared_weights=self.config["model_g"]["args"]["shared_weights"],
        )
        neuron_ops = compute_neuronops(fb_out, sb_out)

        # to tensor
        synops = torch.tensor([synops], device=self.accelerator.device).unsqueeze(0)
        synops = synops.repeat(enhanced_y.shape[0], 1)
        neuron_ops = torch.tensor([neuron_ops], device=self.accelerator.device).unsqueeze(0)
        neuron_ops = neuron_ops.repeat(enhanced_y.shape[0], 1)

        return noisy_y, clean_y, enhanced_y, synops, neuron_ops

    def compute_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, synops, neuron_ops = step_out

        si_sdr = self.si_sdr(enhanced, clean)
        intel_si_snr = self.intel_si_snr(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)

        return si_sdr | intel_si_snr | dns_mos | {"synops": synops.item()} | {"neuron_ops": neuron_ops.item()}

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
            logger.info(f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}...")

            rows = []
            for step_out in tqdm(dataloader_outputs):
                rows += self.compute_batch_metrics(dataloader_idx, step_out)

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            score += df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.current_epoch)

        return score

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        noisy_y, clean_y, noisy_file = batch
        noisy_y = noisy_y.to(self.accelerator.device)
        clean_y = clean_y.to(self.accelerator.device)

        enhanced_y, enhanced_mag, fb_out, sb_out = self.model_g(noisy_y)

        # save enhanced audio
        for i in range(enhanced_y.shape[0]):
            enhanced_y_i = enhanced_y[i, :]
            stem = Path(noisy_file[i]).stem
            enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
            enhanced_dir.mkdir(exist_ok=True, parents=True)
            enhanced_fpath = enhanced_dir / f"{stem}.wav"
            save_wav(enhanced_y_i, enhanced_fpath.as_posix(), self.sr)

        # detach and move to cpu
        synops = compute_synops(
            fb_out,
            sb_out,
            shared_weights=self.config["model_g"]["args"]["shared_weights"],
        )
        neuron_ops = compute_neuronops(fb_out, sb_out)

        # to tensor
        synops = torch.tensor([synops], device=self.accelerator.device).unsqueeze(0)
        synops = synops.repeat(enhanced_y.shape[0], 1)
        neuron_ops = torch.tensor([neuron_ops], device=self.accelerator.device).unsqueeze(0)
        neuron_ops = neuron_ops.repeat(enhanced_y.shape[0], 1)

        return noisy_y, clean_y, enhanced_y, synops, neuron_ops

    def test_epoch_end(self, outputs):
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}...")

            # rows = []
            # for step_out in tqdm(dataloader_outputs):
            #     rows += self.compute_batch_metrics(dataloader_idx, step_out)

            # df_metrics = pd.DataFrame(rows)

            # df_metrics_mean = df_metrics.mean(numeric_only=True)
            # df_metrics_mean_df = df_metrics_mean.to_frame().T

            # logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
