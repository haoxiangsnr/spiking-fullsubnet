import logging
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from audiozen.acoustics.audio_feature import save_wav
from audiozen.loss import freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer.base_trainer import BaseTrainer

# from spikingjelly.activation_based import functional


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.si_sdr = SISDR()
        self.synops_reg_coeff = self.trainer_config.get("synops_reg_coeff", 0)
        # assert self.synops_reg_coeff > 0
        self.target_synops = float(self.trainer_config.get("target_synops", 10))

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        # print(f"noisy: {noisy.size()}")
        # print(f"clean: {clean.size()}")
        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        # functional.reset_net(self.model)
        enhanced, fb_all_layer_outputs, sb_all_layer_outputs = self.model(noisy)
        # synops = self.compute_firing_rate(fb_all_layer_outputs, sb_all_layer_outputs)
        # print(f"enhanced: {enhanced.size()}")
        # exit()
        # print(f"{batch_idx} synops {synops}")
        loss = freq_MAE(enhanced, clean) + mag_MAE(
            enhanced, clean
        )  # + self.synops_reg_coeff * ((synops - self.target_synops) ** 2)

        return loss

    def compute_synops(self, fb_all_layer_outputs, sb_all_layer_outputs):
        synops = 0.0
        for i in range(1, len(fb_all_layer_outputs) - 1):
            synops += (
                torch.gt(fb_all_layer_outputs[i], 0).float().mean()
                * fb_all_layer_outputs[i].size(-1)
                * (
                    fb_all_layer_outputs[i + 1].size(-1)
                    + fb_all_layer_outputs[i].size(-1)
                )
            )
        for i in range(len(sb_all_layer_outputs)):
            for j in range(1, len(sb_all_layer_outputs[i]) - 1):
                # print(sb_all_layer_outputs[i][j].size())
                synops += (
                    torch.gt(sb_all_layer_outputs[i][j], 0).float().mean()
                    * sb_all_layer_outputs[i][j].size(-1)
                    * (
                        sb_all_layer_outputs[i][j + 1].size(-1)
                        + sb_all_layer_outputs[i][j].size(-1)
                    )
                )
        return synops

    def compute_firing_rate(self, fb_all_layer_outputs, sb_all_layer_outputs):
        avg_firing_rate = 0.0
        num_neurons = 0.0
        for i in range(1, len(fb_all_layer_outputs) - 1):
            events = fb_all_layer_outputs[i]
            avg_firing_rate += events.mean() * events.size(-1)
            num_neurons += events.size(-1)

        for i in range(len(sb_all_layer_outputs)):
            for j in range(1, len(sb_all_layer_outputs[i]) - 1):
                events = sb_all_layer_outputs[i][j]
                avg_firing_rate += events.mean() * events.size(-1)
                num_neurons += events.size(-1)
        return avg_firing_rate / num_neurons

    def training_epoch_end(self, training_epoch_output):
        loss_mean = torch.mean(torch.tensor(training_epoch_output))

        if self.rank == 0:
            logger.info(f"Training loss on epoch {self.current_epoch}: {loss_mean}")
            self.writer.add_scalar(f"Loss/Train", loss_mean, self.current_epoch)

    # @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean = batch
        clean = clean.to(self.device)
        noisy = noisy.to(self.device)
        # functional.reset_net(self.model)

        enhanced, fb_all_layer_outputs, sb_all_layer_outputs = self.model.module(noisy)

        return noisy, clean, enhanced, fb_all_layer_outputs, sb_all_layer_outputs

    def compute_validation_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, fb_all_layer_outputs, sb_all_layer_outputs = step_out
        si_sdr = self.si_sdr(enhanced, clean)
        stoi = self.stoi(enhanced, clean)
        pesq_wb = self.pesq_wb(enhanced, clean)
        pesq_nb = self.pesq_nb(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)

        synops = {
            "synops": self.compute_synops(
                fb_all_layer_outputs, sb_all_layer_outputs
            ).item()
        }
        # print(synops)
        return stoi | pesq_wb | pesq_nb | si_sdr | dns_mos | synops

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=40, backend="threading")(
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

    def compute_test_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced, fb_all_layer_outputs, sb_all_layer_outputs = step_out
        si_sdr = self.si_sdr(enhanced, clean)
        dns_mos = self.dns_mos(enhanced)

        synops = {
            "synops": self.compute_synops(
                fb_all_layer_outputs, sb_all_layer_outputs
            ).item()
        }
        # print(synops)
        return si_sdr | dns_mos | synops

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        noisy, clean = batch
        clean = clean.to(self.device)
        noisy = noisy.to(self.device)
        # functional.reset_net(self.model)

        enhanced, fb_all_layer_outputs, sb_all_layer_outputs = self.model.module(noisy)

        return noisy, clean, enhanced, fb_all_layer_outputs, sb_all_layer_outputs

    def test_epoch_end(self, outputs):
        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(
                f"Computing metrics on epoch {self.current_epoch} for dataloader {dataloader_idx}..."
            )

            rows = Parallel(n_jobs=40, backend="threading")(
                delayed(self.compute_validation_metrics)(dataloader_idx, step_out)
                for step_out in tqdm(dataloader_outputs)
            )

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
