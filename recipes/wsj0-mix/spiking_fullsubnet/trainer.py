import pandas as pd
import torch
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.loss import SISNRLoss
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.pit import PairwiseNegSDR, PITWrapper
from audiozen.trainer import Trainer as BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.si_sdr = SISDR()
        self.sisnr_loss = SISNRLoss(return_neg=True)
        self.north_star_metric = "si_sdr"
        self.neg_si_sdr = PairwiseNegSDR()
        self.pit_wrapper = PITWrapper(self.neg_si_sdr)

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        mix_y, ref_y, _ = batch
        est_y, *_ = self.model(mix_y)  # [batch_size, num_spks, num_samples]

        loss, ordered_est_y = self.pit_wrapper(est_y, ref_y)
        self.accelerator.backward(loss)
        self.optimizer.step()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mix_y, ref_y, stem = batch
        if len(stem) != 1:
            raise ValueError(f"Expected batch size 1 during validation, got {len(stem)}")

        est_y, *_ = self.model(mix_y)
        loss, est_y = self.pit_wrapper(est_y, ref_y)

        # calculate metrics for each sample
        mix_y = mix_y.squeeze(0).detach().cpu()
        ref_y = ref_y.squeeze(0).detach().cpu()
        est_y = est_y.squeeze(0).detach().cpu()

        # compute metrics
        si_sdr = self.si_sdr(est_y, ref_y)
        dns_mos = self.dns_mos(est_y)
        pesq_nb = self.pesq_nb(est_y, ref_y)

        out = si_sdr | dns_mos | pesq_nb
        return [out]

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.state.epochs_trained} for dataloader {dataloader_idx}...")

            loss_dict_list = []
            for step_loss_dict_list in tqdm(dataloader_outputs):
                loss_dict_list.extend(step_loss_dict_list)

            df_metrics = pd.DataFrame(loss_dict_list)

            # Compute mean of all metrics
            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T
            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            time_now = self._get_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            score += df_metrics_mean[self.north_star_metric]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.state.epochs_trained)

        return score

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)
