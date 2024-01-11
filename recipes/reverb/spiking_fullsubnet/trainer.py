from pathlib import Path

import pandas as pd
import soundfile as sf
import torch.nn.functional as F
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI
from audiozen.trainer import Trainer as BaseTrainer


logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.stoi = STOI(sr=self.sr)
        self.pesq_wb = PESQ(sr=self.sr, mode="wb")
        self.pesq_nb = PESQ(sr=self.sr, mode="nb")
        self.sisnr_loss = SISNRLoss(return_neg=False)
        self.si_sdr = SISDR()
        self.north_star_metric = "si_sdr"

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        mix_y, ref_y, _ = batch
        est_y, *_ = self.model(mix_y)  # [batch_size, num_spks, num_samples]

        loss_freq_mae = freq_MAE(est_y, ref_y)
        loss_mag_mae = mag_MAE(est_y, ref_y)
        loss_time_mae = F.l1_loss(est_y, ref_y)
        loss = loss_freq_mae + loss_mag_mae + loss_time_mae

        self.accelerator.backward(loss)
        self.optimizer.step()

        return {
            "loss": loss,
            "loss_freq_mae": loss_freq_mae,
            "loss_mag_mae": loss_mag_mae,
            "loss_time_mae": loss_time_mae,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mix_y, ref_y, id = batch
        est_y, *_ = self.model(mix_y)

        if len(id) != 1:
            raise ValueError(f"Expected batch size 1 during validation, got {len(id)}")

        # calculate metrics
        mix_y = mix_y.squeeze(0).detach().cpu().numpy()
        ref_y = ref_y.squeeze(0).detach().cpu().numpy()
        est_y = est_y.squeeze(0).detach().cpu().numpy()

        si_sdr = self.si_sdr(est_y, ref_y)
        dns_mos = self.dns_mos(est_y)

        out = si_sdr | dns_mos
        return [out]

    def validation_epoch_end(self, outputs, log_to_tensorboard=True):
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

            time_now = self._get_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir / f"dl_{dataloader_idx}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")
            score += df_metrics_mean[self.north_star_metric]

            if log_to_tensorboard:
                for metric, value in df_metrics_mean.items():
                    self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.state.epochs_trained)

        return score

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        mix_y, fpath = batch
        fpath = Path(fpath[0])
        est_y, *_ = self.model(mix_y)

        # save audio
        est_y = est_y.squeeze(0).detach().cpu().numpy()

        mix_root = Path("/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL")
        est_root = Path("/nfs/xhao/data/reverb_challenge/kaldi/egs/reverb/s5/wav/spiking_fullsubnet")
        save_fpath = est_root / fpath.relative_to(mix_root)
        save_fpath.parent.mkdir(parents=True, exist_ok=True)

        sf.write(save_fpath, est_y, samplerate=self.sr)

    def test_epoch_end(self, outputs):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx):
        mix_y, fpath = batch
        fpath = Path(fpath[0])
        est_y, *_ = self.model(mix_y)

        # save audio
        est_y = est_y.squeeze(0).detach().cpu().numpy()

        mix_root = Path("/nfs/xhao/data/reverb_challenge/REVERB_DATA_OFFICIAL")
        est_root = Path("/nfs/xhao/data/reverb_challenge/kaldi/egs/reverb/s5/wav/spiking_fullsubnet")
        save_fpath = est_root / fpath.relative_to(mix_root)
        save_fpath.parent.mkdir(parents=True, exist_ok=True)

        sf.write(save_fpath, est_y, samplerate=self.sr)
