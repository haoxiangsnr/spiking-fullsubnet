import pandas as pd
from accelerate.logging import get_logger

from audiozen.loss import SISNRLoss, freq_MAE, mag_MAE
from audiozen.metric import DNSMOS, SISDR, compute_neuronops, compute_synops
from audiozen.trainer_v2 import Trainer as BaseTrainer


logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dns_mos = DNSMOS(input_sr=self.sr, device=self.accelerator.process_index)
        self.sisnr_loss = SISNRLoss(return_neg=False)
        self.si_sdr = SISDR()
        self.north_star_metric = "si_sdr"

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        noisy_y, clean_y, _ = batch

        batch_size, *_ = noisy_y.shape

        enhanced_y, enhanced_mag, *_ = self.model(noisy_y)

        loss_freq_mae = freq_MAE(enhanced_y, clean_y)
        loss_mag_mae = mag_MAE(enhanced_y, clean_y)
        loss_sdr = self.sisnr_loss(enhanced_y, clean_y)
        loss_sdr_norm = 0.001 * (100 - loss_sdr)
        loss = loss_freq_mae + loss_mag_mae + loss_sdr_norm  # + loss_g_fake

        self.accelerator.backward(loss)

        # If accumulated gradients are ready, clip them
        if self.accelerator.sync_gradients:
            norm_before = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        self.optimizer.step()

        return {
            "loss": loss,
            "loss_freq_mae": loss_freq_mae,
            "loss_mag_mae": loss_mag_mae,
            "loss_sdr": loss_sdr,
            "loss_sdr_norm": loss_sdr_norm,
            "norm_before": norm_before,
        }

    def evaluation_step(self, batch, batch_idx, dataloader_id=0):
        mix_y, ref_y, id = batch

        # In this case, the batch size is larger than one. We need to iterate over the batch size later
        est_y, enh_mag, fb_all_layer_outputs, sb_all_layer_outputs = self.model(mix_y)  # [B, T]

        # Compute synops and neuronops, which has been averaged over the samples in the batch
        synops = compute_synops(
            fb_all_layer_outputs,
            sb_all_layer_outputs,
            shared_weights=self.accelerator.unwrap_model(self.model).args.shared_weights,
        )
        neuron_ops = compute_neuronops(fb_all_layer_outputs, sb_all_layer_outputs)

        # Compute other metrics for each sample in the batch
        batch_size = mix_y.shape[0]
        out = []
        for i in range(batch_size):
            ref_y_i = ref_y[i].squeeze(0).detach().cpu().numpy()
            est_y_i = est_y[i].squeeze(0).detach().cpu().numpy()

            si_sdr = self.si_sdr(est_y_i, ref_y_i)
            dns_mos = self.dns_mos(est_y_i)

            out_i = si_sdr | dns_mos | {"synops": synops} | {"neuron_ops": neuron_ops}
            out.append(out_i)

        return out

    def evaluation_epoch_end(self, outputs, log_to_tensorboard=True):
        # We use this variable to store the score for the current epoch
        score = 0.0

        for dl_id, dataloader_outputs in outputs.items():
            logger.info(f"Computing metrics on epoch {self.state.epochs_trained} for dataloader `{dl_id}`...")

            # It should be a list of dictionaries, where each dictionary contains the metrics for a sample
            metric_dict_list = dataloader_outputs
            logger.info(f"The number of samples in the dataloader `{dl_id}` is {len(metric_dict_list)}")

            # Use pandas to compute the mean of all metrics and save them to a csv file
            df_metrics = pd.DataFrame(metric_dict_list)
            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T  # Convert mean to a DataFrame

            time_now = self._get_time_now()
            df_metrics.to_csv(
                self.metrics_dir / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}.csv",
                index=False,
            )
            df_metrics_mean_df.to_csv(
                self.metrics_dir / f"dl_{dl_id}_epoch_{self.state.epochs_trained}_{time_now}_mean.csv",
                index=False,
            )

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            # We use the `north_star_metric` to compute the score. In this case, it is the `si_sdr`.
            score += df_metrics_mean[self.north_star_metric]

            if log_to_tensorboard:
                for metric, value in df_metrics_mean.items():
                    self.writer.add_scalar(f"metrics_{dl_id}/{metric}", value, self.state.epochs_trained)

        return score
