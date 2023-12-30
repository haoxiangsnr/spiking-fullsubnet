import pandas as pd
import torch
from accelerate.logging import get_logger
from tqdm import tqdm

from audiozen.common_trainer import Trainer as BaseTrainer
from audiozen.loss import SISNRLoss
from audiozen.metric import DNSMOS, PESQ, SISDR, STOI, IntelSISNR
from audiozen.pit import PairwiseNegSDR, PITWrapper

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
        self.sisnr_loss = SISNRLoss(return_neg=True)
        self.north_star_metric = "intel_si_snr"
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

    def training_epoch_end(self, training_epoch_output):
        # Compute mean loss on all loss items on epoch
        for key in training_epoch_output[0].keys():
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                logger.info(f"Loss '{key}' on epoch {self.state.epochs_trained}: {loss_mean}")
                self.writer.add_scalar(f"Train_Epoch/{key}", loss_mean, self.state.epochs_trained)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mix_y, ref_y, _ = batch
        est_y, *_ = self.model(mix_y)

        # save enhanced audio
        # stem = Path(noisy_file[0]).stem
        # enhanced_dir = self.enhanced_dir / f"dataloader_{dataloader_idx}"
        # enhanced_dir.mkdir(exist_ok=True, parents=True)
        # enhanced_fpath = enhanced_dir / f"{stem}.wav"
        # save_wav(enhanced_y, enhanced_fpath.as_posix(), self.sr)

        # detach and move to cpu
        # synops = compute_synops(
        #     fb_out,
        #     sb_out,
        #     shared_weights=self.config["model_g"]["args"]["shared_weights"],
        # )
        # neuron_ops = compute_neuronops(fb_out, sb_out)

        # to tensor
        # synops = torch.tensor([synops], device=self.accelerator.device).unsqueeze(0)
        # synops = synops.repeat(enhanced_y.shape[0], 1)
        # neuron_ops = torch.tensor([neuron_ops], device=self.accelerator.device).unsqueeze(0)
        # neuron_ops = neuron_ops.repeat(enhanced_y.shape[0], 1)

        return mix_y, ref_y, est_y  # , synops, neuron_ops

    def validation_epoch_end(self, outputs):
        score = 0.0

        for dataloader_idx, dataloader_outputs in enumerate(outputs):
            logger.info(f"Computing metrics on epoch {self.state.epochs_trained} for dataloader {dataloader_idx}...")

            rows = []
            for step_out in tqdm(dataloader_outputs):
                rows += self.compute_batch_metrics(dataloader_idx, step_out)

            df_metrics = pd.DataFrame(rows)

            df_metrics_mean = df_metrics.mean(numeric_only=True)
            df_metrics_mean_df = df_metrics_mean.to_frame().T

            logger.info(f"\n{df_metrics_mean_df.to_markdown()}")

            score += df_metrics_mean["OVRL"]

            for metric, value in df_metrics_mean.items():
                self.writer.add_scalar(f"metrics_{dataloader_idx}/{metric}", value, self.state.epochs_trained)

        return score

    def compute_metrics(self, dataloader_idx, step_out):
        mix_y, ref_y, est_y = step_out

        # mix_y: [num_samples]
        # ref_y: [num_spks, num_samples]
        # est_y: [num_spks, num_samples]
        # unsqueeze for compatibility with metrics
        loss, est_y = self.pit_wrapper(est_y.unsqueeze(0), ref_y.unsqueeze(0))
        est_y = est_y.squeeze(0)

        si_sdr = self.si_sdr(est_y, ref_y)
        intel_si_snr = self.intel_si_snr(est_y, ref_y)
        dns_mos = self.dns_mos(est_y)

        return si_sdr | intel_si_snr | dns_mos | {"pit_loss": loss}

    def compute_batch_metrics(self, dataloader_idx, step_out):
        noisy, clean, enhanced = step_out
        assert noisy.shape[-1] == clean.shape[-1] == enhanced.shape[-1]

        # [num_ranks * batch_size, num_samples]
        results = []
        for i in range(noisy.shape[0]):
            enhanced_i = enhanced[i, ...]
            clean_i = clean[i, ...]
            noisy_i = noisy[i, ...]
            results.append(
                self.compute_metrics(
                    dataloader_idx,
                    (noisy_i, clean_i, enhanced_i),
                )
            )

        return results
