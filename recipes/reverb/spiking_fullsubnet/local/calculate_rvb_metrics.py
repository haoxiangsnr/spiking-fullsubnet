import sys

sys.path.append("../")

from pathlib import Path

import accelerate
import numpy as np
import pandas as pd
import pysepm
import torch
from accelerate import Accelerator
from dataloader import SimDTDataset
from tqdm import tqdm

from audiozen.metric import DNSMOS, IntelSISNR

pd.set_option("display.float_format", lambda x: "%.3f" % x)


def calculate_sim_data(est_scp_fpath, ref_scp_fpath, batch_size=1, num_workers=0):
    accelerator = Accelerator()
    dataset = SimDTDataset(est_scp_fpath, ref_scp_fpath, is_train=False, sr=16000)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    dataloader = accelerator.prepare(dataloader)

    dnsmos = DNSMOS(input_sr=16000, device=accelerator.process_index)

    dnsmos_ovrl_output = []
    dnsmos_sig_output = []
    dnsmos_bak_output = []
    fw_snr_seg_output = []
    snr_seg_output = []
    llr_output = []
    cd_output = []
    pesq_output = []
    srmr_output = []

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        est_y, ref_y, id = batch

        dnsmos_ovrl_list = []
        dnsmos_sig_list = []
        dnsmos_bak_list = []
        fw_snr_seg_list = []
        snr_seg_list = []
        llr_list = []
        cd_list = []
        pesq_list = []
        srmr_list = []

        for i in range(ref_y.shape[0]):
            dnsmos_score = dnsmos(est_y[i])
            dnsmos_ovrl_list.append(dnsmos_score["OVRL"])
            dnsmos_sig_list.append(dnsmos_score["SIG"])
            dnsmos_bak_list.append(dnsmos_score["BAK"])

            ref_y_i = ref_y[i].cpu().numpy()
            est_y_i = est_y[i].cpu().numpy()
            est_y_i = est_y_i[: ref_y_i.shape[0]]

            print(ref_y_i.shape, est_y_i.shape)

            fw_snr_seg = pysepm.fwSNRseg(ref_y_i, est_y_i, 16000)
            fw_snr_seg_list.append(fw_snr_seg)

            snr_seg = pysepm.SNRseg(ref_y_i, est_y_i, 16000)
            snr_seg_list.append(snr_seg)

            llr = pysepm.llr(ref_y_i, est_y_i, 16000)
            llr_list.append(llr)

            cd = pysepm.cepstrum_distance(ref_y_i, est_y_i, 16000)
            cd_list.append(cd)

            pesq = pysepm.pesq(ref_y_i, est_y_i, 16000)
            pesq_list.append(pesq)

            srmr = pysepm.srmr(ref_y_i, 16000)
            srmr_list.append(srmr)

        # to Tensor
        dnsmos_ovrl_list = torch.from_numpy(np.array(dnsmos_ovrl_list))
        dnsmos_sig_list = torch.from_numpy(np.array(dnsmos_sig_list))
        dnsmos_bak_list = torch.from_numpy(np.array(dnsmos_bak_list))
        fw_snr_seg_list = torch.from_numpy(np.array(fw_snr_seg_list))
        snr_seg_list = torch.from_numpy(np.array(snr_seg_list))
        llr_list = torch.from_numpy(np.array(llr_list))
        cd_list = torch.from_numpy(np.array(cd_list))
        pesq_list = torch.from_numpy(np.array(pesq_list))
        srmr_list = torch.from_numpy(np.array(srmr_list))

        # Gather the results
        # [[64, 1], [64, 1], [64, 1], [64, 1], ...]
        dnsmos_ovrl_list = dnsmos_ovrl_list.to(accelerator.device)
        dnsmos_sig_list = dnsmos_sig_list.to(accelerator.device)
        dnsmos_bak_list = dnsmos_bak_list.to(accelerator.device)
        fw_snr_seg_list = fw_snr_seg_list.to(accelerator.device)
        snr_seg_list = snr_seg_list.to(accelerator.device)
        llr_list = llr_list.to(accelerator.device)
        cd_list = cd_list.to(accelerator.device)
        pesq_list = pesq_list.to(accelerator.device)
        srmr_list = srmr_list.to(accelerator.device)

        dnsmos_ovrl_list = accelerator.gather_for_metrics(dnsmos_ovrl_list)
        dnsmos_sig_list = accelerator.gather_for_metrics(dnsmos_sig_list)
        dnsmos_bak_list = accelerator.gather_for_metrics(dnsmos_bak_list)
        fw_snr_seg_list = accelerator.gather_for_metrics(fw_snr_seg_list)
        snr_seg_list = accelerator.gather_for_metrics(snr_seg_list)
        llr_list = accelerator.gather_for_metrics(llr_list)
        cd_list = accelerator.gather_for_metrics(cd_list)
        pesq_list = accelerator.gather_for_metrics(pesq_list)
        srmr_list = accelerator.gather_for_metrics(srmr_list)

        dnsmos_ovrl_output.append(dnsmos_ovrl_list)
        dnsmos_sig_output.append(dnsmos_sig_list)
        dnsmos_bak_output.append(dnsmos_bak_list)
        fw_snr_seg_output.append(fw_snr_seg_list)
        snr_seg_output.append(snr_seg_list)
        llr_output.append(llr_list)
        cd_output.append(cd_list)
        pesq_output.append(pesq_list)
        srmr_output.append(srmr_list)

    mean_dnsmos_ovrl = torch.mean(torch.cat(dnsmos_ovrl_output))
    mean_dnsmos_sig = torch.mean(torch.cat(dnsmos_sig_output))
    mean_dnsmos_bak = torch.mean(torch.cat(dnsmos_bak_output))
    mean_fw_snr_seg = torch.mean(torch.cat(fw_snr_seg_output))
    mean_snr_seg = torch.mean(torch.cat(snr_seg_output))
    mean_llr = torch.mean(torch.cat(llr_output))
    mean_cd = torch.mean(torch.cat(cd_output))
    mean_pesq = torch.mean(torch.cat(pesq_output))
    mean_srmr = torch.mean(torch.cat(srmr_output))

    if accelerator.is_local_main_process:
        print(f"Calculated metrics for {est_scp_fpath} ...")
        print(
            f"mean_dnsmos_ovrl: {mean_dnsmos_ovrl}, mean_dnsmos_sig: {mean_dnsmos_sig}, mean_dnsmos_bak: {mean_dnsmos_bak}, mean_fw_snr_seg: {mean_fw_snr_seg}, mean_snr_seg: {mean_snr_seg}, mean_llr: {mean_llr}, mean_cd: {mean_cd}, mean_pesq: {mean_pesq}, mean_srmr: {mean_srmr}"
        )


if __name__ == "__main__":
    est_scp_fpath = Path("/home/xhao/proj/spiking-fullsubnet/recipes/reverb/data/dt_simu_1ch.scp")
    ref_scp_fpath = Path("/home/xhao/proj/spiking-fullsubnet/recipes/reverb/data/dt_cln.scp")
    calculate_sim_data(est_scp_fpath, ref_scp_fpath)
