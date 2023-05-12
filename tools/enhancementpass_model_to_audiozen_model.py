from collections import OrderedDict

import torch

load_path = "/mnt/private_xianghao/proj/Enhancement-Paas/egs2/denoise/exp/dns/16k/dns_16k_fsb_v1_mag_causal.yaml-202305081744/checkpoint-15000.pt"
save_path = "/mnt/private_xianghao/proj/Enhancement-Paas/egs2/denoise/exp/dns/16k/dns_16k_fsb_v1_mag_causal.yaml-202305081744/checkpoint-15000.pt.audiozen.pth"

ckpt = torch.load(load_path)

model_state_dict = ckpt["denoise"]
new_model_state_dict = OrderedDict()

for key, value in model_state_dict.items():
    if "module." in key:
        new_key = key.replace("module.", "")
        new_model_state_dict[new_key] = value

torch.save(new_model_state_dict, save_path)
