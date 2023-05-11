import torch

load_path = "/mnt/private_xianghao/proj/Enhancement-Paas/egs2/denoise/exp/dns/16k/dns_16k_fsb_v1_mag_causal.yaml-202305011653/checkpoint-best.pt"
save_path = "/mnt/private_xianghao/proj/Enhancement-Paas/egs2/denoise/exp/dns/16k/dns_16k_fsb_v1_mag_causal.yaml-202305011653/checkpoint-best.pt.audiozen.pth"

ckpt = torch.load(load_path)

model_state_dict = ckpt["denoise"]

for key, value in model_state_dict.items():
    if "module." in key:
        new_key = key.replace("module.", "")
        model_state_dict[new_key] = value
        del model_state_dict[key]

torch.save(model_state_dict, save_path)
