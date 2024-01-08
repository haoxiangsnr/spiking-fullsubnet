import torch


def collate_fn_wsj0mix_train(samples, sr, duration):
    target_num_frames = int(sr * duration)

    batch_mix, batch_ref, batch_stem = [], [], []

    for sample in samples:
        mix, ref = fix_num_frames(sample, target_num_frames, sr, random_start=True)


def fix_num_frames(sample, target_num_frames, sr, random_start):
    mix = sample[1].unsqueeze(0)  # [1, num_samples]
    ref = sample[2]  # [num_spks, num_samples]

    num_channels, num_frames = ref.shape
    num_seconds = torch.div(num_frames, sr, rounding_mode="floor")
    target_seconds = torch.div(target_num_frames, sr, rounding_mode="floor")

    if num_frames >= target_num_frames:
        if random_start:
            if num_frames > target_num_frames:
                start = torch.randint(num_seconds - target_seconds + 1, [1]) * sr
                mix = mix[:, start:]
                ref = ref[:, start:]
    else:
        num_padding = target_num_frames - num_frames
        pad = torch.zeros([1, num_padding], dtype=mix.dtype, device=mix.device)
        mix = torch.cat([mix, pad], dim=-1)
        ref = torch.cat([ref, pad.expand(num_channels, -1)], dim=-1)
