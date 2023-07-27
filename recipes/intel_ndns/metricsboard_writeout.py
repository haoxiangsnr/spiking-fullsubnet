import yaml

intel_noisy = {
    "SI-SNR": 7.62,
    "MOS_ovrl": 2.45,
    "MOS_sig": 3.19,
    "MOS_bak": 2.72,
}

custom_noisy = {
    "SI-SNR": 6.89,
    "MOS_ovrl": 2.40,
    "MOS_sig": 3.10,
    "MOS_bak": 2.66,
}

entries = [
    # ======================= Official baseline =======================
    {
        "team": "Intel Neuromorphic Computing Lab",
        "model": "Baseline SDNN solution",
        "date": "2023-02-20",
        "SI-SNR": 12.50,
        "SI-SNRi_data": 12.50 - intel_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 12.50 - intel_noisy["SI-SNR"],
        "MOS_ovrl": 2.71 - intel_noisy["MOS_ovrl"],
        "MOS_sig": 3.21 - intel_noisy["MOS_sig"],
        "MOS_bak": 3.46 - intel_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 11.59 * 10**6,
        "PDP_proxy_Ops": 0.09 * 10**6,
        "params": 525 * 10**3,
        "size_kilobytes": 465,
    },
    {
        "team": "Intel Neuromorphic Computing Lab",
        "model": "Intel proprietary DNS",
        "date": "2023-02-28",
        "SI-SNR": 12.71,
        "SI-SNRi_data": 12.71 - intel_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 12.71 - intel_noisy["SI-SNR"],
        "MOS_ovrl": 3.09 - intel_noisy["MOS_ovrl"],
        "MOS_sig": 3.35 - intel_noisy["MOS_sig"],
        "MOS_bak": 4.08 - intel_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": 1901 * 10**3,
        "size_kilobytes": 3802,  # float16
    },
    # ======================= FSB + ALIF =======================
    {
        "team": "Clairaudience",
        "model": "FSB+ALIF",
        "date": "2023-07-26",
        "SI-SNR": 13.68,
        "SI-SNRi_data": 13.68 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 13.68 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 2.75 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.16 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.61 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 16.036,
        "power_proxy_Ops/s": 14.6 * 10**6,
        "PDP_proxy_Ops": 229554,
        "params": 1580 * 10**3,
        "size_kilobytes": 1580 * 4,
    },
    # ======================= FSB + SNN =======================
    {
        "team": "Clairaudience",
        "model": "FSB+SNN",
        "date": "2023-07-25",
        "SI-SNR": 14.24,
        "SI-SNRi_data": 14.24 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.24 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 2.92 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.25 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.88 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 16.036,
        "power_proxy_Ops/s": 24.6 * 10**6,
        "PDP_proxy_Ops": 395078,
        "params": 911 * 10**3,
        "size_kilobytes": 911 * 4,
    },
    {
        "team": "Clairaudience",
        "model": "FSB+SNN (small)",
        "date": "2023-07-25",
        "SI-SNR": 14.09,
        "SI-SNRi_data": 14.09 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.09 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 2.90 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.23 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.86 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 16.036,
        "power_proxy_Ops/s": 24.6 * 10**6,
        "PDP_proxy_Ops": 395078,
        "params": 643 * 10**3,
        "size_kilobytes": 911 * 4,
    },
    # ======================= FSB + SNN + GAN =======================
    {
        "team": "Clairaudience",
        "model": "FSB + SNN + GAN + MF + SISDRLoss",
        "date": "2023-07-26",
        "SI-SNR": 14.43,
        "SI-SNRi_data": 14.43 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.43 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.00 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.31 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.95 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": 910 * 10**3,
        "size_kilobytes": 910 * 4,
    },
]

if __name__ == "__main__":
    import pandas as pd

    with open("metricsboard_track_1_validation.yml", "w") as outfile:
        yaml.dump(entries, outfile, sort_keys=False)

    # markdown table
    df = pd.DataFrame(entries)
    df = df[
        [
            "team",
            "model",
            "date",
            "SI-SNR",
            "SI-SNRi_data",
            "SI-SNRi_enc+dec",
            "MOS_ovrl",
            "MOS_sig",
            "MOS_bak",
            "latency_enc+dec_ms",
            "latency_total_ms",
            "power_proxy_Ops/s",
            "PDP_proxy_Ops",
            "params",
            "size_kilobytes",
        ]
    ]
    df = df.sort_values(by=["SI-SNRi_enc+dec"], ascending=False)

    # save to markdown
    with open("metricsboard_track_1_validation.md", "w") as outfile:
        outfile.write(df.to_markdown(index=False))
