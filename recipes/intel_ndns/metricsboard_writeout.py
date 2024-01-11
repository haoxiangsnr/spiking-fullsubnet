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
        "SI-SNRi_data": 12.50 - 7.62,
        "SI-SNRi_enc+dec": 12.50 - 7.62,
        "MOS_ovrl": 2.71,
        "MOS_sig": 3.21,
        "MOS_bak": 3.46,
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 11.59 * 10**6,
        "PDP_proxy_Ops": 0.09 * 10**6,
        "params": 525 * 10**3,
        "size_kilobytes": 465,
        "model_path": "./baseline_solution/sdnn_delays/Trained/network.pt",
    },
    {
        "team": "Intel Neuromorphic Computing Lab",
        "model": "Intel proprietary DNS",
        "date": "2023-02-28",
        "SI-SNR": 12.71,
        "SI-SNRi_data": 12.71 - 7.62,
        "SI-SNRi_enc+dec": 12.71 - 7.62,
        "MOS_ovrl": 3.09,
        "MOS_sig": 3.35,
        "MOS_bak": 4.08,
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": 1901 * 10**3,
        "size_kilobytes": 3802,
        "model_path": None,
    },
    # ======================= FSB + ALIF =======================
    {
        "team": "Clairaudience",
        "model": "ALIF",
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
        "model_path": "model_zoo/intel_ndns/spike_fsb/ALIF/checkpoints/best.tar",
    },
    {
        "team": "Clairaudience",
        "model": "model_S",
        "date": "2023-07-25",
        "SI-SNR": 13.67,
        "SI-SNRi_data": 13.67 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 13.67 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 2.95 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.25 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.93 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 29 * 10**6,
        "PDP_proxy_Ops": 234815,
        "params": 512 * 10**3,
        "size_kilobytes": 512 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_s/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_M",
        "date": "2023-07-26",
        "SI-SNR": 14.50,
        "SI-SNRi_data": 14.50 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.50 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.02 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.32 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.97 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 53.6 * 10**6,
        "PDP_proxy_Ops": 431 * 10**3,
        "params": 954 * 10**3,
        "size_kilobytes": 954 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_m/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_L",
        "date": "2023-07-27",
        "SI-SNR": 14.51,
        "SI-SNRi_data": 14.51 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.51 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.01 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.31 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.97 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 74101000,
        "PDP_proxy_Ops": 595475,
        "params": 1289 * 10**3,
        "size_kilobytes": 1289 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_l/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_XL",
        "date": "2023-07-27",
        "SI-SNR": 14.93,
        "SI-SNRi_data": 14.93 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.93 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.05 - custom_noisy["MOS_ovrl"],
        "MOS_sig": 3.35 - custom_noisy["MOS_sig"],
        "MOS_bak": 3.98 - custom_noisy["MOS_bak"],
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": 55911500,
        "PDP_proxy_Ops": 449305,
        "params": 1798 * 10**3,
        "size_kilobytes": 1798 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_xl/checkpoints/best",
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
            "model_path",
        ]
    ]
    df = df.sort_values(by=["SI-SNRi_enc+dec"], ascending=False)

    # save to markdown
    with open("metricsboard_track_1_validation.md", "w") as outfile:
        outfile.write(df.to_markdown(index=False))
