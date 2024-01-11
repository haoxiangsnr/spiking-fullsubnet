import yaml


custom_noisy = {
    "SI-SNR": 7.37,
    "MOS_ovrl": 2.4360453928657417,
    "MOS_sig": 3.163788245485056,
    "MOS_bak": 2.6949790321576534,
}

entries = [
    # ======================= FSB + ALIF =======================
    {
        "team": "Clairaudience",
        "model": "Noisy",
        "date": "2023-07-25",
        "SI-SNR": 7.37,
        "SI-SNRi_data": 7.37,
        "SI-SNRi_enc+dec": 7.37,
        "MOS_ovrl": 2.4360453928657417,
        "MOS_sig": 3.163788245485056,
        "MOS_bak": 2.6949790321576534,
        "latency_enc+dec_ms": 0.030,
        "latency_total_ms": 32.030,
        "power_proxy_Ops/s": "",
        "PDP_proxy_Ops": "",
        "params": "",
        "size_kilobytes": "",
        "model_path": "",
    },
    {
        "team": "Clairaudience",
        "model": "model_S",
        "date": "2023-07-25",
        "SI-SNR": 13.89,
        "SI-SNRi_data": 13.89 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 13.89 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 2.97,
        "MOS_sig": 3.28,
        "MOS_bak": 3.93,
        "latency_enc+dec_ms": 0.030,
        "latency_total_ms": 32.030,
        "power_proxy_Ops/s": 29.24 * 10**6,
        "PDP_proxy_Ops": 940 * 10**3,
        "params": 521 * 10**3,
        "size_kilobytes": 521 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_s/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_M",
        "date": "2023-07-26",
        "SI-SNR": 14.71,
        "SI-SNRi_data": 14.71 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.71 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.05,
        "MOS_sig": 3.35,
        "MOS_bak": 3.97,
        "latency_enc+dec_ms": 0.030,
        "latency_total_ms": 32.030,
        "power_proxy_Ops/s": 53.6 * 10**6,
        "PDP_proxy_Ops": 1720 * 10**3,
        "params": 954 * 10**3,
        "size_kilobytes": 954 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_m/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_L",
        "date": "2023-07-27",
        "SI-SNR": 14.80,
        "SI-SNRi_data": 14.80 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 14.80 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.03,
        "MOS_sig": 3.33,
        "MOS_bak": 3.96,
        "latency_enc+dec_ms": 0.030,
        "latency_total_ms": 32.030,
        "power_proxy_Ops/s": 74101000,
        "PDP_proxy_Ops": 2370 * 10**3,
        "params": 1289 * 10**3,
        "size_kilobytes": 1289 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_l/checkpoints/best",
    },
    {
        "team": "Clairaudience",
        "model": "model_XL",
        "date": "2023-07-27",
        "SI-SNR": 15.20,
        "SI-SNRi_data": 15.20 - custom_noisy["SI-SNR"],
        "SI-SNRi_enc+dec": 15.20 - custom_noisy["SI-SNR"],
        "MOS_ovrl": 3.07,
        "MOS_sig": 3.37,
        "MOS_bak": 3.99,
        "latency_enc+dec_ms": 0.030,
        "latency_total_ms": 32.030,
        "power_proxy_Ops/s": 55911500,
        "PDP_proxy_Ops": 1790 * 10**3,
        "params": 1798 * 10**3,
        "size_kilobytes": 1798 * 4,
        "model_path": "model_zoo/intel_ndns/spike_fsb/baseline_xl/checkpoints/best",
    },
]

if __name__ == "__main__":
    import pandas as pd

    with open("metricsboard_track_1_test.yml", "w") as outfile:
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
    with open("metricsboard_track_1_test.md", "w") as outfile:
        outfile.write(df.to_markdown(index=False))
