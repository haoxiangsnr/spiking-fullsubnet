import yaml

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
    },
    {
        "team": "Intel Neuromorphic Computing Lab",
        "model": "Intel proprietary DNS",
        "date": "2023-02-28",
        "SI-SNR": 12.71,
        "SI-SNRi_data": 12.71 - 7.62,
        "SI-SNRi_enc+dec": 12.71 - 7.62,
        "MOS_ovrl": 3.09 - 2.45,
        "MOS_sig": 3.35,
        "MOS_bak": 4.08,
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": 1901 * 10**3,
        "size_kilobytes": 3802,  # float16
    },
    # ======================= Official Noisy =======================
    {
        "team": "Clairaudience",
        "model": "validation_set",
        "date": "2023-07-25",
        "SI-SNR": 6.89,
        "SI-SNRi_data": None,
        "SI-SNRi_enc+dec": None,
        "MOS_ovrl": 2.40,
        "MOS_sig": 3.10,
        "MOS_bak": 2.66,
        "latency_enc+dec_ms": None,
        "latency_total_ms": None,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": None,
        "size_kilobytes": None,
    },
    # ======================= FSB + SNN =======================
    {
        "team": "Clairaudience",
        "model": "FSB+SNN",
        "date": "2023-07-25",
        "SI-SNR": 14.24,
        "SI-SNRi_data": 14.24 - 6.89,
        "SI-SNRi_enc+dec": 14.24 - 6.89,
        "MOS_ovrl": 2.92 - 2.40,
        "MOS_sig": 3.25,
        "MOS_bak": 3.88,
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 16.036,
        "power_proxy_Ops/s": 24.6 * 10**6,
        "PDP_proxy_Ops": 395078,
        "params": 911 * 10**3,
        "size_kilobytes": 3644,
    },
    # ======================= FSB + SNN + GAN =======================
    {
        "team": "Clairaudience",
        "model": "baseline_sharedParam",
        "date": "2023-07-25",
        "SI-SNR": 14.06,
        "SI-SNRi_data": 13.92 - 6.89,
        "SI-SNRi_enc+dec": 13.92 - 6.89,
        "MOS_ovrl": 2.94,
        "MOS_sig": 3.27,
        "MOS_bak": 3.87,
        "latency_enc+dec_ms": 0.036,
        "latency_total_ms": 8.036,
        "power_proxy_Ops/s": None,
        "PDP_proxy_Ops": None,
        "params": 910 * 10**3,
        "size_kilobytes": None,
    },
]

if __name__ == "__main__":
    with open("metricsboard_track_1_validation.yml", "w") as outfile:
        yaml.dump(entries, outfile, sort_keys=False)

    # save to markdown table
    with open("metricsboard_track_1_validation.md", "w") as outfile:
        outfile.write(
            "| Team | Model | Date | SI-SNR | SI-SNRi_data | SI-SNRi_enc+dec | MOS_ovrl | MOS_sig | MOS_bak | latency_enc+dec_ms | latency_total_ms | power_proxy_Ops/s | PDP_proxy_Ops | params | size_kilobytes |\n"
        )
        outfile.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for entry in entries:
            outfile.write(
                "| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.0f} | {:.0f} |\n".format(
                    entry["team"],
                    entry["model"],
                    entry["date"],
                    entry["SI-SNR"],
                    entry["SI-SNRi_data"] if entry["SI-SNRi_data"] is not None else 0,
                    entry["SI-SNRi_enc+dec"]
                    if entry["SI-SNRi_enc+dec"] is not None
                    else 0,
                    entry["MOS_ovrl"],
                    entry["MOS_sig"],
                    entry["MOS_bak"],
                    entry["latency_enc+dec_ms"]
                    if entry["latency_enc+dec_ms"] is not None
                    else 0,
                    entry["latency_total_ms"]
                    if entry["latency_total_ms"] is not None
                    else 0,
                    entry["power_proxy_Ops/s"]
                    if entry["power_proxy_Ops/s"] is not None
                    else 0,
                    entry["PDP_proxy_Ops"] if entry["PDP_proxy_Ops"] is not None else 0,
                    entry["params"] if entry["params"] is not None else 0,
                    entry["size_kilobytes"]
                    if entry["size_kilobytes"] is not None
                    else 0,
                )
            )
