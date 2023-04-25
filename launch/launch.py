import argparse
import json
import os
import tempfile
from pathlib import Path

"""
python launch.py -C dns_icassp_2020/bsrnn/baseline_bsrnn.toml -N 8 --run

--config: config path, e.g., dns_icassp_2020/bsrnn/baseline_bsrnn.toml
--gpu_num: gpu number, e.g., 8
--run: run the task, otherwise just print the command
--ghost_run: ghost run, just apply for gpu resource and sleep 2 days
"""


def config_path_to_task_flag(config_path, username="hx"):
    """Convert config path to task flag.

    recipes/dns_icassp_2023/bsrnn/baseline_bsrnn.toml => hx_dns_icassp_2023_bsrnn_baseline_bsrnn

    """
    print(f"Convert config path to task flag: {config_path}")
    abs_path = Path(config_path).absolute()
    config_name = abs_path.stem
    model_name = abs_path.parent.name
    data_name = abs_path.parent.parent.name
    task_flag = f"{username}_{data_name}_{model_name}_{config_name}"
    return task_flag, data_name, model_name, config_name


def main(args):
    config = {
        "Token": "vOwUVHKwR__zYB3t5tBleQ",
        "business_flag": "TEG_AILAB_SPEECH_shenzhen",
        "mount_ceph_business_flag": "TEG_AILAB_TRC_CGC",
        "GPUName": "P40",
        "model_local_file_path": "",
        "init_cmd": "taiji_client mount -l sz -tk MeTKgi2s6aolfGLaIVz6iA /mnt/private_xianghao",
        "host_num": 1,
        "host_gpu_num": 8,
        "keep_alive": True,
        "cuda_version": "10.2",
        "is_elasticity": False,
        "is_store_core_file": False,  # 是否存储core文件
        "image_full_name": "mirrors.tencent.com/xianghao/tlinux2.2-cuda10.2-py3.10-pytorch1.12",
        "permission": {
            "admin_group": "xianghao",
            "view_group": "",
            "alert_group": "xianghao",
        },
    }

    # fsb/xxx.toml
    task_flag, data_name, model_name, config_name = config_path_to_task_flag(
        args.config
    )
    config["task_flag"] = task_flag
    config["host_gpu_num"] = args.gpu_num

    # Create a temporary directory and write config.json and start.sh
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Created temporary directory: {tmp_dir}")
        config["model_local_file_path"] = tmp_dir
        start_sh_path = os.path.join(tmp_dir, "start.sh")
        config_json_path = os.path.join(tmp_dir, "config.json")

        # Create "start.sh" in this tmp dir
        with open(start_sh_path, "w") as f:
            if args.ghost_run:
                content = f"#!/bin/bash \n" f"sleep 2d"
            else:
                content = (
                    f"#!/bin/bash \n"
                    f"cd /mnt/private_xianghao/proj/audiozen \n"
                    f"pip install -e . -i https://mirrors.tencent.com/pypi/simple/ \n"
                    f"cd /mnt/private_xianghao/proj/audiozen/recipes/{data_name} \n"
                    f"torchrun --nnodes=1 --nproc_per_node={args.gpu_num} run.py -C {model_name}/{config_name}.toml -M train -R"
                )

            f.write(content)
            print(f"=" * 80)
            print(content)

        # Create "config.json" in this tmp dir
        with open(config_json_path, "w") as f:
            json.dump(config, f, indent=4)

            print(f"=" * 80)
            print(json.dumps(config, indent=4))

        print(f"=" * 80)
        print(f"taiji_client start -scfg {config_json_path}")

        if args.run:
            os.system(f"taiji_client start -scfg {config_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-C",
        "--config",
        type=str,
        default="dns_icassp_2020/bsrnn/baseline_bsrnn.toml",
    )
    parser.add_argument(
        "-N",
        "--gpu_num",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the task, otherwise just print the command",
    )
    parser.add_argument(
        "-G",
        "--ghost_run",
        action="store_true",
        help="ghost run, just apply for gpu resource and sleep 2 days",
    )
    args = parser.parse_args()
    main(args)
