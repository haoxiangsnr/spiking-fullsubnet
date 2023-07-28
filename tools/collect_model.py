import shutil
from pathlib import Path

import click


@click.command()
@click.argument("dataset", type=str)
@click.argument("model_name", type=str)
@click.argument("exp_id", type=str)
@click.option("--ckpt_id", type=str, default="best")
@click.option("--model_zoo_path", type=str, default="model_zoo")
def main(dataset, model_name, exp_id, ckpt_id, model_zoo_path):
    exp_dir = Path("recipes", dataset, model_name) / "exp" / exp_id

    # prepare checkpoint with ckpt_id
    ckpt_path = exp_dir / "checkpoints" / ckpt_id
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist."

    # prepare tb_log
    tb_log_path = exp_dir / "tb_log"
    assert tb_log_path.exists(), f"Tensorboard log {tb_log_path} does not exist."

    # prepare log
    log_path = exp_dir / f"{exp_id}.log"
    assert log_path.exists(), f"Log {log_path} does not exist."

    # prepare config file by sorting the timestamps
    config_files = sorted(exp_dir.glob("*.toml"))
    assert len(config_files) > 0, f"No config file found in {exp_dir}."
    latest_config_file = config_files[-1]

    # prepare model zoo
    model_zoo_dir = Path(model_zoo_path)
    assert model_zoo_dir.exists(), f"Model zoo {model_zoo_dir} does not exist."
    model_zoo_exp_save_path = model_zoo_dir / dataset / model_name / exp_id
    model_zoom_ckpt_save_path = model_zoo_exp_save_path / "checkpoints"
    model_zoom_ckpt_save_path.mkdir(parents=True, exist_ok=True)

    # copy to model_zoo
    shutil.copytree(ckpt_path, model_zoom_ckpt_save_path / ckpt_id)
    shutil.copytree(tb_log_path, model_zoo_exp_save_path / "tb_log")

    # copy log file to model_zoo
    shutil.copy(log_path, model_zoo_exp_save_path / f"{exp_id}.log")

    # copy config file to model_zoo
    shutil.copy(latest_config_file, model_zoo_exp_save_path / f"{exp_id}.toml")


if __name__ == "__main__":
    main()
