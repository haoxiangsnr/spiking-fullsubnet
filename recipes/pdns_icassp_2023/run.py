import argparse
import os
import sys
from pathlib import Path

import toml
import torch
from torch.utils.data import DataLoader, DistributedSampler

import audiozen.loss as loss
from audiozen.logger import init_logging_logger
from audiozen.utils import initialize_ddp, instantiate, set_random_seed


def run(config, resume):
    set_random_seed(config["meta"]["seed"])

    rank = int(os.environ["LOCAL_RANK"])
    initialize_ddp(rank)

    if rank == 0:
        init_logging_logger(config)

    # ====================================================
    # Training dataset
    # ====================================================
    train_dataset = instantiate(
        config["train_dataset"]["path"],
        args=config["train_dataset"]["args"],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=DistributedSampler(dataset=train_dataset, rank=rank, shuffle=True),
        collate_fn=None,
        shuffle=False,
        **config["train_dataset"]["dataloader"],
    )

    # ====================================================
    # Validation datasets
    # ====================================================
    valid_dataset_dns5_track1 = instantiate(
        config["validation_dataset_dns5_track1"]["path"],
        args=config["validation_dataset_dns5_track1"]["args"],
    )
    valid_dataloader_dns5_track1 = DataLoader(
        dataset=valid_dataset_dns5_track1,
        num_workers=0,
        batch_size=1,
    )

    valid_dataset_dns5_track2 = instantiate(
        config["validation_dataset_dns5_track2"]["path"],
        args=config["validation_dataset_dns5_track2"]["args"],
    )
    valid_dataloader_dns5_track2 = DataLoader(
        dataset=valid_dataset_dns5_track2,
        num_workers=0,
        batch_size=1,
    )

    # ====================================================
    # Test datasets
    # ====================================================
    test_dataset_dns5_track1 = instantiate(
        config["test_dataset_dns5_track1"]["path"],
        args=config["test_dataset_dns5_track1"]["args"],
    )
    test_dataloader_dns5_track1 = DataLoader(
        dataset=test_dataset_dns5_track1,
        num_workers=0,
        batch_size=1,
    )

    test_dataset_dns5_track2 = instantiate(
        config["test_dataset_dns5_track2"]["path"],
        args=config["test_dataset_dns5_track2"]["args"],
    )
    test_dataloader_dns5_track2 = DataLoader(
        dataset=test_dataset_dns5_track2,
        num_workers=0,
        batch_size=1,
    )

    # ====================================================
    # Model, Optimizer, loss function, and trainer
    # ====================================================
    model = instantiate(config["model"]["path"], args=config["model"]["args"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
    )

    loss_function = getattr(loss, config["loss_function"]["name"])(
        **config["loss_function"]["args"],
    )

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        rank=rank,
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    for flag in args.mode:
        if flag == "train":
            trainer.train(
                train_dataloader,
                [valid_dataloader_dns5_track1, valid_dataloader_dns5_track2],
            )
        elif flag == "validate":
            trainer.validate(
                [valid_dataloader_dns5_track1, valid_dataloader_dns5_track2],
            )
        elif flag == "test":
            trainer.test(
                [test_dataloader_dns5_track1, test_dataloader_dns5_track2],
                ckpt_path=config["meta"]["ckpt_path"],
            )
        elif flag == "predict":
            raise NotImplementedError("Predict is not implemented yet.")
        elif flag == "finetune":
            raise NotImplementedError("Finetune is not implemented yet.")
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate", "test", "predict", "finetune"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for test. It can be 'best', 'latest', or a path to a checkpoint.",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    if "test" in args.mode:
        if args.ckpt_path is None:
            raise ValueError(
                "Checkpoint path is required for test. Check '--ckpt_path'."
            )
        else:
            config["meta"]["ckpt_path"] = args.ckpt_path

    # "dataset_*.py" and "run.py" within specific model folder are used first
    sys.path.insert(0, config_path.parent.parent.as_posix())
    sys.path.insert(0, config_path.parent.as_posix())

    run(config, args.resume)