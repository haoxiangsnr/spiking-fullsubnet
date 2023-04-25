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

    validation_dataset_dns_1_with_reverb = instantiate(
        config["validation_dataset_dns_1_with_reverb"]["path"],
        args=config["validation_dataset_dns_1_with_reverb"]["args"],
    )
    validation_dataset_dns_1_with_reverb = DataLoader(
        dataset=validation_dataset_dns_1_with_reverb,
        num_workers=0,
        batch_size=1,
    )

    validation_dataset_dns_1_no_reverb = instantiate(
        config["validation_dataset_dns_1_no_reverb"]["path"],
        args=config["validation_dataset_dns_1_no_reverb"]["args"],
    )
    validation_dataset_dns_1_no_reverb = DataLoader(
        dataset=validation_dataset_dns_1_no_reverb,
        num_workers=0,
        batch_size=1,
    )

    test_dataset_dns_1_with_reverb = instantiate(
        config["test_dataset_dns_1_with_reverb"]["path"],
        args=config["test_dataset_dns_1_with_reverb"]["args"],
    )
    test_dataset_dns_1_with_reverb = DataLoader(
        dataset=test_dataset_dns_1_with_reverb,
        num_workers=0,
        batch_size=1,
    )

    test_dataset_dns_1_no_reverb = instantiate(
        config["test_dataset_dns_1_no_reverb"]["path"],
        args=config["test_dataset_dns_1_no_reverb"]["args"],
    )
    test_dataset_dns_1_no_reverb = DataLoader(
        dataset=test_dataset_dns_1_no_reverb,
        num_workers=0,
        batch_size=1,
    )

    model = instantiate(
        config["model"]["path"],
        args=config["model"]["args"],
    )

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
                [
                    validation_dataset_dns_1_with_reverb,
                    validation_dataset_dns_1_no_reverb,
                ],
            )
        elif flag == "validate":
            trainer.validate(
                [
                    validation_dataset_dns_1_with_reverb,
                    validation_dataset_dns_1_no_reverb,
                ]
            )
        elif flag == "test":
            trainer.test(
                [
                    validation_dataset_dns_1_with_reverb,
                    validation_dataset_dns_1_no_reverb,
                ],
                config["meta"]["ckpt_path"],
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

    # e.g., add sys.path to "model.Model"
    sys.path.append(config_path.parent.as_posix())
    sys.path.append(config_path.parent.parent.as_posix())

    run(config, args.resume)
