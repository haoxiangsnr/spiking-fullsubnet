import argparse
from pathlib import Path

import toml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from audiozen.logger import init_logging_logger
from audiozen.utils import instantiate


def run(config, resume):
    init_logging_logger(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model_g = instantiate(config["model_g"]["path"], args=config["model_g"]["args"])
    model_d = instantiate(config["model_d"]["path"], args=config["model_d"]["args"])

    optimizer_g = instantiate(
        config["optimizer_g"]["path"],
        args={"params": model_g.parameters()} | config["optimizer_g"]["args"],
    )

    optimizer_d = instantiate(
        config["optimizer_d"]["path"],
        args={"params": model_d.parameters()} | config["optimizer_d"]["args"],
    )

    loss_function = instantiate(
        config["loss_function"]["path"],
        args=config["loss_function"]["args"],
    )

    scheduler_g = instantiate(
        config["lr_scheduler_g"]["path"],
        args={"optimizer": optimizer_g} | config["lr_scheduler_g"]["args"],
    )

    scheduler_d = instantiate(
        config["lr_scheduler_d"]["path"],
        args={"optimizer": optimizer_d} | config["lr_scheduler_d"]["args"],
    )

    (
        model_g,
        optimizer_g,
        scheduler_g,
        model_d,
        optimizer_d,
        scheduler_d,
    ) = accelerator.prepare(
        model_g,
        optimizer_g,
        scheduler_g,
        model_d,
        optimizer_d,
        scheduler_d,
    )

    if "train" in args.mode:
        train_dataset = instantiate(
            config["train_dataset"]["path"],
            args=config["train_dataset"]["args"],
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            collate_fn=None,
            shuffle=True,
            **config["train_dataset"]["dataloader"],
        )

        train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        if not isinstance(config["validate_dataset"], list):
            config["validate_dataset"] = [config["validate_dataset"]]

        validate_dataloaders = []
        for validate_config in config["validate_dataset"]:
            validate_dataset = instantiate(validate_config["path"], args=validate_config["args"])

            validate_dataloaders.append(
                accelerator.prepare(
                    DataLoader(
                        dataset=validate_dataset,
                        **validate_config["dataloader"],
                    )
                )
            )

    if "test" in args.mode:
        if not isinstance(config["test_dataset"], list):
            config["test_dataset"] = [config["test_dataset"]]

        test_dataloaders = []
        for test_config in config["test_dataset"]:
            test_dataset = instantiate(test_config["path"], args=test_config["args"])

            test_dataloaders.append(
                accelerator.prepare(
                    DataLoader(
                        dataset=test_dataset,
                        **test_config["dataloader"],
                    )
                )
            )

    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model_g=model_g,
        optimizer_g=optimizer_g,
        lr_scheduler_g=scheduler_g,
        model_d=model_d,
        optimizer_d=optimizer_d,
        lr_scheduler_d=scheduler_d,
        loss_function=loss_function,
    )

    for flag in args.mode:
        if flag == "train":
            trainer.train(train_dataloader, validate_dataloaders)
        elif flag == "validate":
            trainer.validate(validate_dataloaders)
        elif flag == "test":
            trainer.test(test_dataloaders, config["meta"]["ckpt_path"])
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
            raise ValueError("checkpoint path is required for test. Use '--ckpt_path'.")
        else:
            config["meta"]["ckpt_path"] = args.ckpt_path

    run(config, args.resume)
