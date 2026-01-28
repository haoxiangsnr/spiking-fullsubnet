from dataloader import DatasetArgs, DNSAudio
from model import ModelArgs, SpikingFullSubNet
from simple_parsing import ArgumentParser
from trainer import Trainer

from audiozen.logger import init_logging_logger
from audiozen.trainer_args import TrainingArgs


def run(args):
    # Extract arguments
    trainer_args: TrainingArgs = args.trainer
    train_dataset_args: DatasetArgs = args.train_dataset
    eval_dataset_args: DatasetArgs = args.eval_dataset
    model_args: ModelArgs = args.model

    # Initialize logger
    init_logging_logger(args.trainer.output_dir)

    # Initialize model, datasets
    model = SpikingFullSubNet(model_args)
    train_dataset = DNSAudio(train_dataset_args)
    eval_dataset = DNSAudio(eval_dataset_args)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if trainer_args.do_eval:
        trainer.evaluate()
    elif trainer_args.do_predict:
        trainer.predict()
    elif trainer_args.do_train:
        trainer.train()
    else:
        raise ValueError("At least one of do_train, do_eval, or do_predict must be True.")


if __name__ == "__main__":
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(TrainingArgs, dest="trainer")
    parser.add_arguments(ModelArgs, dest="model")
    parser.add_arguments(DatasetArgs, dest="train_dataset")
    parser.add_arguments(DatasetArgs, dest="eval_dataset")
    args = parser.parse_args()
    run(args)
