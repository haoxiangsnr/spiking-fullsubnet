import logging
import os

logger = logging.getLogger(__name__)


def hp_search_ray(trainer, n_trails, direction, **kwargs):
    import ray

    def _objective(trial, local_trainer, checkpoint_dir=None):
        checkpoint = None

        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith("checkpoint"):
                    checkpoint = os.path.join(checkpoint_dir, subdir)

        local_trainer.objective = None
        local_trainer.train(resume_from_checkpoint=checkpoint, trail=trial)

        if getattr(local_trainer, "objective", None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            local_trainer.tune_save_checkpoint()
            ray.tune.report(objective=local_trainer.objective, **metrics, done=True)

    if "resources_per_trial" not in kwargs:
        kwargs["resources_per_trial"] = {"cpu": 1}

        if trainer.args.gpus > 0:
            kwargs["resources_per_trial"]["gpu"] = 1

        resource_msg = "1 CPU" + (" and 1 GPU" if trainer.args.n_gpu > 0 else "")

        logger.info(f"No `resources_per_trial` specified.")
        logger.info(f"Using {resource_msg} per trial by default.")

        gpus_per_trial = kwargs["resources_per_trial"].get("gpu", 0)
        trainer.args.n_gpus = gpus_per_trial

        trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

        analysis = ray.tune.run(
            trainable,
            config=trainer.hp_space(None),
            num_samples=n_trails,
            **kwargs,
        )

        best_trial = analysis.get_best_trial("objective", direction)

        return best_trial.config, best_trial.last_result["objective"]
