# Trainer

For each experiment, we need to define a custom trainer to train the model. The custom trainer must inherit from `audiozen.trainer.base_trainer.BaseTrainer` and implement the following methods:

- `training_step`: The training step. It contains the operations to be executed in each training iteration.
- `training_epoch_end`: The training epoch end. It contains the operations to be executed at the end of each training epoch.
- `validation_step`: The validation step. It contains the operations to be executed in each validation iteration.
- `validation_epoch_end`: The validation epoch end. It contains the operations to be executed at the end of each validation epoch.
- `test_step`: The test step. It contains the operations to be executed in each test iteration.
- `test_epoch_end`: The test epoch end. It contains the operations to be executed at the end of each test epoch.

```{eval-rst}
.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.training_step

.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.training_epoch_end

.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.validation_step

.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.validation_epoch_end

.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.test_step

.. autofunction:: audiozen.trainer.base_trainer.BaseTrainer.test_epoch_end
```