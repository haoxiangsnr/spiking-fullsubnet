# Trainer

For each experiment, we need to define a custom trainer to train the model. The custom trainer must inherit from `audiozen.common_trainer.Trainer` and implement the following methods:

- `training_step`: The training step. It contains the operations to be executed in each training iteration.
- `training_epoch_end`: The training epoch end. It contains the operations to be executed at the end of each training epoch.
- `validation_step`: The validation step. It contains the operations to be executed in each validation iteration.
- `validation_epoch_end`: The validation epoch end. It contains the operations to be executed at the end of each validation epoch.
- `test_step` (optional): The test step. It contains the operations to be executed in each test iteration.
- `test_epoch_end` (optional): The test epoch end. It contains the operations to be executed at the end of each test epoch.

```{eval-rst}
.. autofunction:: audiozen.common_trainer.Trainer.training_step

.. autofunction:: audiozen.common_trainer.Trainer.training_epoch_end

.. autofunction:: audiozen.common_trainer.Trainer.validation_step

.. autofunction:: audiozen.common_trainer.Trainer.validation_epoch_end

.. autofunction:: audiozen.common_trainer.Trainer.test_step

.. autofunction:: audiozen.common_trainer.Trainer.test_epoch_end
```

## APIs

```{eval-rst}
.. autofunction:: audiozen.common_trainer.Trainer.train