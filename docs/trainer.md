# Trainer

For each experiment, we need to define a trainer to train the model. The custom trainer must inherit from `audiozen.trainer.base_trainer.BaseTrainer` and implement the following methods:

- `training_step`: The training step. It contains the operations to be executed in each training iteration.
- `training_epoch_end`: The training epoch end. It contains the operations to be executed at the end of each training epoch.
- `validation_step`: The validation step. It contains the operations to be executed in each validation iteration.
- `validation_epoch_end`: The validation epoch end. It contains the operations to be executed at the end of each validation epoch.
- `test_step`: The test step. It contains the operations to be executed in each test iteration.
- `test_epoch_end`: The test epoch end. It contains the operations to be executed at the end of each test epoch.

## Training

Here is the persuade code for training a model.

```python hl_lines="9 16"
for epoch in range(start_epoch, end_epoch):
    self.model.train()

    training_epoch_output = []

    for batch, batch_index in dataloader:
        zero_grad()

        loss = training_step(batch, batch_idx)

        loss.backward()
        optimizer.step()

    training_epoch_output.append(loss)

    training_epoch_end(training_epoch_output)

    save_checkpoint()

    if some_condition:
        score = validate()

        if score > best_score:
            save_checkpoint(best=True)
```

## Validate

Here is the persuade code for validating a model.

```python
for batch, batch_index in dataloader:
    loss = validation_step(batch, batch_idx)

    validation_epoch_output.append(loss)

validation_epoch_end(validation_epoch_output)

return score
```

## Test


```python
load_checkpoint(ckpt_path)

for batch, batch_index in dataloader:
    loss = test_step(batch, batch_idx)

    test_epoch_output.append(loss)

test_epoch_end(test_epoch_output)

return score
```

::: audiozen.trainer.base_trainer.BaseTrainer
