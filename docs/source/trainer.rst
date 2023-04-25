================
Trainer
================

Training
========

Here is the persuade code for training a model.

.. code-block:: python
    :caption: train
    :linenos:
    :emphasize-lines: 9,16

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

Validate
========

Here is the persuade code for validating a model.

.. code-block:: python
    :caption: validate
    :linenos:

    for batch, batch_index in dataloader:
        loss = validation_step(batch, batch_idx)

        validation_epoch_output.append(loss)

    validation_epoch_end(validation_epoch_output)

    return score

Test
====

.. code-block:: python
    :caption: test
    :linenos:

    load_checkpoint(ckpt_path)

    for batch, batch_index in dataloader:
        loss = test_step(batch, batch_idx)

        test_epoch_output.append(loss)

    test_epoch_end(test_epoch_output)

    return score


API of Trainer
==============

.. automodule:: audiozen.trainer.base_trainer
   :members:
   :undoc-members: