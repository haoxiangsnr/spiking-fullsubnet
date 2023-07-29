- run.py
  - generative adversarial network (GAN)
  - ddp-supported training, validation, and testing

- trainer.py
  - multiple training loss functions
  - batch-metric calculation

- dataloader
  - dataloader.py
  - dataloader_v2.py
    - dataloader_v2.py is a more efficient version of dataloader.py
    - use numpy instead of dictionary to store data