# FAQ

## `torch.distributed.barrier()` hangs in DDP

Use `model.module` instead of `model` during validation.

Check https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522/6 for more details.
