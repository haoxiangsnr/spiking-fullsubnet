from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_lr(self):
        pass
