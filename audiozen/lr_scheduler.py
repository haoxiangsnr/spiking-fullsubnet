from torch.optim import lr_scheduler

step_lr_scheduler = lr_scheduler.StepLR
reduce_lr_on_plateau_scheduler = lr_scheduler.ReduceLROnPlateau
constant_lr_scheduler = lr_scheduler.ConstantLR
