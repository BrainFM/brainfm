import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from brainfm.utils import Config

def build_scheduler(optimizer: optim.Optimizer, config: Config, logger=None):
    sched_cfg = config.scheduler
    scheduler_type = sched_cfg.type.lower()

    if hasattr(sched_cfg, "max_epochs"):
        max_epochs = sched_cfg.max_epochs
    else:
        raise ValueError("Configuration must contain max_epochs for scheduler setup (e.g., under config.scheduler.max_epochs).")

    if scheduler_type == "warmupcosine":
        warmup_epochs = sched_cfg.warmup_epochs
        min_lr = sched_cfg.min_lr
        start_factor = getattr(sched_cfg, 'start_factor', 0.001)
        end_factor = getattr(sched_cfg, 'end_factor', 1.0)

        if warmup_epochs >= max_epochs:
            raise ValueError(
                f"Warmup epochs ({warmup_epochs}) must be less than max epochs ({max_epochs})"
            )
        if warmup_epochs < 0:
             raise ValueError("Warmup epochs must be non-negative.")

        if warmup_epochs > 0:
            # Linear warmup phase
            scheduler_warmup = LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=warmup_epochs
            )
            # Cosine decay phase
            main_epochs = max_epochs - warmup_epochs
            scheduler_decay = CosineAnnealingLR(
                optimizer,
                T_max=main_epochs, # Number of iterations for the cosine decay part
                eta_min=min_lr     # Minimum learning rate
            )
            # Chain them together
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_decay],
                milestones=[warmup_epochs] # Epoch index to switch schedulers
            )
            if logger:
                logger.info("Using learning rate scheduler 'WarmupCosineLR'")
        else:
            # No warmup requested, just use CosineAnnealingLR for the whole duration
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=min_lr
            )
            if logger:
                logger.info("Warmup Epochs set to 0, learning rate scheduler 'CosineAnnealingLR'")

    elif scheduler_type == "cosine":
        # Cosine annealing without explicit warmup config
        min_lr = sched_cfg.min_lr
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=min_lr
        )
        if logger:
            logger.info("Using learning rate scheduler 'CosineAnnealingLR'")

    elif scheduler_type == "step":
        step_size = sched_cfg.step_size
        gamma = sched_cfg.gamma
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        if logger:
             logger.info("Using learning rate scheduler 'step'")

    else:
        raise ValueError(
            f"Unsupported scheduler type: {sched_cfg.type}. Supported types are: warmupcosine, cosine, step."
        )
    
    return scheduler