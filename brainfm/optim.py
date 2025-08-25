import torch
import torch.optim as optim
from brainfm.utils import Config

def build_optimizer(model: torch.nn.Module, config: Config, logger=None):
    opt_cfg = config.optimizer
    optimizer_type = opt_cfg.type.lower()

    # TODO: Implement layer-wise learning rate decay in the future.
    params = model.parameters()

    if logger:
        logger.info(f"Building optimizer: {opt_cfg.type}")

    if optimizer_type == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = optim.Adam(
            params,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            params,
            lr=opt_cfg.lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_cfg.type}")

    return optimizer
