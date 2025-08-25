import argparse

from brainfm.data import build_loader
from brainfm.models import build_model
from brainfm.optim import build_optimizer
from brainfm.scheduler import build_scheduler
# from brainfm.trainer import train
from brainfm.utils  import (
    set_seed,
    get_config,
    get_logger,
    get_device
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainFM"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to configuration file. Supports .yaml, .yml, or .json formats.",
    )
    parser.add_argument(
        "--experiment_name",        
        type=str,
        default="pretrain_brainfm",
        help="Name of the experiment for logger.",
    )
    return parser.parse_args()

def main() -> None:
    args   = parse_arguments()
    config = get_config(path=args.cfg)
    logger = get_logger(
        log_dir=config.paths.log_dir,
        experiment_name=args.experiment_name
    )
    device = get_device(device_str=config.train.device)
    logger.info(f"Using device: {device}")

    set_seed(config.train.seed)
    logger.info(f"Set random seed to: {config.train.seed}")

    dataloader = build_loader(
        config=config,
        logger=logger
    )
    logger.info(f"Built DataLoader with {len(dataloader)} batches.")
    
    model = build_model(
        config=config,
        device=device,
        logger=logger
    )
    logger.info("Built model:")
    logger.info(model)

    optimizer = build_optimizer(
        model=model,
        config=config,
        logger=logger,
    )

    lr_scheduler = build_scheduler(
        optimizer=optimizer,
        config=config,
        logger=logger
    )

    # train(
    #     model=model,
    #     dataloader=dataloader,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     config=config,
    #     logger=logger,
    #     device=deivce,
    # )


if __name__ == "__main__":
    main()