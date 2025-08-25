import argparse

from brainfm.data import build_loader
from brainfm.models import build_model
from brainfm.optim import build_optimizer
from brainfm.scheduler import build_scheduler
from brainfm.trainer import train
from brainfm.utils  import (
    set_seed,
    load_config,
    get_logger,
    validate_config_path
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BrainFM"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        help="Path to configuration file. Supports .yaml, .yml, or .json formats.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (e.g., 'cpu', 'cuda:0').",
        dest="device"
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

    validate_config_path(args.cfg)
    config = load_config(path=args.cfg)

    

    # logger = get_logger(
    #     log_dir=config.path.log_dir,
    #     experiment_name=args.experiment_name
    # )

    # set_seed(config.seed)

    # dataloader = build_loader(
    #     config=config,
    #     logger=logger
    # )
    # model = build_model(
    #     config=config,
    #     logger=logger
    # )
    # optimizer = build_optimizer(
    #     model=model,
    #     config=config,
    #     logger=logger
    # )
    # lr_scheduler = build_scheduler(
    #     optimizer=optimizer,
    #     config=config,
    #     logger=logger
    # )

    # train(
    #     model=model,
    #     dataloader=dataloader,
    #     optimizer=optimizer,
    #     lr_scheduler=lr_scheduler,
    #     config=config,
    #     logger=logger,
    # )


if __name__ == "__main__":
    main()