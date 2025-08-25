import argparse
from brainfm.utils  import set_seed, load_config, get_logger
from brainfm.data import build_loader
from brainfm.models import build_model
from brainfm.optim import build_optimizer
from brainfm.scheduler import build_scheduler
from brainfm.trainer import train


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train the BrainFM model"
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Flag to use GPU for training.",
        dest="use_gpu"
    )
    parser.add_argument(
        "--experiment_name",        
        type=str,
        default="pretrain_modis",
        help="Name of the experiment for logger.",
    )

    return parser.parse_args()


def main() -> None:
    print("Pretraining script is not yet implemented.")
    # args   = parse_arguments()
    # config = load_config(path=args.cfg)
    # # Update config with parsed args
    # config.USE_GPU = args.use_gpu

    # logger = get_logger(
    #     log_dir=config.DIR.LOG,
    #     experiment_name=args.experiment_name
    # )
    # set_seed(getattr(config, "SEED", 42))

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