import argparse
import logging

from configs import Configs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motivation front end time")
    parser.add_argument(
        "--model",
        type=str,
        choices=Configs.ALL_MODELS,
        default=Configs.ALL_MODELS[0],
        help="Model name")
    parser.add_argument(
        "--target",
        type=str,
        choices=["expr", "plot"],
        required=True,
        help="Target to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arg()
    configs = Configs(
        model=args.model
    )
    logger.info(f"Configs: {configs}")
    if args.target == "expr":
        from expr import run
        run(configs)
    elif args.target == "plot":
        from plot import run
        run(configs)
    else:
        raise ValueError(f"Unknown target: {args.target}")