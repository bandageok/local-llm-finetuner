"""部署入口 — python deploy.py [model_path] [--name model_name]"""
import sys
import os
import logging
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.deployer import deploy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deploy")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # 命令行参数
    model_path = None
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if not arg.startswith("--"):
            model_path = arg
            break

    if not model_path:
        model_path = os.path.join(config["output"]["output_dir"], "final_model")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Usage: python deploy.py [model_path]")
        return

    logger.info(f"Deploying model: {model_path}")
    results = deploy(config, model_path)

    logger.info("\nDeployment results:")
    if "package" in results:
        logger.info(f"  Package: {results['package']}")
    if "ollama" in results:
        logger.info(f"  Ollama: ollama run {results['ollama']}")


if __name__ == "__main__":
    main()
