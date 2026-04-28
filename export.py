"""导出入口 — python export.py [model_path] [--format hf|gguf|both]"""
import sys
import os
import logging
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.exporter import export_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # 命令行参数
    model_path = None
    export_format = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--format" and i + 1 < len(args):
            export_format = args[i + 1]
        elif not arg.startswith("--") and arg != "--format":
            model_path = arg

    if not model_path:
        model_path = os.path.join(config["output"]["output_dir"], "final_model")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Usage: python export.py [model_path] [--format hf|gguf|both]")
        return

    # 覆盖格式
    if export_format:
        config["export"]["format"] = export_format

    logger.info(f"Exporting model: {model_path}")
    logger.info(f"Format: {config['export']['format']}")
    logger.info(f"GGUF quant: {config['export']['gguf_quant']}")

    results = export_model(config, model_path)

    logger.info("\nExport results:")
    for fmt, path in results.items():
        logger.info(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
