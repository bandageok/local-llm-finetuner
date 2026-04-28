"""训练入口 — python train.py"""
import sys
import os
import logging
import yaml

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gpu_detector import print_gpu_summary
from src.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("train")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    logger.info("=" * 60)
    logger.info("  LLM QLoRA Fine-Tuning")
    logger.info("=" * 60)

    # 加载配置
    config = load_config()
    logger.info(f"Model: {config['model']['name_or_path']}")
    logger.info(f"Task: {config['direction']['task_type']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")

    # GPU 检测
    gpu_info = print_gpu_summary()
    gpu_optimal = gpu_info.get_optimal_config()

    if not gpu_info.available:
        logger.error("No GPU detected. Training on CPU is not recommended.")
        resp = input("Continue on CPU? (y/N): ").strip().lower()
        if resp != "y":
            return

    # 显存安全检查
    if not gpu_info.check_memory_safe():
        logger.error("Insufficient VRAM. Consider a smaller model or QLoRA with rank=4.")
        resp = input("Continue anyway? (y/N): ").strip().lower()
        if resp != "y":
            return

    # 训练
    final_model_dir = train(config, gpu_optimal)

    logger.info("=" * 60)
    logger.info(f"  Training Complete!")
    logger.info(f"  Model saved: {final_model_dir}")
    logger.info("=" * 60)

    # 自动评估
    resp = input("\nRun evaluation? (Y/n): ").strip().lower()
    if resp != "n":
        from src.evaluator import evaluate_training
        evaluate_training(config, final_model_dir)

    # 自动导出
    resp = input("\nExport model? (Y/n): ").strip().lower()
    if resp != "n":
        from src.exporter import export_model
        export_model(config, final_model_dir)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    main()
