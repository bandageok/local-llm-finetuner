"""评估入口 — python evaluate.py [model_path] [--full]"""
import sys
import os
import json
import logging
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluator import evaluate_training, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_data(config: dict) -> list:
    """加载测试数据（从 data_dir 中取 val_split 作为测试集）"""
    from src.data_loader import DatasetLoader
    loader = DatasetLoader(config)
    train_data, val_data, fmt = loader.load()
    return val_data if val_data else train_data[-max(1, int(len(train_data) * 0.1)):]


def main():
    config = load_config()

    # 解析参数
    args = sys.argv[1:]
    full_eval = "--full" in args
    model_path = None
    for a in args:
        if not a.startswith("--"):
            model_path = a
            break

    if not model_path:
        model_path = os.path.join(config["output"]["output_dir"], "final_model")

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Usage: python evaluate.py [model_path] [--full]")
        return

    if full_eval:
        # 综合评估：BERTScore + Rouge + 可视化
        logger.info(f"Full evaluation: {model_path}")
        test_data = load_test_data(config)
        if not test_data:
            logger.error("No test data found")
            return
        results = evaluate_model(
            model_path=model_path,
            test_data=test_data,
            output_dir=config["output"]["output_dir"],
            lang=config.get("evaluation", {}).get("lang", "zh"),
        )
        logger.info(f"\n{'='*50}")
        logger.info(f"Eval Loss:    {results.get('eval_loss', 'N/A')}")
        logger.info(f"BERTScore F1: {results.get('bertscore_f1', 'N/A')}")
        logger.info(f"Rouge-L:      {results.get('rouge-l', 'N/A')}")
        logger.info(f"Report:       {results.get('report_chart', 'N/A')}")
    else:
        # 原有评估流程（向后兼容）
        logger.info(f"Evaluating model: {model_path}")
        results = evaluate_training(config, model_path)
        if results["plot_path"]:
            logger.info(f"Report: {results['plot_path']}")
        if results["samples"]:
            for s in results["samples"]:
                logger.info(f"\n[Prompt] {s['prompt']}\n[Response] {s['response'][:200]}")


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
