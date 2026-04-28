"""部署与打包 — Ollama 部署 + 模型打包分享"""
import os
import json
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def deploy_to_ollama(modelfile_path: str, model_name: str = "finetuned") -> bool:
    """部署模型到 Ollama"""
    # 检查 Ollama 是否安装
    if not _check_ollama_installed():
        logger.error("Ollama not installed. Install from https://ollama.ai")
        return False

    # 检查 Ollama 是否运行
    if not _check_ollama_running():
        logger.warning("Ollama is not running. Attempting to start...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

    # 创建模型
    logger.info(f"Creating Ollama model: {model_name}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode == 0:
        logger.info(f"Model deployed to Ollama as '{model_name}'")
        logger.info(f"Run: ollama run {model_name}")
        return True
    else:
        logger.error(f"Ollama deploy failed: {result.stderr}")
        return False


def _check_ollama_installed() -> bool:
    """检查 Ollama 是否已安装"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_ollama_running() -> bool:
    """检查 Ollama 是否在运行"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def package_model(model_dir: str, output_dir: str,
                  meta: Optional[dict] = None,
                  name: str = None) -> str:
    """打包模型为 zip 方便分享"""
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"model_{timestamp}"

    zip_path = os.path.join(output_dir, f"{name}.zip")
    logger.info(f"Packaging model to {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        model_path = Path(model_dir)
        for file in model_path.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(model_path)
                zf.write(file, arcname)

        # 添加元信息
        if meta:
            zf.writestr("README.json", json.dumps(meta, indent=2, ensure_ascii=False))

        # 添加部署说明
        readme = _generate_readme(meta)
        zf.writestr("README.md", readme)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    logger.info(f"Package created: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def _generate_readme(meta: Optional[dict] = None) -> str:
    """生成部署说明"""
    lines = [
        "# Fine-tuned Model Package",
        "",
        "## 使用方法",
        "",
        "### 方法1: HuggingFace Transformers",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        'model = AutoModelForCausalLM.from_pretrained("./hf_export", trust_remote_code=True)',
        'tokenizer = AutoTokenizer.from_pretrained("./hf_export", trust_remote_code=True)',
        "```",
        "",
        "### 方法2: Ollama 部署",
        "```bash",
        "# 1. 安装 Ollama (https://ollama.ai)",
        "# 2. 运行:",
        "ollama create mymodel -f ./gguf_export/Modelfile",
        "ollama run mymodel",
        "```",
        "",
        "### 方法3: llama.cpp",
        "```bash",
        "git clone https://github.com/ggerganov/llama.cpp",
        "cd llama.cpp && make",
        "./llama-cli -m ./gguf_export/model-*.gguf -p '你的问题'",
        "```",
        "",
    ]

    if meta:
        lines.extend([
            "## 模型信息",
            "",
            f"- 基座模型: {meta.get('base_model', 'N/A')}",
            f"- 任务类型: {meta.get('task_type', 'N/A')}",
            f"- LoRA Rank: {meta.get('qlora_r', 'N/A')}",
            f"- 最大序列长度: {meta.get('max_seq_length', 'N/A')}",
            f"- 训练轮数: {meta.get('num_epochs', 'N/A')}",
            f"- 数据集格式: {meta.get('dataset_format', 'N/A')}",
            "",
        ])

    return "\n".join(lines)


def deploy(config: dict, model_path: str) -> dict:
    """完整部署流程"""
    output_dir = config["output"]["output_dir"]

    # 读取元信息
    meta_path = os.path.join(model_path, "training_meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    results = {}

    # 打包模型
    logger.info("Step 1: Packaging model for sharing...")
    zip_path = package_model(model_path, output_dir, meta)
    results["package"] = zip_path

    # 尝试 Ollama 部署
    modelfile_path = None
    for gguf_dir in Path(output_dir).rglob("Modelfile"):
        modelfile_path = str(gguf_dir)
        break

    if modelfile_path:
        logger.info("Step 2: Deploying to Ollama...")
        model_name = meta.get("description", "finetuned").replace(" ", "-").lower()[:32]
        if deploy_to_ollama(modelfile_path, model_name):
            results["ollama"] = model_name
    else:
        logger.info("No Modelfile found, skipping Ollama deployment")

    return results
