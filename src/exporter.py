"""模型导出 — HF safetensors + GGUF 格式"""
import os
import json
import logging
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def export_hf(model_path: str, tokenizer_path: str, output_dir: str,
              meta: Optional[dict] = None) -> str:
    """导出为 HuggingFace safetensors 格式"""
    from peft import PeftModel, AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info(f"Exporting HF model from {model_path}")

    export_dir = os.path.join(output_dir, "hf_export")
    os.makedirs(export_dir, exist_ok=True)

    # 加载并合并 LoRA
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # 合并 LoRA 到基座模型
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
    except Exception:
        # 如果不是 PEFT 格式，直接加载
        logger.info("Loading as base model (not PEFT format)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 保存
    logger.info(f"Saving merged model to {export_dir}")
    model.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)

    # 保存元信息
    if meta:
        meta["export_format"] = "hf_safetensors"
        meta["export_date"] = datetime.now().isoformat()
        with open(os.path.join(export_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    # 清理显存
    del model
    import torch
    torch.cuda.empty_cache()

    logger.info(f"HF export complete: {export_dir}")
    return export_dir


def export_gguf(hf_model_dir: str, output_dir: str,
                quant_type: str = "Q4_K_M") -> str:
    """导出为 GGUF 格式（用于 Ollama/llama.cpp）"""
    logger.info(f"Exporting GGUF from {hf_model_dir}")

    gguf_dir = os.path.join(output_dir, "gguf_export")
    os.makedirs(gguf_dir, exist_ok=True)

    # 检查 llama.cpp 是否可用
    llama_cpp = _find_llama_cpp()
    if not llama_cpp:
        logger.warning("llama.cpp not found. Attempting to install via llama-cpp-python...")
        llama_cpp = _install_llama_cpp_tools()

    # Step 1: 转换为 GGUF（F16）
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py") if llama_cpp else None
    if convert_script and os.path.exists(convert_script):
        f16_path = os.path.join(gguf_dir, "model-f16.gguf")
        logger.info("Converting HF -> GGUF (F16)...")
        result = subprocess.run(
            ["python", convert_script, hf_model_dir, "--outfile", f16_path, "--outtype", "f16"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            # 降级：直接用 safetensors
            logger.info("Falling back: copying HF model as-is")
            if os.path.exists(hf_model_dir):
                shutil.copytree(hf_model_dir, os.path.join(gguf_dir, "hf_model"), dirs_exist_ok=True)
            return gguf_dir

        # Step 2: 量化
        quantize_bin = os.path.join(llama_cpp, "llama-quantize")
        if os.path.exists(quantize_bin) and os.path.exists(f16_path):
            quant_path = os.path.join(gguf_dir, f"model-{quant_type}.gguf")
            logger.info(f"Quantizing to {quant_type}...")
            subprocess.run(
                [quantize_bin, f16_path, quant_path, quant_type],
                capture_output=True, text=True, timeout=600,
            )
            # 删除 F16 大文件
            if os.path.exists(quant_path):
                os.remove(f16_path)
                logger.info(f"GGUF quantized model: {quant_path}")
    else:
        # llama.cpp 不可用，复制 HF 模型
        logger.warning("llama.cpp tools not available. Copying HF model for manual GGUF conversion.")
        if os.path.exists(hf_model_dir):
            shutil.copytree(hf_model_dir, os.path.join(gguf_dir, "hf_model"), dirs_exist_ok=True)

    logger.info(f"GGUF export complete: {gguf_dir}")
    return gguf_dir


def _find_llama_cpp() -> Optional[str]:
    """查找 llama.cpp 目录"""
    # 常见路径
    candidates = [
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/.local/share/llama.cpp"),
        "/usr/local/share/llama.cpp",
        os.environ.get("LLAMA_CPP_DIR", ""),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _install_llama_cpp_tools():
    """尝试安装 llama.cpp 转换工具"""
    try:
        import llama_cpp
        # llama-cpp-python 已安装，找到转换脚本
        pkg_dir = Path(llama_cpp.__file__).parent
        convert = pkg_dir / "llama_cpp" / "convert_hf_to_gguf.py"
        if convert.exists():
            return str(convert.parent)
    except ImportError:
        pass

    logger.info("llama.cpp conversion tools not found.")
    logger.info("To enable GGUF export, install llama.cpp:")
    logger.info("  git clone https://github.com/ggerganov/llama.cpp")
    logger.info("  cd llama.cpp && pip install -r requirements.txt")
    return None


def create_ollama_modelfile(gguf_path: str, model_name: str,
                            meta: Optional[dict] = None) -> str:
    """生成 Ollama Modelfile"""
    description = meta.get("description", "") if meta else ""

    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

SYSTEM \"\"\"{description}\"\"\"
"""
    return modelfile_content


def export_model(config: dict, model_path: str) -> dict:
    """完整导出流程"""
    export_cfg = config["export"]
    output_dir = config["output"]["output_dir"]
    export_format = export_cfg["format"]
    quant_type = export_cfg["gguf_quant"]

    # 读取训练元信息
    meta_path = os.path.join(model_path, "training_meta.json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    results = {}

    if export_format in ("hf", "both"):
        hf_dir = export_hf(model_path, model_path, output_dir, meta)
        results["hf"] = hf_dir

    if export_format in ("gguf", "both"):
        # GGUF 需要 HF 格式的模型
        if "hf" in results:
            gguf_dir = export_gguf(results["hf"], output_dir, quant_type)
        else:
            # 先导出 HF 再转 GGUF
            hf_dir = export_hf(model_path, model_path, output_dir, meta)
            gguf_dir = export_gguf(hf_dir, output_dir, quant_type)
        results["gguf"] = gguf_dir

    # 生成 Ollama Modelfile
    if "gguf" in results:
        gguf_dir = results["gguf"]
        gguf_files = list(Path(gguf_dir).glob("*.gguf"))
        if gguf_files:
            modelfile = create_ollama_modelfile(str(gguf_files[0]), "finetuned-model", meta)
            modelfile_path = os.path.join(gguf_dir, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(modelfile)
            results["ollama_modelfile"] = modelfile_path

    logger.info(f"Export complete. Formats: {list(results.keys())}")
    return results
