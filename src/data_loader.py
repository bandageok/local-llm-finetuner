"""数据集自动识别与加载 — 支持多种格式"""
import os
import json
import csv
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTS = {".json", ".jsonl", ".txt", ".csv", ".tsv"}

# 格式检测结果
FORMAT_TYPES = {
    "plain_text": "纯文本",
    "alpaca": "Alpaca (instruction/input/output)",
    "sharegpt": "ShareGPT (conversations)",
    "chatml": "ChatML (messages)",
    "openai_ft": "OpenAI Fine-tuning (messages)",
    "cot": "Chain-of-Thought (Question/Complex_CoT/Response)",
    "general_json": "通用 JSON",
}


def scan_dataset_dir(dataset_dir: str) -> List[str]:
    """扫描数据集目录，返回所有支持的文件"""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    files = []
    for f in dataset_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            files.append(str(f))

    if not files:
        raise ValueError(f"No supported files found in {dataset_dir}. "
                         f"Supported: {SUPPORTED_EXTS}")

    logger.info(f"Found {len(files)} dataset file(s) in {dataset_dir}")
    for f in files:
        logger.info(f"  - {f}")
    return files


def _read_jsonl(path: str) -> List[dict]:
    """读取 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"JSON parse error at {path}:{line_no}, skipping")
    return data


def _read_json(path: str) -> List[dict]:
    """读取 JSON / JSON 数组文件"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        # 可能是 {data: [...]} 格式
        if "data" in raw and isinstance(raw["data"], list):
            return raw["data"]
        return [raw]
    return [raw]


def _read_txt(path: str) -> List[dict]:
    """读取纯文本文件（按空行分段）"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按空行分段
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    # 如果没有空行，按行分
    if len(paragraphs) == 1 and "\n" in paragraphs[0]:
        paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

    return [{"text": p} for p in paragraphs]


def _read_csv(path: str) -> List[dict]:
    """读取 CSV/TSV 文件"""
    delimiter = "\t" if path.lower().endswith(".tsv") else ","
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
    return data


def read_file(path: str) -> List[dict]:
    """根据文件扩展名读取数据"""
    ext = Path(path).suffix.lower()
    if ext == ".jsonl":
        return _read_jsonl(path)
    elif ext == ".json":
        return _read_json(path)
    elif ext == ".txt":
        return _read_txt(path)
    elif ext in (".csv", ".tsv"):
        return _read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def detect_format(data: List[dict]) -> str:
    """自动检测数据集格式"""
    if not data:
        return "plain_text"

    sample = data[0]

    # Alpaca 格式: instruction + (input) + output
    if "instruction" in sample and "output" in sample:
        return "alpaca"

    # ShareGPT 格式: conversations [{from, value}, ...]
    if "conversations" in sample:
        convs = sample["conversations"]
        if isinstance(convs, list) and convs and "from" in convs[0]:
            return "sharegpt"

    # ChatML / OpenAI 格式: messages [{role, content}, ...]
    if "messages" in sample:
        msgs = sample["messages"]
        if isinstance(msgs, list) and msgs and "role" in msgs[0]:
            return "chatml"

    # COT 格式: Question + Complex_CoT + Response
    if "Question" in sample and "Complex_CoT" in sample and "Response" in sample:
        return "cot"

    # 纯文本格式: text 字段
    if "text" in sample:
        return "plain_text"

    # 通用 JSON：尝试找包含文本的字段
    for key in ["content", "body", "answer", "response", "completion"]:
        if key in sample and isinstance(sample[key], str):
            return "general_json"

    return "general_json"


def format_to_prompt(sample: dict, fmt: str, system_prompt: str) -> Dict[str, str]:
    """将不同格式的样本转换为统一的 prompt/response 格式"""
    # _prebuilt 样本（ShareGPT 扩展时已构建好）直接返回
    if sample.get("_prebuilt"):
        return {
            "prompt": sample.get("prompt", ""),
            "response": sample.get("response", ""),
            "cot": sample.get("cot", ""),
            "system": sample.get("system", system_prompt),
        }

    if fmt == "alpaca":
        instruction = sample.get("instruction", "")
        inp = sample.get("input", "")
        output = sample.get("output", "")
        if inp:
            prompt = f"{instruction}\n{inp}"
        else:
            prompt = instruction
        return {"prompt": prompt, "response": output, "system": system_prompt}

    elif fmt == "sharegpt":
        convs = sample.get("conversations", [])
        # 构建完整多轮对话
        messages = []
        for turn in convs:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})

        # 如果只有一轮，返回最后的 user->assistant 对
        if len(messages) <= 2:
            user_msg = ""
            assistant_msg = ""
            for m in messages:
                if m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]
            return {"prompt": user_msg, "response": assistant_msg, "system": system_prompt}

        # 多轮对话：拼接完整对话上下文
        # 最后一个 assistant 回复作为 response，前面所有消息作为 prompt context
        context_parts = []
        final_response = ""
        for m in messages:
            if m["role"] == "assistant" and m == messages[-1]:
                final_response = m["content"]
            elif m["role"] == "user":
                context_parts.append(f"用户: {m['content']}")
            elif m["role"] == "assistant":
                context_parts.append(f"助手: {m['content']}")

        context = "\n".join(context_parts)
        return {"prompt": context, "response": final_response, "system": system_prompt}

    elif fmt == "chatml":
        msgs = sample.get("messages", [])
        user_msg = ""
        assistant_msg = ""
        for m in reversed(msgs):
            if m["role"] == "assistant" and not assistant_msg:
                assistant_msg = m["content"]
            elif m["role"] == "user" and not user_msg:
                user_msg = m["content"]
            if user_msg and assistant_msg:
                break
        return {"prompt": user_msg, "response": assistant_msg, "system": system_prompt}

    elif fmt == "cot":
        question = sample.get("Question", "")
        cot = sample.get("Complex_CoT", "")
        response = sample.get("Response", "")
        return {"prompt": question, "cot": cot, "response": response, "system": system_prompt}

    elif fmt == "plain_text":
        text = sample.get("text", "")
        return {"prompt": "", "response": text, "system": system_prompt}

    else:  # general_json
        # 尝试提取 prompt/response
        prompt = ""
        response = ""
        for key in ["instruction", "input", "question", "prompt", "q", "query"]:
            if key in sample:
                prompt = str(sample[key])
                break
        for key in ["output", "response", "answer", "a", "completion", "content"]:
            if key in sample:
                response = str(sample[key])
                break
        return {"prompt": prompt, "response": response, "system": system_prompt}


def build_text(sample: dict, fmt: str, system_prompt: str, task_type: str,
               tokenizer=None) -> str:
    """将样本构建为完整的训练文本"""
    pr = format_to_prompt(sample, fmt, system_prompt)
    prompt = pr["prompt"]
    response = pr["response"]
    system = pr["system"]
    cot = pr.get("cot", "")

    # 优先使用 tokenizer.apply_chat_template（保证训练/推理 token 序列一致）
    if tokenizer is not None and task_type in ("instruction", "cot", "classification"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if task_type == "cot" and cot:
            # CoT: 将推理过程注入 assistant 回复
            messages.append({"role": "assistant", "content": f"让我们一步步思考：{cot}\n{response}"})
        else:
            messages.append({"role": "assistant", "content": response})
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            # fallback: 手动拼接（ tokenizer 不支持该模板时）
            pass

    # Fallback: 手动拼接格式（兼容不支持 apply_chat_template 的 tokenizer）
    if task_type == "instruction":
        if prompt:
            text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
        else:
            text = response
    elif task_type == "text_completion":
        text = response if not prompt else f"{prompt}\n{response}"
    elif task_type == "cot":
        if cot:
            text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n让我们一步步思考：{cot}\n{response}<|im_end|>"
            )
        else:
            text = response
    elif task_type == "classification":
        text = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
    else:
        text = response if not prompt else f"{prompt}\n{response}"

    return text


class DatasetLoader:
    """数据集加载器 — 自动识别格式，统一输出"""

    def __init__(self, config: dict):
        self.config = config
        self.dataset_dir = config["data"]["dataset_dir"]
        self.val_split = config["data"]["val_split"]
        self.shuffle = config["data"].get("shuffle", True)
        self.seed = config["data"].get("seed", 42)
        self.system_prompt = config["direction"]["system_prompt"]
        self.task_type = config["direction"]["task_type"]

        # 加载 tokenizer（用于 apply_chat_template，保证训练/推理 token 序列一致）
        model_path = config["model"]["name_or_path"]
        trust_remote = config["model"].get("trust_remote_code", True)
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote,
                padding_side="right",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info(f"Tokenizer loaded for chat template: {model_path}")
        except Exception as e:
            logger.warning(f"Could not load tokenizer for apply_chat_template: {e}")
            self._tokenizer = None

    def load(self) -> Tuple[List[Dict], List[Dict], str]:
        """
        加载数据集

        Returns:
            (train_data, val_data, detected_format)
            train_data: [{"text": "..."}, ...]
            val_data: [{"text": "..."}, ...]
            detected_format: 检测到的格式名称
        """
        import random
        random.seed(self.seed)

        # 扫描文件
        files = scan_dataset_dir(self.dataset_dir)

        # 读取所有文件
        all_data = []
        for f in files:
            data = read_file(f)
            logger.info(f"Read {len(data)} samples from {f}")
            all_data.extend(data)

        if not all_data:
            raise ValueError("No data loaded from dataset directory")

        # 自动检测格式（基于第一个样本）
        fmt = detect_format(all_data)
        fmt_name = FORMAT_TYPES.get(fmt, fmt)
        logger.info(f"Detected format: {fmt_name} ({fmt})")
        logger.info(f"Total samples: {len(all_data)}")

        # ShareGPT 多轮对话：为每一轮 assistant 回复生成训练样本
        if fmt == "sharegpt":
            expanded = []
            for sample in all_data:
                convs = sample.get("conversations", [])
                if not convs:
                    continue
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                messages = []
                for turn in convs:
                    role = role_map.get(turn["from"], turn["from"])
                    messages.append({"role": role, "content": turn["value"]})

                # 为每个 assistant 回复生成一个样本
                for i, msg in enumerate(messages):
                    if msg["role"] != "assistant":
                        continue
                    # prompt = 该轮之前所有消息
                    context_parts = []
                    for prev in messages[:i]:
                        if prev["role"] == "user":
                            context_parts.append(f"用户: {prev['content']}")
                        elif prev["role"] == "assistant":
                            context_parts.append(f"助手: {prev['content']}")
                    if not context_parts:
                        continue
                    context = "\n".join(context_parts)
                    expanded.append({"prompt": context, "response": msg["content"],
                                     "system": self.system_prompt, "_prebuilt": True})
            if expanded:
                all_data = expanded
                logger.info(f"ShareGPT multi-turn: expanded to {len(all_data)} training samples")

        # 打乱
        if self.shuffle:
            random.shuffle(all_data)

        # 构建训练文本
        processed = []
        skipped = 0
        min_len = self.config["data"].get("min_length", 10)
        max_len = self.config["data"].get("max_length", 8192)

        for sample in all_data:
            try:
                text = build_text(sample, fmt, self.system_prompt, self.task_type,
                             tokenizer=self._tokenizer)
                text = text.strip()
                if not text:
                    skipped += 1
                    continue

                # 质量检查：长度过滤
                char_len = len(text)
                if char_len < min_len or char_len > max_len:
                    skipped += 1
                    continue

                # 质量检查：过滤纯空白/特殊字符
                if len(text.replace("\n", "").replace(" ", "").replace("\t", "")) < 5:
                    skipped += 1
                    continue

                processed.append({"text": text})
            except Exception as e:
                skipped += 1
                logger.debug(f"Skipped sample: {e}")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} samples (quality check + processing errors)")

        # 去重
        if self.config["data"].get("dedup", False):
            before = len(processed)
            seen = set()
            deduped = []
            for item in processed:
                # 用 text 前 200 字符做去重 key（避免完全重复样本）
                key = item["text"][:200]
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            dedup_count = before - len(deduped)
            processed = deduped
            if dedup_count > 0:
                logger.info(f"Deduplication: removed {dedup_count} duplicates ({before} -> {len(processed)})")

        if not processed:
            raise ValueError("No valid training samples after processing")

        # 划分训练/验证集
        val_size = max(1, int(len(processed) * self.val_split))
        val_data = processed[:val_size]
        train_data = processed[val_size:]

        logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

        return train_data, val_data, fmt


def print_dataset_preview(dataset_dir: str):
    """预览数据集"""
    files = scan_dataset_dir(dataset_dir)
    for f in files[:3]:
        data = read_file(f)
        fmt = detect_format(data)
        print(f"\n--- {f} ---")
        print(f"Format: {FORMAT_TYPES.get(fmt, fmt)}")
        print(f"Samples: {len(data)}")
        print(f"Sample keys: {list(data[0].keys()) if data else 'N/A'}")
        if data:
            sample_str = json.dumps(data[0], ensure_ascii=False, indent=2)
            print(f"First sample:\n{sample_str[:500]}")
