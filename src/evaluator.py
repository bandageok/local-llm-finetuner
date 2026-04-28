"""评估与可视化 — loss 曲线 + 拟合曲线 + 样例输出 + BERTScore/Rouge + 推理性能监控"""
import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── 可选依赖检测 ──────────────────────────────────────────────
try:
    from bert_score import score as bert_score_fn
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False

try:
    from rouge import Rouge as _Rouge
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False


def parse_tensorboard_events(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """从 TensorBoard 日志解析训练指标"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        logger.error("tensorboard not installed, cannot parse logs")
        return {}

    log_dir = Path(log_dir)
    metrics = {}

    for event_file in log_dir.rglob("events.out.tfevents.*"):
        ea = event_accumulator.EventAccumulator(str(event_file.parent))
        ea.Reload()

        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            if tag not in metrics:
                metrics[tag] = []
            for e in events:
                metrics[tag].append((e.step, e.value))

    return metrics


def polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """多项式拟合"""
    if len(x) < degree + 1:
        degree = max(1, len(x) - 1)
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)

    # R² score
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return y_fit, coeffs, r2


def generate_loss_plot(metrics: Dict, output_dir: str, num_epochs: int):
    """生成 loss 曲线图（含拟合曲线）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取 loss 数据
    train_loss = metrics.get("train/loss", metrics.get("loss", []))
    eval_loss = metrics.get("eval/loss", [])

    if not train_loss:
        logger.warning("No training loss data found")
        return None

    train_steps = np.array([x[0] for x in train_loss])
    train_vals = np.array([x[1] for x in train_loss])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Report — {num_epochs} Epochs", fontsize=14, fontweight="bold")

    # 1. Train loss + 拟合曲线
    ax = axes[0, 0]
    ax.plot(train_steps, train_vals, alpha=0.4, color="#00d4ff", label="Raw Loss")

    # 拟合
    if len(train_steps) > 4:
        degree = min(5, len(train_steps) - 1)
        y_fit, coeffs, r2 = polynomial_fit(train_steps.astype(float), train_vals, degree)
        ax.plot(train_steps, y_fit, color="#00ffd5", linewidth=2,
                label=f"Fit (R²={r2:.4f})")
        ax.text(0.02, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss + Fitting Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#0a0a0f")

    # 2. Eval loss
    ax = axes[0, 1]
    if eval_loss:
        eval_steps = np.array([x[0] for x in eval_loss])
        eval_vals = np.array([x[1] for x in eval_loss])
        ax.plot(eval_steps, eval_vals, "o-", color="#ec4899", markersize=4, label="Eval Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Evaluation Loss")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No eval data", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#0a0a0f")

    # 3. Learning Rate
    ax = axes[1, 0]
    lr_data = metrics.get("train/learning_rate", [])
    if lr_data:
        lr_steps = np.array([x[0] for x in lr_data])
        lr_vals = np.array([x[1] for x in lr_data])
        ax.plot(lr_steps, lr_vals, color="#a855f7", linewidth=1.5)
        ax.set_xlabel("Steps")
        ax.set_ylabel("LR")
        ax.set_title("Learning Rate Schedule")
    else:
        ax.text(0.5, 0.5, "No LR data", transform=ax.transAxes,
                ha="center", va="center", color="#888")
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#0a0a0f")

    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis("off")

    summary_lines = []
    if train_vals.size > 0:
        summary_lines.append(f"Initial Loss: {train_vals[0]:.4f}")
        summary_lines.append(f"Final Loss: {train_vals[-1]:.4f}")
        summary_lines.append(f"Best Loss: {train_vals.min():.4f} (step {train_steps[train_vals.argmin()]})")
        summary_lines.append(f"Loss Reduction: {((train_vals[0] - train_vals[-1]) / train_vals[0] * 100):.1f}%")
    summary_lines.append(f"Total Steps: {int(train_steps[-1]) if train_steps.size > 0 else 0}")
    summary_lines.append(f"Train Samples: {len(train_vals)}")
    if eval_loss:
        eval_vals = np.array([x[1] for x in eval_loss])
        summary_lines.append(f"Best Eval Loss: {eval_vals.min():.4f}")

    summary_text = "\n".join(summary_lines)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#0a0a0f", edgecolor="#00ffd5", alpha=0.9))

    plt.tight_layout()
    plot_path = output_dir / "training_report.png"
    plt.savefig(plot_path, dpi=150, facecolor="#050810", edgecolor="none")
    plt.close()

    logger.info(f"Training report saved to {plot_path}")
    return str(plot_path)


def generate_sample_output(model_path: str, tokenizer, num_samples: int = 3,
                           max_length: int = 256,
                           system_prompt: str = "你是一个有用的AI助手。"):
    """
    加载微调模型生成样例（集成 AdvancedPerformanceMonitor 流式推理监控）

    Args:
        model_path: 模型路径
        tokenizer: tokenizer 实例
        num_samples: 生成样例数
        max_length: 最大生成长度
        system_prompt: 系统提示词

    Returns:
        [{"prompt": ..., "response": ..., "performance": {...}}, ...]
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
    from threading import Thread

    from .performance_monitor import AdvancedPerformanceMonitor, print_detailed_performance_report

    logger.info("Generating sample outputs with performance monitoring...")

    # 加载模型
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 尝试启用 Unsloth 推理模式
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        logger.info("Unsloth inference mode enabled")
    except Exception:
        logger.info("Using standard HF inference mode")

    test_prompts = [
        "请介绍一下人工智能的发展历史。",
        "写一首关于春天的诗。",
        "解释一下量子计算的基本原理。",
    ]

    results = []
    for idx, prompt in enumerate(test_prompts[:num_samples]):
        logger.info(f"\n--- Sample {idx + 1}/{num_samples} ---")
        logger.info(f"[Prompt] {prompt}")

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_token_count = inputs.input_ids.shape[1]

        # 初始化监控器
        monitor = AdvancedPerformanceMonitor()
        monitor.tokenizer = tokenizer
        monitor.start()

        # 流式生成
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "use_cache": True,
            "streamer": streamer,
            "do_sample": True,
        }

        # 推理期间后台采样显存/CPU/GPU
        def background_monitor():
            while monitor.monitoring:
                monitor.collect_snapshot()
                time.sleep(0.1)

        monitor_thread = Thread(target=background_monitor, daemon=True)

        # 开始推理计时
        monitor.start_inference_timing(input_token_count)
        monitor_thread.start()

        # 启动生成线程
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 流式收集输出 + 记录首 Token
        first_token_received = False
        generated_text = ""

        for new_text in streamer:
            if not first_token_received:
                monitor.record_first_token()
                first_token_received = True
            monitor.record_token_generation(new_text)
            generated_text += new_text

        thread.join()
        monitor.generation_end_time = time.time()
        monitor.stop()
        monitor_thread.join(timeout=2)

        # 计算性能指标
        monitor.output_tokens = len(tokenizer.encode(generated_text))
        speeds = monitor.calculate_speeds()
        summary = monitor.get_summary()
        memory_analysis = monitor.get_memory_analysis()

        # 打印报告
        print_detailed_performance_report(speeds, summary, memory_analysis)

        results.append({
            "prompt": prompt,
            "response": generated_text,
            "performance": {
                "first_token_latency": speeds.get("first_token_latency"),
                "total_inference_time": speeds.get("total_inference_time"),
                "total_inference_speed": speeds.get("total_inference_speed"),
                "pure_generation_speed": speeds.get("pure_generation_speed"),
                "input_tokens": speeds.get("input_tokens"),
                "output_tokens": speeds.get("output_tokens"),
                "gpu_memory_peak": summary.get("memory_peaks", {}).get("pytorch_memory_peak"),
                "gpu_usage_avg": summary.get("gpu_usage_avg"),
                "gpu_temp_avg": summary.get("gpu_temp_avg"),
                "gpu_power_avg": summary.get("gpu_power_avg"),
            },
        })

        logger.info(f"[Response] {generated_text[:200]}...")

    del model
    torch.cuda.empty_cache()

    return results


def compute_bertscore(predictions: List[str], references: List[str], lang: str = "zh") -> Dict:
    """计算 BERTScore（需 bert-score 包）"""
    if not _HAS_BERTSCORE:
        logger.warning("bert-score not installed, skipping BERTScore")
        return {}
    P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=True)
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall": round(R.mean().item(), 4),
        "bertscore_f1": round(F1.mean().item(), 4),
    }


def _tokenize_chinese(text: str) -> str:
    """使用 jieba 分词，中文文本转为空格分隔的词序列

    Rouge 的 n-gram 基于空格分词，不做分词直接比较中文会严重低估分数。
    """
    try:
        import jieba
        words = jieba.cut(text)
        return " ".join(words)
    except Exception:
        # fallback: 直接返回原文本（按字符 n-gram 处理）
        return text


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    """计算 Rouge 分数（支持中文 jieba 分词预处理）"""
    if not _HAS_ROUGE:
        logger.warning("rouge not installed, skipping Rouge")
        return {}

    # 尝试加载 jieba（检测是否为中文文本）
    try:
        import jieba as _jieba
        _HAS_JIEBA = True
    except ImportError:
        _HAS_JIEBA = False
        logger.warning("jieba not installed, Rouge on Chinese text may be inaccurate")

    # 非空字符串
    preds = [p if p.strip() else "." for p in predictions]
    refs = [r if r.strip() else "." for r in references]

    # 中文文本（包含CJK字符）做 jieba 分词
    if _HAS_JIEBA:
        preds_tokenized = [_tokenize_chinese(p) for p in preds]
        refs_tokenized = [_tokenize_chinese(r) for r in refs]
    else:
        preds_tokenized = preds
        refs_tokenized = refs

    scores = _Rouge().get_scores(preds_tokenized, refs_tokenized, avg=True)
    return {
        "rouge-1": round(scores["rouge-1"]["f"], 4),
        "rouge-2": round(scores["rouge-2"]["f"], 4),
        "rouge-l": round(scores["rouge-l"]["f"], 4),
    }


def generate_evaluation_report(eval_results: Dict, output_dir: str):
    """生成评估可视化报告 — BERTScore/Rouge 柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_bertscore = "bertscore_precision" in eval_results
    has_rouge = "rouge-1" in eval_results

    if not has_bertscore and not has_rouge:
        logger.info("No BERTScore/Rouge data, skipping evaluation report chart")
        return None

    # 确定子图数量
    n_plots = 1 + int(has_bertscore) + int(has_rouge)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Evaluation Report", fontsize=14, fontweight="bold")

    idx = 0

    # Loss 摘要（文本卡片）
    ax = axes[idx]
    idx += 1
    ax.axis("off")
    lines = []
    if "eval_loss" in eval_results:
        lines.append(f"Eval Loss: {eval_results['eval_loss']:.4f}")
    if has_bertscore:
        lines.append(f"BERTScore F1: {eval_results['bertscore_f1']:.4f}")
    if has_rouge:
        lines.append(f"Rouge-L: {eval_results['rouge-l']:.4f}")
    if lines:
        ax.text(0.5, 0.5, "\n".join(lines), transform=ax.transAxes,
                fontsize=13, ha="center", va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#0a0a0f", edgecolor="#00ffd5", alpha=0.9))

    # BERTScore 柱状图
    if has_bertscore:
        ax = axes[idx]
        idx += 1
        labels = ["Precision", "Recall", "F1"]
        vals = [eval_results["bertscore_precision"],
                eval_results["bertscore_recall"],
                eval_results["bertscore_f1"]]
        colors = ["#00d4ff", "#00ffd5", "#a855f7"]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title("BERTScore")
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, alpha=0.2, axis="y")

    # Rouge 柱状图
    if has_rouge:
        ax = axes[idx]
        idx += 1
        labels = ["Rouge-1", "Rouge-2", "Rouge-L"]
        vals = [eval_results["rouge-1"], eval_results["rouge-2"], eval_results["rouge-l"]]
        colors = ["#ec4899", "#f59e0b", "#10b981"]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title("Rouge Scores")
        ax.set_facecolor("#0a0a0f")
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    report_path = output_dir / "evaluation_report.png"
    plt.savefig(report_path, dpi=150, facecolor="#050810", edgecolor="none")
    plt.close()
    logger.info(f"Evaluation report saved to {report_path}")
    return str(report_path)


def evaluate_model(model_path: str, test_data: List[Dict],
                   output_dir: str, lang: str = "zh",
                   max_length: int = 256,
                   system_prompt: str = "你是一个有用的AI助手。") -> Dict:
    """综合评估：加载模型 → 生成预测 → Loss + BERTScore + Rouge + 可视化报告"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # 生成预测
    predictions: List[str] = []
    references: List[str] = []
    total_loss = 0.0
    n_samples = 0

    logger.info(f"Running inference on {len(test_data)} samples ...")
    for item in test_data:
        prompt = item.get("prompt", item.get("instruction", ""))
        ref = item.get("response", item.get("output", ""))
        if not prompt or not ref:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(pred)
        references.append(ref)
        n_samples += 1

        # 计算单条 loss（拼接 prompt+ref，labels 中 prompt 部分设为 -100）
        ref_ids = tokenizer(ref, return_tensors="pt").to(model.device)["input_ids"]
        full_ids = torch.cat([inputs["input_ids"], ref_ids], dim=-1)
        labels = full_ids.clone()
        labels[:, :inputs["input_ids"].shape[1]] = -100
        with torch.no_grad():
            loss_out = model(input_ids=full_ids, labels=labels)
        total_loss += loss_out.loss.item()

    del model
    torch.cuda.empty_cache()

    eval_results: Dict = {}
    if n_samples > 0:
        eval_results["eval_loss"] = round(total_loss / n_samples, 4)
        eval_results["num_samples"] = n_samples

        # BERTScore
        bs = compute_bertscore(predictions, references, lang=lang)
        eval_results.update(bs)

        # Rouge
        rg = compute_rouge(predictions, references)
        eval_results.update(rg)

    # 保存 JSON 报告
    report_json = output_dir / "evaluation_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON report saved to {report_json}")

    # 可视化
    chart_path = generate_evaluation_report(eval_results, str(output_dir))
    eval_results["report_chart"] = chart_path

    # 保存预测样例
    samples_path = output_dir / "eval_predictions.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump([{"prompt": d.get("prompt", d.get("instruction", "")),
                     "reference": d.get("response", d.get("output", "")),
                     "prediction": p}
                    for d, p in zip(test_data[:n_samples], predictions)],
                   f, indent=2, ensure_ascii=False)

    return eval_results


def evaluate_training(config: dict, model_path: str = None):
    """完整评估流程"""
    output_dir = config["output"]["output_dir"]
    log_dir = config["output"]["log_dir"]
    num_epochs = config["training"]["num_epochs"]
    output_samples = config["training"]["output_samples"]
    num_sample_outputs = config["training"]["num_samples"]

    # 解析 TensorBoard 日志
    logger.info("Parsing TensorBoard logs...")
    metrics = parse_tensorboard_events(log_dir)

    if metrics:
        # 生成 loss 曲线图
        plot_path = generate_loss_plot(metrics, output_dir, num_epochs)

        # 保存原始指标数据
        metrics_path = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.warning("No metrics found, skipping plot generation")
        plot_path = None

    # 可选：生成样例输出
    sample_results = None
    if output_samples and model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            sample_results = generate_sample_output(
                model_path, tokenizer,
                num_samples=num_sample_outputs,
                max_length=256,
            )
            # 保存样例
            samples_path = os.path.join(output_dir, "sample_outputs.json")
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(sample_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Samples saved to {samples_path}")
        except Exception as e:
            logger.error(f"Failed to generate samples: {e}")

    return {
        "plot_path": plot_path,
        "metrics": metrics,
        "samples": sample_results,
    }
