"""QLoRA 训练核心模块"""
import os
import gc
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

logger = logging.getLogger(__name__)


def build_bnb_config(config: dict) -> BitsAndBytesConfig:
    """构建 BitsAndBytes 量化配置"""
    qlora = config["qlora"]
    compute_dtype = getattr(torch, qlora["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(config: dict, gpu_optimal: Optional[dict] = None) -> LoraConfig:
    """构建 LoRA 配置"""
    qlora = config["qlora"]
    r = gpu_optimal.get("lora_r", qlora["r"]) if gpu_optimal else qlora["r"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        target_modules=qlora["target_modules"],
        bias="none",
    )


def load_model(config: dict, gpu_optimal: Optional[dict] = None):
    """加载量化模型和 tokenizer（支持 Unsloth 加速）"""
    model_cfg = config["model"]
    model_path = model_cfg["name_or_path"]

    # Unsloth 加速（2x 速度，60% 显存节省）
    use_unsloth = config.get("unsloth", {}).get("enabled", False)
    unsloth_fallback = config.get("unsloth", {}).get("fallback", True)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            logger.info("Loading model with Unsloth acceleration...")

            max_seq = config["training"]["max_seq_length"]
            if gpu_optimal:
                max_seq = gpu_optimal.get("max_seq_length", max_seq)

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq,
                dtype=None,  # auto-detect
                load_in_4bit=True,
            )

            # Unsloth LoRA
            qlora = config["qlora"]
            r = gpu_optimal.get("lora_r", qlora["r"]) if gpu_optimal else qlora["r"]

            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                lora_alpha=qlora["lora_alpha"],
                lora_dropout=qlora["lora_dropout"],
                target_modules=qlora["target_modules"],
                bias="none",
                use_gradient_checkpointing="unsloth",  # 30% less VRAM
            )

            trainable_params, all_params = model.get_nb_trainable_parameters()
            trainable_pct = 100 * trainable_params / all_params
            logger.info(f"Unsloth loaded. Trainable: {trainable_params:,} / {all_params:,} ({trainable_pct:.2f}%)")

            return model, tokenizer, True  # True = unsloth mode

        except ImportError:
            if not unsloth_fallback:
                raise
            logger.warning("Unsloth not installed, falling back to native HF+PEFT")
        except Exception as e:
            if not unsloth_fallback:
                raise
            logger.warning(f"Unsloth load failed: {e}, falling back to native HF+PEFT")

    # 原生 HF+PEFT 加载
    bnb_config = build_bnb_config(config)
    logger.info(f"Loading model (native HF+PEFT): {model_path}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 确定 dtype
    if gpu_optimal and gpu_optimal.get("bf16", False):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if _has_flash_attn() else "sdpa",
    )

    # 梯度检查点（必须同时禁用 use_cache）
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled, use_cache disabled")

    # 准备模型用于 k-bit 训练
    model = prepare_model_for_kbit_training(model)

    # 应用 LoRA
    lora_config = build_lora_config(config, gpu_optimal)
    model = get_peft_model(model, lora_config)

    # 打印可训练参数
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_pct = 100 * trainable_params / all_params
    logger.info(f"Trainable params: {trainable_params:,} / {all_params:,} ({trainable_pct:.2f}%)")

    return model, tokenizer, False  # False = native mode


def _has_flash_attn() -> bool:
    """检查是否安装了 flash-attn"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def prepare_dataset(train_data: List[Dict], val_data: List[Dict],
                    tokenizer, max_seq_length: int):
    """将数据转换为 HuggingFace Dataset 并 tokenize"""

    def tokenize_fn(examples):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        # labels = input_ids（causal LM）
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
    )
    if val_dataset:
        val_dataset = val_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing val",
        )

    return train_dataset, val_dataset


def build_training_args(config: dict, gpu_optimal: Optional[dict] = None) -> TrainingArguments:
    """构建 TrainingArguments"""
    training = config["training"]
    output_dir = config["output"]["output_dir"]

    # 使用 GPU 推荐值覆盖
    batch_size = gpu_optimal.get("batch_size", 1) if gpu_optimal else 1
    grad_accum = gpu_optimal.get("gradient_accumulation_steps",
                                   training["gradient_accumulation_steps"]) if gpu_optimal else training["gradient_accumulation_steps"]
    max_seq = gpu_optimal.get("max_seq_length", training["max_seq_length"]) if gpu_optimal else training["max_seq_length"]
    bf16 = gpu_optimal.get("bf16", training.get("bf16", False)) if gpu_optimal else training.get("bf16", False)
    fp16 = not bf16

    # 保存间隔（每个 epoch 评估）
    eval_every_n = training["eval_every_n_epochs"]
    num_epochs = training["num_epochs"]

    # WandB 配置
    wandb_cfg = config.get("wandb", {})
    report_to = "tensorboard"
    if wandb_cfg.get("enabled", False):
        report_to = "wandb"
        api_key = wandb_cfg.get("api_key", "")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        run_name = wandb_cfg.get("run_name", "")
        if run_name:
            os.environ["WANDB_RUN_NAME"] = run_name
        project = wandb_cfg.get("project", "llm-finetune")
        os.environ["WANDB_PROJECT"] = project
        logger.info(f"WandB enabled: project={project}, run_name={run_name or 'auto'}")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        bf16=bf16,
        fp16=fp16,
        logging_steps=10,
        logging_dir=config["output"]["log_dir"],
        evaluation_strategy="epoch" if eval_every_n == 1 else "steps",
        eval_steps=100 if eval_every_n > 1 else None,
        save_strategy="epoch" if eval_every_n == 1 else "steps",
        save_steps=100 if eval_every_n > 1 else None,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=training["optimizer"],
        gradient_checkpointing=training["gradient_checkpointing"],
        report_to=report_to,
        seed=config["data"]["seed"],
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )


def train(config: dict, gpu_optimal: Optional[dict] = None):
    """完整训练流程"""
    from .data_loader import DatasetLoader

    # 1. 检测 GPU
    if gpu_optimal is None:
        from .gpu_detector import GPUInfo
        info = GPUInfo()
        gpu_optimal = info.get_optimal_config()

    max_seq = gpu_optimal.get("max_seq_length", config["training"]["max_seq_length"])

    # 2. 加载数据
    logger.info("=" * 50)
    logger.info("Stage 1: Loading dataset")
    loader = DatasetLoader(config)
    train_data, val_data, fmt = loader.load()

    # 3. 加载模型
    logger.info("=" * 50)
    logger.info("Stage 2: Loading model with QLoRA")
    model, tokenizer, use_unsloth = load_model(config, gpu_optimal)

    # 4. 准备数据集
    logger.info("=" * 50)
    logger.info("Stage 3: Tokenizing dataset")
    train_dataset, val_dataset = prepare_dataset(train_data, val_data, tokenizer, max_seq)

    # 5. 训练参数
    training_args = build_training_args(config, gpu_optimal)

    # 6. Checkpoint 恢复路径
    resume_from = config["training"].get("resume_from_checkpoint", "")
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
    else:
        resume_from = None

    # 7. 创建 Trainer（Unsloth 用 SFTTrainer，否则用标准 Trainer）
    # EarlyStoppingCallback（有 eval dataset 时启用）
    callbacks = []
    if val_dataset:
        patience = config["training"].get("early_stopping_patience", 3)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        logger.info(f"EarlyStopping enabled: patience={patience}")

    if use_unsloth:
        # Unsloth gradient_checkpointing 时禁用 use_cache
        if config["training"]["gradient_checkpointing"]:
            model.config.use_cache = False
            logger.info("Unsloth: use_cache disabled for gradient checkpointing")

        try:
            from trl import SFTTrainer

            # Unsloth SFTTrainer 需要原始文本列，不走 prepare_dataset()
            # 重新从 loader 获取原始数据
            raw_train = Dataset.from_list(train_data)
            raw_val = Dataset.from_list(val_data) if val_data else None

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=raw_train,
                eval_dataset=raw_val,
                dataset_text_field="text",
                max_seq_length=max_seq,
                args=training_args,
                callbacks=callbacks,
            )
            logger.info("Using Unsloth SFTTrainer (raw text dataset)")
        except ImportError:
            use_unsloth = False
            logger.warning("SFTTrainer not available, falling back to standard Trainer")

    if not use_unsloth:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    # 9. 训练
    logger.info("=" * 50)
    logger.info("Stage 4: Starting training")
    trainer.train(resume_from_checkpoint=resume_from)

    # 10. 保存最终模型
    logger.info("=" * 50)
    logger.info("Stage 5: Saving model")
    output_dir = config["output"]["output_dir"]
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 保存配置信息到模型目录
    import json
    meta = {
        "base_model": config["model"]["name_or_path"],
        "task_type": config["direction"]["task_type"],
        "description": config["direction"]["description"],
        "qlora_r": config["qlora"]["r"],
        "dataset_format": fmt,
        "num_epochs": config["training"]["num_epochs"],
        "max_seq_length": max_seq,
    }
    with open(os.path.join(final_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Model saved to {final_dir}")

    # 11. 清理显存
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return final_dir
