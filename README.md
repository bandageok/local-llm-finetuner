# Local LLM Fine-Tuner

QLoRA 微调管道，本地训练、评估、导出、部署一站式。

**硬件：** RTX 4060 8GB 可跑 Qwen3-8B（QLoRA），Unsloth 加速 2x，显存节省 60%。

---

## 架构

```
llm-finetune/
├── train.py              # 训练入口
├── evaluate.py           # 评估入口
├── export.py             # 导出入口（HF / GGUF）
├── deploy.py             # 部署入口（Ollama）
├── setup.py              # 环境自动配置（检测 CUDA 版本）
├── config.yaml           # 全部参数配置
├── requirements.txt
├── src/
│   ├── trainer.py        # QLoRA 训练核心（SFTTrainer）
│   ├── data_loader.py    # 数据加载、格式化、质量过滤
│   ├── evaluator.py      # BERTScore / Rouge / Loss 可视化
│   ├── exporter.py       # HF safetensors / GGUF 量化导出
│   ├── deployer.py       # Ollama 模型打包
│   ├── gpu_detector.py   # GPU 检测 + 显存安全检查
│   └── performance_monitor.py  # 训练过程监控
└── datasets/             # 数据集目录（放置 .jsonl）
```

---

## 快速开始

### 1. 环境安装

```bash
# 自动检测 CUDA 并安装 PyTorch + 依赖
python setup.py
```

或手动：

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`：

```yaml
model:
  name_or_path: "Qwen/Qwen3-8B"   # 或本地路径 "D:/models/Qwen3-8B"

direction:
  task_type: "instruction"         # instruction / chat / text_completion / cot / classification
  system_prompt: "你是一个有用的AI助手。"
  description: "My Fine-tuned Model"

qlora:
  r: 16                            # LoRA rank：8=省显存，16=平衡，32=高精度

training:
  num_epochs: 3
  learning_rate: 2e-4
  max_seq_length: 2048

data:
  dataset_dir: "./datasets"
  val_split: 0.05
```

### 3. 准备数据集

在 `datasets/` 下放置 `.jsonl` 文件，每行一条样本：

```jsonl
{"messages": [{"role": "system", "content": "你是一个有帮助的助手。"}, {"role": "user", "content": "什么是大语言模型？"}, {"role": "assistant", "content": "大语言模型是..."}]}
```

支持格式：`instruction` / `chat` / `text_completion`，自动识别。

### 4. 训练

```bash
python train.py
```

自动流程：GPU 检测 → 显存安全检查 → 训练 → 交互式评估 → 导出

### 5. 评估

```bash
python evaluate.py [model_path] --full
```

输出：`eval_loss` / `BERTScore F1` / `Rouge-L` + 可视化图表。

### 6. 导出

```bash
python export.py [model_path] --format both   # HF + GGUF
python export.py [model_path] --format gguf   # 仅 GGUF（Q4_K_M）
```

### 7. 部署（Ollama）

```bash
python deploy.py [model_path] --name mymodel
ollama run mymodel
```

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **Unsloth 加速** | 2x 训练速度，60% 显存节省；未安装自动降级原生 HF |
| **QLoRA** | 4bit NF4 量化 + LoRA rank 8/16/32 可选 |
| **数据质量** | 自动过滤短/超长样本、去重、格式校验 |
| **显存安全** | 训练前检测 VRAM，不够则提示降级 |
| **多格式导出** | HuggingFace safetensors + GGUF Q4/Q5/Q8 |
| **Ollama 部署** | 一键打包成本地可运行模型 |
| **评估可视化** | BERTScore + Rouge + Loss 曲线 |

---

## 硬件参考

| 模型 | 精度 | LoRA Rank | 显存 |
|------|------|-----------|------|
| Qwen3-8B | Q4_K_M + LoRA | r=16 | ~6 GB |
| Qwen3-4B | Q4_K_M + LoRA | r=16 | ~4 GB |
| Qwen2.5-7B | Q4_K_M + LoRA | r=16 | ~5 GB |

---

## 项目来源

`local-llm-finetuner` — 欢迎 Star / Issue / Fork。
