"""生成带完整注释的 notebook"""
import json

def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": lines}

def code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []

# === HEADER ===
cells.append(md([
    "# LLM QLoRA 微调训练 notebook\n\n**llm-finetune 框架 | 支持 Unsloth 加速 | 推理性能监控 | BERTScore+Rouge 评估**\n\n---\n\n## 快速开始\n\n**Step 1** 安装依赖：`pip install -r requirements.txt`\n\n**Step 2** 放入数据到 `./datasets/`\n\n**Step 3** 修改 Cell 1 的配置（模型路径 + 数据目录）\n\n**Step 4** 按顺序 Run 每个 Cell\n\n---\n\n## Stage 速查\n\n| Stage | Cell | 作用 | 何时跑 |\n|---|---|---|---|\n| 0 | Cell 0 | 环境检查 | 只需一次 |\n| 1 | Cell 1 | 配置参数 | 只需一次 |\n| 2 | Cell 2 | GPU 显存检测 | 只需一次 |\n| 3 | Cell 3 | 数据加载预览 | 数据不变可跳过 |\n| 4 | Cell 4 | 模型加载 | 模型不变可跳过 |\n| 5 | Cell 5 | 训练 | 每次训练 |\n| 6 | Cell 6 | 评估+推理 | 训练完成后 |\n| 7 | Cell 7 | 独立推理测试 | 随时可跑 |\n| 8 | Cell 8 | 导出模型 | 训练完成后 |\n\n---\n\n## 任务类型说明\n\n| task_type | 说明 | 适用场景 |\n|---|---|---|\n| `instruction` | 标准指令微调 | 推荐新手，通用问答 |\n| `cot` | Chain-of-Thought 思维链 | 需要数据含 Complex_CoT 字段 |\n| `classification` | 文本分类 | 输入文本，输出类别 |\n| `text_completion` | 文本续写 | 给一段文字，续写 |\n"]))

# === CELL 0 ===
cells.append(md(["---\n\n## Stage 0: 环境检查\n\n**作用**：检查 Python 版本、GPU、核心依赖是否就绪\n\n**正常输出示例**：\n```\nPython 版本: 3.10.x ...\nCUDA 可用: True\nGPU 型号: NVIDIA GeForce RTX 4060 Laptop GPU\n显存大小: 8.0 GB\n  [OK] transformers\n  [OK] peft\n  ...\n```\n\n**GPU 显示 False**：确认 CUDA 驱动已安装\n**依赖显示 [MISSING]**：运行 `pip install -r requirements.txt`\n"]))

cells.append(code([
    "import sys, os\n\n# 将项目根目录加入 Python 模块搜索路径\nsys.path.insert(0, os.path.dirname(os.path.abspath('.')))\n\n# 打印 Python 版本\nprint('Python 版本:', sys.version)\n\n# ── 检查 GPU ──\nimport torch\nprint()\nprint('CUDA 可用:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('GPU 型号:', torch.cuda.get_device_name(0))\n    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3\n    print('显存大小: {:.1f} GB'.format(vram))\nelse:\n    print('警告: 未检测到 GPU，将使用 CPU 训练（极慢）')\n\n# ── 检查核心依赖 ──\nprint()\nprint('核心依赖检查:')\n\ndeps = [\n    ('yaml',        'yaml'),\n    ('transformers','transformers'),\n    ('peft',       'peft'),\n    ('unsloth',    'unsloth'),\n    ('trl',        'trl'),\n    ('datasets',   'datasets'),\n    ('bitsandbytes','bitsandbytes'),\n    ('pynvml',     'pynvml'),\n    ('psutil',     'psutil'),\n    ('jieba',      'jieba'),\n    ('rouge',      'rouge'),\n    ('bert_score', 'bert_score'),\n]\n\nall_ok = True\nfor display, imp in deps:\n    try:\n        __import__(imp)\n        print('  [OK] {}'.format(display))\n    except ImportError:\n        print('  [MISSING] {} -- 运行: pip install {}'.format(display, display))\n        all_ok = False\n\nif not all_ok:\n    print()\n    print('一键安装: pip install -r requirements.txt')\n"]))

# === CELL 1 ===
cells.append(md(["---\n\n## Stage 1: 配置参数\n\n**必须修改的三项**：\n\n| 参数 | 说明 | 示例 |\n|---|---|---|\n| `model.name_or_path` | 模型路径 | `Qwen/Qwen3-8B` 或 `D:/models/Qwen3-8B` |\n| `data.dataset_dir` | 数据集目录 | `./datasets` |\n| `direction.task_type` | 任务类型 | `instruction` |\n\n**task_type**：`instruction`（标准）/ `cot`（思维链）/ `classification`（分类）/ `text_completion`（续写）\n\n**常用训练参数**：\n\n| 参数 | 默认 | 调整 |\n|---|---|---|\n| `num_epochs` | 3 | 数据少用3，数据多用1-2 |\n| `lora_r` | 16 | 显存紧张降到8 |\n| `max_seq_length` | 2048 | 显存紧张降到1024 |\n"]))

cells.append(code([
    "import yaml\n\n# 加载 config.yaml\nconfig = yaml.safe_load(open('config.yaml', encoding='utf-8'))\n\n# ═══════════════════════════════════════════════════════════════\n# 【必须修改】以下三项根据你的环境填写\n# ═══════════════════════════════════════════════════════════════\n\n# ① 模型路径：HuggingFace ID（自动下载）或本地目录\nconfig['model']['name_or_path'] = 'Qwen/Qwen3-8B'\n\n# ② 数据集目录：放入 json/jsonl/csv/tsv/txt 文件\nconfig['data']['dataset_dir'] = './datasets'\n\n# ③ 任务类型：instruction / cot / classification / text_completion\nconfig['direction']['task_type'] = 'instruction'\n\n# ═══════════════════════════════════════════════════════════════\n# 【可选修改】\n# ═══════════════════════════════════════════════════════════════\n\n# 训练轮数\nconfig['training']['num_epochs'] = 3\n\n# LoRA rank（越大效果越好，显存占用越高）\nconfig['qlora']['r'] = 16\n\n# 最大序列长度（显存紧张用2048，8GB+可用4096）\nconfig['training']['max_seq_length'] = 2048\n\n# 启用 Unsloth 加速（推荐开启，2x 加速）\nconfig['unsloth']['enabled'] = True\n\n# WandB 云日志（可选）\n# config['wandb']['enabled'] = True\n# config['wandb']['project'] = 'my-llm-finetune'\n\n# 打印配置确认\nprint('=' * 50)\nprint('当前配置确认')\nprint('=' * 50)\nfor key in ['model', 'data', 'direction', 'training', 'qlora', 'unsloth']:\n    if key in config:\n        print('{}: {}'.format(key, config[key]))\nprint('=' * 50)\n"]))

# === CELL 2 ===
cells.append(md(["---\n\n## Stage 2: GPU 检测与显存安全检查\n\n**作用**：检测 GPU 型号、显存、BF16 支持，给出推荐训练参数\n\n**BF16 Support: No** = RTX 3060 及以下，自动降级为 FP16\n\n**显存不足警告**：降低 max_seq_length / batch_size / LoRA rank\n"]))

cells.append(code([
    "from src.gpu_detector import print_gpu_summary, GPUInfo\n\n# 打印 GPU 信息\ngpu_info = print_gpu_summary()\n\n# 获取推荐训练参数\noptimal = gpu_info.get_optimal_config()\nprint()\nprint('GPU 推荐训练参数:')\nfor k, v in optimal.items():\n    print('  {}: {}'.format(k, v))\n\n# 显存安全检查（填入你的模型参数量，单位：B）\nMODEL_PARAMS_B = 8.0  # Qwen3-8B → 8.0, Qwen3-4B → 4.0\nprint()\nprint('显存安全检查: {}B 模型'.format(MODEL_PARAMS_B))\nif gpu_info.check_memory_safe(MODEL_PARAMS_B):\n    print('  [OK] 显存估算通过')\nelse:\n    print('  [警告] 显存可能不足！')\n    print('    1. 降低 max_seq_length（2048 → 1024）')\n    print('    2. 降低 batch_size（2 → 1）')\n    print('    3. 降低 LoRA rank（16 → 8）')\n"]))

# === CELL 3 ===
cells.append(md(["---\n\n## Stage 3: 数据加载与格式检测\n\n**作用**：扫描目录，自动识别格式，打印样本预览\n\n**支持格式（自动检测，无需指定）**：\n\n| 格式 | 必需字段 |\n|---|---|\n| Alpaca | `instruction` + `output` |\n| ShareGPT | `conversations` |\n| ChatML | `messages` |\n| CoT | `Question` + `Complex_CoT` + `Response` |\n| 通用 JSON | 任意含 `prompt/response/content` |\n\n**数据文件放置**：`./datasets/my_data.jsonl` 等\n\n**此 Cell 可跳过**：数据不变时不需要重复运行\n"]))

cells.append(code([
    "from src.data_loader import DatasetLoader, print_dataset_preview\n\n# 预览数据集\nprint('=' * 50)\nprint('数据集预览（扫描 {}）'.format(config['data']['dataset_dir']))\nprint('=' * 50)\ntry:\n    print_dataset_preview(config['data']['dataset_dir'])\nexcept Exception as e:\n    print('预览失败: {}'.format(e))\n    print('请确认: 1) dataset_dir 路径正确  2) 存在 json/jsonl/csv/tsv/txt 文件')\n\n# 完整加载\nprint()\nprint('=' * 50)\nprint('开始完整加载数据...')\nprint('=' * 50)\n\nloader = DatasetLoader(config)\ntrain_data, val_data, detected_fmt = loader.load()\n\nprint()\nprint('检测到的数据格式:', detected_fmt)\nprint('训练集样本数:', len(train_data))\nprint('验证集样本数:', len(val_data))\n\n# 预览第一个样本\nif train_data:\n    text = train_data[0].get('text', '')\n    print()\n    print('第一个训练样本（前 500 字符）:')\n    print('-' * 50)\n    print(text[:500])\n    print('-' * 50)\n"]))

# === CELL 4 ===
cells.append(md(["---\n\n## Stage 4: 模型加载\n\n**作用**：加载基座模型 + Tokenizer，应用 QLoRA 4bit 量化\n\n**显存占用（8B 模型 QLoRA 4bit）**：\n- 基座模型（4bit）：约 4 GB\n- LoRA 权重：约 0.5 GB\n- 训练激活值：约 2-3 GB（取决于 max_seq_length）\n- **总计：约 8 GB（RTX 4060 8GB 刚好）**\n\n**首次运行会下载模型**（约 16GB），耐心等待。\n\n**此 Cell 可跳过**：模型不变时不需要重复运行\n"]))

cells.append(code([
    "from src.trainer import load_model\n\nprint('开始加载模型（首次可能需要几分钟下载）...')\nprint('模型路径:', config['model']['name_or_path'])\nprint()\n\n# load_model() 自动完成: Unsloth判断 + 基座加载 + QLoRA量化\nmodel, tokenizer, use_unsloth = load_model(\n    config,\n    gpu_info.get_optimal_config()\n)\n\nprint()\nprint('=' * 50)\nprint('模型加载完成!')\nprint('  Unsloth 加速:', use_unsloth)\nprint('  模型设备:', next(model.parameters()).device)\nprint('  Tokenizer vocab size:', tokenizer.vocab_size)\nprint('=' * 50)\n"]))

# === CELL 5 ===
cells.append(md(["---\n\n## Stage 5: 训练\n\n**作用**：运行 QLoRA 微调训练\n\n**支持**：EarlyStopping（连续3次eval_loss无改善自动停止）+ WandB 日志\n\n**输出**：`outputs/` 目录保存最终模型\n\n**中断**：`Ctrl+C` 可安全中断，已保存的 checkpoint 不会丢失\n\n**恢复训练**：`config['training']['resume_from_checkpoint'] = 'outputs/checkpoint-100'`\n"]))

cells.append(code([
    "from src.trainer import train\n\n# ！！！运行此 Cell 会从头开始训练 ！！！\n# 如需修改参数，先修改 Cell 1 的 config，再运行此 Cell\n\nprint('=' * 50)\nprint('开始训练!')\nprint('=' * 50)\nprint('  模型:', config['model']['name_or_path'])\nprint('  训练轮数:', config['training']['num_epochs'])\nprint('  LoRA rank:', config['qlora']['r'])\nprint('  Unsloth:', config['unsloth']['enabled'])\nprint('  输出目录:', config['output']['output_dir'])\nprint()\nprint('Ctrl+C 可安全中断（已保存的 checkpoint 不会丢失）')\nprint()\n\n# 开始训练\nfinal_model_dir = train(config, gpu_info.get_optimal_config())\n\nprint()\nprint('=' * 50)\nprint('训练完成! 模型保存路径:', final_model_dir)\nprint('下一步: Cell 6 (评估) 或 Cell 8 (导出)')\nprint('=' * 50)\n"]))

# === CELL 6 ===
cells.append(md(["---\n\n## Stage 6: 评估 + 推理性能监控\n\n**Part A**：从训练日志解析 loss，生成 `training_report.png`\n\n**Part B**：计算 BERTScore + Rouge（中文 jieba 分词）\n\n**Part C**：样例推理 + 每条打印详细性能报告\n\n---\n\n### 性能指标说明\n\n| 指标 | 说明 | 评判 |\n|---|---|---|\n| First Token Latency | 首个 token 生成时间 | <0.5s 优秀 |\n| Pure Generation Speed | 排除首 Token 的速度 | >30 优秀，>20 良好 |\n| GPU Memory Peak | 显存峰值 | <6GB 优秀，<8GB 良好 |\n| GPU Temp | GPU 温度 | <70°C 正常 |\n| GPU Utilization | GPU 利用率 | >85% 优秀 |\n\n---\n\n### 速度评级（Pure Generation Speed）\n\n| 速度 | 评级 |\n|---|---|\n| >30 tokens/s | 优秀 ✅ |\n| 20-30 | 良好 🟢 |\n| 15-20 | 中等 🟡 |\n| 10-15 | 偏低 🟠 |\n| <10 | 需优化 🔴 |\n"]))

cells.append(code([
    "from src.evaluator import evaluate_training, generate_sample_output\n",
    "from transformers import AutoTokenizer\n",
    "import torch\n",
    "\n",
    "# 加载 tokenizer\n",
    "MODEL_PATH = 'outputs/final_model'  # 训练完成后的模型路径\n",
    "\n",
    "print('加载 tokenizer from:', MODEL_PATH)\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# Part A: 训练过程评估（loss曲线 + BERTScore + Rouge）\n",
    "# ═══════════════════════════════════════════════════════\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('Part A: 训练过程评估')\n",
    "print('=' * 60)\n",
    "\n",
    "eval_result = evaluate_training(config, MODEL_PATH)\n",
    "\n",
    "print()\n",
    "print('报告: outputs/training_report.png')\n",
    "print('JSON: outputs/evaluation_results.json')\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# Part B: 推理测试 + 性能监控\n",
    "# ═══════════════════════════════════════════════════════\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('Part B: 推理性能监控')\n",
    "print('=' * 60)\n",
    "\n",
    "# 运行 2 条样本推理，每条打印详细性能报告\n",
    "sample_results = generate_sample_output(\n",
    "    model_path=MODEL_PATH,\n",
    "    tokenizer=tokenizer,\n",
    "    num_samples=2,\n",
    "    max_length=512,\n",
    "    system_prompt=config['direction']['system_prompt'],\n",
    ")\n",
    "\n",
    "print()\n",
    "print('=' * 60)\n",
    "print('评估完成!')\n",
    "print('下一步: Cell 8 (导出) 或 Cell 7 (自定义推理测试)')\n",
    "print('=' * 60)\n",
]))

# === CELL 7 ===
cells.append(md(["---\n\n## Stage 7: 独立推理测试（完全可自定义）\n\n**作用**：直接加载模型进行推理，随时可跑\n\n**使用场景**：训练过程中测试 / 导出后验证质量 / 对比不同 checkpoint\n\n---\n\n### 可调参数\n\n| 参数 | 说明 | 推荐值 |\n|---|---|---|\n| `TEST_PROMPT` | 测试问题 | 改成你的场景 |\n| `MAX_NEW_TOKENS` | 最大生成长度 | 256-1024 |\n| `TEMPERATURE` | 随机性 | 0.1（确定）~ 0.9（随机） |\n| `top_p` | 采样策略 | 0.9 |\n"]))

cells.append(code([
    "from transformers import (\n",
    "    AutoModelForCausalLM,\n",
    "    AutoTokenizer,\n",
    "    BitsAndBytesConfig,\n",
    "    TextIteratorStreamer,\n",
    ")\n",
    "from src.performance_monitor import (\n",
    "    AdvancedPerformanceMonitor,\n",
    "    print_detailed_performance_report,\n",
    ")\n",
    "from threading import Thread\n",
    "import time\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 【修改这里】测试配置\n",
    "# ═══════════════════════════════════════════════════════\n",
    "\n",
    "MODEL_PATH = 'outputs/final_model'               # 已训练模型路径\n",
    "TEST_PROMPT = '请介绍一下人工智能的发展历史。'   # ← 改成你的问题\n",
    "MAX_NEW_TOKENS = 512                           # 最大生成长度\n",
    "TEMPERATURE = 0.7                              # 随机性（0.1~0.9）\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 加载模型（4bit QLoRA 量化）\n",
    "# ═══════════════════════════════════════════════════════\n",
    "print('加载模型:', MODEL_PATH)\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_compute_dtype=torch.float16,\n",
    "    bnb_4bit_quant_type='nf4',\n",
    ")\n",
    "\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_PATH,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map='auto',\n",
    "    trust_remote_code=True,\n",
    ")\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)\n",
    "\n",
    "# 尝试启用 Unsloth 推理模式（2x 加速）\n",
    "try:\n",
    "    from unsloth import FastLanguageModel\n",
    "    FastLanguageModel.for_inference(model)\n",
    "    print('Unsloth 推理模式已启用')\n",
    "except ImportError:\n",
    "    print('使用标准 HuggingFace 推理模式')\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 准备输入\n",
    "# ═══════════════════════════════════════════════════════\n",
    "messages = [\n",
    "    {'role': 'system', 'content': config['direction']['system_prompt']},\n",
    "    {'role': 'user', 'content': TEST_PROMPT},\n",
    "]\n",
    "\n",
    "# 使用 chat template（与训练时的格式完全一致）\n",
    "text = tokenizer.apply_chat_template(\n",
    "    messages,\n",
    "    tokenize=False,\n",
    "    add_generation_prompt=True,\n",
    ")\n",
    "\n",
    "inputs = tokenizer(text, return_tensors='pt').to(model.device)\n",
    "input_token_count = inputs.input_ids.shape[1]\n",
    "print('输入 Token 数:', input_token_count)\n",
    "print('生成中...')\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 性能监控初始化\n",
    "# ═══════════════════════════════════════════════════════\n",
    "monitor = AdvancedPerformanceMonitor()\n",
    "monitor.tokenizer = tokenizer\n",
    "monitor.start()\n",
    "\n",
    "# 流式输出器\n",
    "streamer = TextIteratorStreamer(\n",
    "    tokenizer,\n",
    "    skip_prompt=True,\n",
    "    skip_special_tokens=True,\n",
    ")\n",
    "\n",
    "# 后台线程：每 0.1 秒采样一次 GPU/CPU 状态\n",
    "def background_monitor():\n",
    "    while monitor.monitoring:\n",
    "        monitor.collect_snapshot()\n",
    "        time.sleep(0.1)\n",
    "\n",
    "monitor_thread = Thread(target=background_monitor, daemon=True)\n",
    "monitor_thread.start()\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 启动推理\n",
    "# ═══════════════════════════════════════════════════════\n",
    "generation_kwargs = dict(\n",
    "    input_ids=inputs.input_ids,\n",
    "    attention_mask=inputs.attention_mask,\n",
    "    max_new_tokens=MAX_NEW_TOKENS,\n",
    "    temperature=TEMPERATURE,\n",
    "    top_p=0.9,\n",
    "    use_cache=True,\n",
    "    streamer=streamer,\n",
    "    do_sample=True,\n",
    ")\n",
    "\n",
    "monitor.start_inference_timing(input_token_count)\n",
    "thread = Thread(target=model.generate, kwargs=generation_kwargs)\n",
    "thread.start()\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 流式收集并打印输出\n",
    "# ═══════════════════════════════════════════════════════\n",
    "first_token_received = False\n",
    "generated_text = ''\n",
    "\n",
    "print('-' * 50)\n",
    "print('模型回答:')\n",
    "for new_text in streamer:\n",
    "    # 记录首个 token 的时间\n",
    "    if not first_token_received:\n",
    "        monitor.record_first_token()\n",
    "        first_token_received = True\n",
    "    # 记录每个 token 的时间戳\n",
    "    monitor.record_token_generation(new_text)\n",
    "    # 实时打印\n",
    "    print(new_text, end='', flush=True)\n",
    "    generated_text += new_text\n",
    "\n",
    "thread.join()\n",
    "monitor.generation_end_time = time.time()\n",
    "monitor.stop()\n",
    "monitor_thread.join(timeout=2)\n",
    "\n",
    "# 计算并打印性能报告\n",
    "monitor.output_tokens = len(tokenizer.encode(generated_text))\n",
    "speeds = monitor.calculate_speeds()\n",
    "summary = monitor.get_summary()\n",
    "memory_analysis = monitor.get_memory_analysis()\n",
    "\n",
    "print()\n",
    "print('-' * 50)\n",
    "print_detailed_performance_report(speeds, summary, memory_analysis)\n",
    "\n",
    "del model\n",
    "torch.cuda.empty_cache()\n",
]))

# === CELL 8 ===
cells.append(md(["---\n\n## Stage 8: 导出模型\n\n**作用**：将微调后的模型导出为可部署格式\n\n| 格式 | 说明 | 适用场景 |\n|---|---|---|\n| **HuggingFace safetensors** | 标准模型格式 | 直接 `AutoModelForCausalLM.from_pretrained()` 加载 |\n| **GGUF** | 量化格式 | Ollama 本地部署，Q4_K_M 推荐的平衡精度/大小 |\n\n**导出路径**：默认 `outputs/hf/` 和 `outputs/gguf/`\n\n**GGUF 推荐量化级别**：\n- `Q4_K_M`：平衡（推荐），精度接近 FP16，大小约 4.8GB\n- `Q5_K_S`：更高精度，稍大\n- `Q8_0`：几乎无损，但较大\n"]))

cells.append(code([
    "from src.exporter import export_model\n",
    "\n",
    "# ═══════════════════════════════════════════════════════\n",
    "# 【修改这里】导出的模型路径和格式\n",
    "# ═══════════════════════════════════════════════════════\n",
    "\n",
    "MODEL_TO_EXPORT = 'outputs/final_model'   # 训练完成后的模型路径\n",
    "\n",
    "# 导出格式: 'hf' (HuggingFace) / 'gguf' (Ollama) / 'both' (两者都要)\n",
    "config['export']['format'] = 'both'\n",
    "\n",
    "# GGUF 量化级别: Q4_K_M (推荐) / Q5_K_S / Q8_0 / F16\n",
    "config['export']['gguf_quant'] = 'Q4_K_M'\n",
    "\n",
    "print('=' * 50)\n",
    "print('导出模型: {}'.format(MODEL_TO_EXPORT))\n",
    "print('导出格式: {}'.format(config['export']['format']))\n",
    "print('=' * 50)\n",
    "\n",
    "results = export_model(config, MODEL_TO_EXPORT)\n",
    "\n",
    "print()\n",
    "print('导出完成!')\n",
    "for fmt, path in results.items():\n",
    "    print('  {}: {}'.format(fmt, path))\n",
    "print()\n",
    "print('GGUF 模型可直接用于 Ollama:')\n",
    "print('  ollama create -f Modelfile my-model-name')\n",
]))

# === BUILD NOTEBOOK ===
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

out = r'C:\Users\OK bandage\Desktop\llm-finetune\llm_finetune_trainer.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook written to:', out)
print('Cells:', len(cells))
