# Datasets

将训练数据文件放入此目录，支持以下格式。

## 支持的文件类型

- `.json` / `.jsonl` — JSON 数组或 JSON Lines
- `.txt` — 纯文本（空行分段）
- `.csv` / `.tsv` — 逗号/制表符分隔

## 支持的数据格式

### 1. Alpaca 格式
```json
{
  "instruction": "解释量子计算",
  "input": "",
  "output": "量子计算是一种利用量子力学..."
}
```

### 2. ShareGPT 格式
```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！有什么可以帮你的？"},
    {"from": "human", "value": "介绍一下你自己"},
    {"from": "gpt", "value": "我是一个AI助手..."}
  ]
}
```

### 3. ChatML 格式
```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

### 4. 纯文本格式
```json
{"text": "这是一段训练文本内容。"}
```

或 `.txt` 文件（每段用空行分隔）：
```
这是第一段训练文本。

这是第二段训练文本。
```

### 5. CSV 格式
```csv
instruction,input,output
"解释量子计算","","量子计算是一种利用量子力学..."
"写一首诗","关于春天","春天来了..."
```

## 使用方法

1. 将数据文件放入此目录
2. 运行 `python train.py`
3. 系统会自动检测格式并训练
