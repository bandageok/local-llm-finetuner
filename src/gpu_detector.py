"""GPU 自动检测与训练参数适配"""
import torch
import subprocess
import re
import logging

logger = logging.getLogger(__name__)

# GPU 显存需求参考表（QLoRA 4-bit）
# 模型参数量 -> 建议最大 batch_size, max_seq_length, gradient_accumulation
VRAM_TABLE = {
    4:  {"max_seq": 4096, "batch": 2, "grad_accum": 4, "rank": 8},
    7:  {"max_seq": 2048, "batch": 2, "grad_accum": 4, "rank": 16},
    8:  {"max_seq": 2048, "batch": 2, "grad_accum": 4, "rank": 16},
    13: {"max_seq": 2048, "batch": 1, "grad_accum": 8, "rank": 8},
    14: {"max_seq": 2048, "batch": 1, "grad_accum": 8, "rank": 8},
    32: {"max_seq": 1024, "batch": 1, "grad_accum": 16, "rank": 4},
}

# CUDA Compute Capability 对应版本
COMPUTE_TABLE = {
    (3, 0): "Kepler", (3, 5): "Kepler", (3, 7): "Kepler",
    (5, 0): "Maxwell", (5, 2): "Maxwell",
    (6, 0): "Pascal", (6, 1): "Pascal",
    (7, 0): "Volta", (7, 5): "Turing",
    (8, 0): "Ampere", (8, 6): "Ampere", (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
}


class GPUInfo:
    """GPU 信息"""

    def __init__(self):
        self.available = torch.cuda.is_available()
        self.name = ""
        self.vram_gb = 0.0
        self.vram_mb = 0
        self.cuda_version = ""
        self.compute_capability = (0, 0)
        self.architecture = "unknown"
        self.gpu_count = 0
        self.bf16_supported = False

        if self.available:
            self._detect()

    def _detect(self):
        self.gpu_count = torch.cuda.device_count()
        self.name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        self.vram_mb = props.total_memory // (1024 * 1024)
        self.vram_gb = self.vram_mb / 1024
        self.compute_capability = (props.major, props.minor)
        self.architecture = COMPUTE_TABLE.get(
            (props.major, props.minor), f"SM{props.major}{props.minor}"
        )
        self.cuda_version = torch.version.cuda or "unknown"

        # BF16 support (Ampere+)
        cc = props.major * 10 + props.minor
        self.bf16_supported = cc >= 80

    def to_dict(self):
        return {
            "gpu_name": self.name,
            "vram_gb": round(self.vram_gb, 1),
            "cuda_version": self.cuda_version,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "architecture": self.architecture,
            "bf16_supported": self.bf16_supported,
            "gpu_count": self.gpu_count,
        }

    def get_optimal_config(self, model_param_b: float = 8.0) -> dict:
        """根据 GPU 信息推荐最优训练配置"""
        config = {
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 2048,
            "lora_r": 16,
            "gradient_checkpointing": True,
            "bf16": self.bf16_supported,
            "fp16": not self.bf16_supported,
        }

        if not self.available:
            logger.warning("No GPU detected, using CPU defaults")
            return config

        # VRAM-based adjustment
        # QLoRA 4-bit: ~2GB per billion params + overhead
        # 8GB VRAM -> ~2GB overhead, ~6GB usable
        usable_gb = self.vram_gb - 2  # Reserve 2GB for overhead

        if self.vram_gb >= 24:
            config["batch_size"] = 4
            config["gradient_accumulation_steps"] = 2
            config["max_seq_length"] = 4096
            config["lora_r"] = 32
        elif self.vram_gb >= 16:
            config["batch_size"] = 2
            config["gradient_accumulation_steps"] = 4
            config["max_seq_length"] = 2048
            config["lora_r"] = 16
        elif self.vram_gb >= 12:
            config["batch_size"] = 1
            config["gradient_accumulation_steps"] = 8
            config["max_seq_length"] = 2048
            config["lora_r"] = 16
        elif self.vram_gb >= 8:
            config["batch_size"] = 1
            config["gradient_accumulation_steps"] = 4
            config["max_seq_length"] = 2048
            config["lora_r"] = 16
        elif self.vram_gb >= 6:
            config["batch_size"] = 1
            config["gradient_accumulation_steps"] = 8
            config["max_seq_length"] = 1024
            config["lora_r"] = 8
        else:
            logger.warning(f"Only {self.vram_gb:.1f}GB VRAM, may not be sufficient")
            config["batch_size"] = 1
            config["gradient_accumulation_steps"] = 16
            config["max_seq_length"] = 512
            config["lora_r"] = 4

        logger.info(f"GPU: {self.name} ({self.vram_gb:.1f}GB)")
        logger.info(f"Recommended: batch={config['batch_size']}, "
                     f"grad_accum={config['gradient_accumulation_steps']}, "
                     f"max_seq={config['max_seq_length']}, "
                     f"rank={config['lora_r']}")

        return config

    def check_memory_safe(self, model_param_b: float = 8.0) -> bool:
        """检查显存是否足够"""
        # Rough estimate: 4-bit model needs ~0.5GB per billion params
        # Plus LoRA overhead, optimizer states, activations
        estimated = model_param_b * 0.5 + 2  # GB
        if estimated > self.vram_gb:
            logger.error(
                f"Estimated memory ({estimated:.1f}GB) exceeds VRAM ({self.vram_gb:.1f}GB). "
                f"Consider using a smaller model."
            )
            return False
        return True


def print_gpu_summary():
    """打印 GPU 信息摘要"""
    info = GPUInfo()
    if info.available:
        print(f"\n{'='*50}")
        print(f"  GPU: {info.name}")
        print(f"  VRAM: {info.vram_gb:.1f} GB")
        print(f"  CUDA: {info.cuda_version}")
        print(f"  Architecture: {info.architecture} (CC {info.compute_capability[0]}.{info.compute_capability[1]})")
        print(f"  BF16 Support: {'Yes' if info.bf16_supported else 'No'}")
        print(f"{'='*50}\n")
    else:
        print("\n[WARNING] No NVIDIA GPU detected. Training will be extremely slow on CPU.\n")
    return info


if __name__ == "__main__":
    info = print_gpu_summary()
    cfg = info.get_optimal_config()
    print(f"Recommended config: {cfg}")
