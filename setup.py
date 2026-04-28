"""
环境配置脚本 — python setup.py
自动检测 CUDA 版本，安装正确版本的 PyTorch 和依赖
"""
import subprocess
import sys
import os


def run(cmd):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0 and result.stderr:
        print(f"[ERROR] {result.stderr.strip()}")
    return result.returncode == 0


def detect_cuda():
    """检测 CUDA 版本"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            # 从 nvidia-smi 输出提取 CUDA 版本
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0]
                        print(f"Detected CUDA: {version}")
                        return version
    except FileNotFoundError:
        pass
    print("No CUDA detected, will install CPU-only PyTorch")
    return None


def get_torch_install_cmd(cuda_version):
    """根据 CUDA 版本选择 PyTorch 安装命令"""
    if cuda_version is None:
        # CPU only
        return "pip install torch --index-url https://download.pytorch.org/whl/cpu"

    major = float(".".join(cuda_version.split(".")[:2]))
    if major >= 12.6:
        return "pip install torch --index-url https://download.pytorch.org/whl/cu126"
    elif major >= 12.4:
        return "pip install torch --index-url https://download.pytorch.org/whl/cu124"
    elif major >= 12.1:
        return "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    elif major >= 11.8:
        return "pip install torch --index-url https://download.pytorch.org/whl/cu118"
    else:
        print(f"CUDA {cuda_version} is too old. Using default PyTorch.")
        return "pip install torch"


def main():
    print("=" * 60)
    print("  LLM Fine-Tune Environment Setup")
    print("=" * 60)

    # 1. 检测 CUDA
    cuda_version = detect_cuda()

    # 2. 安装 PyTorch
    print("\n[1/3] Installing PyTorch...")
    torch_cmd = get_torch_install_cmd(cuda_version)
    run(torch_cmd)

    # 3. 安装其他依赖
    print("\n[2/3] Installing dependencies...")
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    # 跳过 requirements.txt 中的 torch（已单独安装）
    with open(req_file, encoding="utf-8") as f:
        deps = [line.strip() for line in f if line.strip() and not line.strip().startswith("torch")]
    dep_str = " ".join(f'"{d}"' for d in deps)
    run(f"pip install {dep_str}")

    # 4. 验证安装
    print("\n[3/3] Verifying installation...")
    checks = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "trl": "trl",
        "pyyaml": "yaml",
        "tensorboard": "tensorboard",
        "matplotlib": "matplotlib",
    }

    all_ok = True
    for name, import_name in checks.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {name}: {version}")
        except ImportError:
            print(f"  {name}: MISSING")
            all_ok = False

    # 检查 CUDA
    import torch
    print(f"\n  PyTorch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if all_ok:
        print("\n" + "=" * 60)
        print("  Setup complete! You can now run: python train.py")
        print("=" * 60)
    else:
        print("\n[WARNING] Some dependencies are missing. Try:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
