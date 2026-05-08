# 快速修复 PyTorch CUDA 版本
# 直接执行此脚本

import subprocess
import sys

print("=" * 80)
print("修复 PyTorch CUDA 版本")
print("=" * 80)

# 1. 卸载当前版本
print("\n1️⃣ 卸载 CPU 版本...")
result = subprocess.run(
    [sys.executable, "-m", "uv", "pip", "uninstall", "torch", "torchvision", "-y"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)

# 2. 安装 CUDA 版本
print("\n2️⃣ 安装 CUDA 版本...")
result = subprocess.run(
    [sys.executable, "-m", "uv", "pip", "install", 
     "torch>=2.6.0", "torchvision>=0.21.0",
     "--index-url", "https://download.pytorch.org/whl/cu124"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)

# 3. 验证
print("\n3️⃣ 验证安装...")
result = subprocess.run(
    [sys.executable, "-c", 
     "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)

print("\n" + "=" * 80)
print("✅ 修复完成！现在可以运行 main.py 了")
print("=" * 80)