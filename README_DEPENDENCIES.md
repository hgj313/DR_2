# ============================================
# DR-2 项目依赖说明
# ============================================

## 当前状态 ✅

```bash
PyTorch: 2.6.0+cu124    ✅ GPU 可用
CUDA: True              ✅ 
GPU: RTX 3060 Laptop    ✅
```

## 如何正确运行项目

### 方法 1：直接使用 .venv 的 Python（推荐）

```bash
# 进入项目目录
cd c:\AA_GSE_DP\DR-2

# 直接运行（不要用 uv run！）
.\.venv\Scripts\python.exe main.py

# 或者运行测试
.\.venv\Scripts\python.exe test_sentence_transformers.py
```

### 方法 2：激活 .venv

```bash
cd c:\AA_GSE_DP\DR-2

# PowerShell
.\.venv\Scripts\Activate.ps1

# CMD
.\.venv\Scripts\activate.bat

# 然后直接运行
python main.py
```

## 重要提醒 ⚠️

### ❌ 不要使用
```bash
uv run python main.py        # 会重新安装依赖
uv pip install xxx           # 可能覆盖 torch 版本
```

### ✅ 应该使用
```bash
.\.venv\Scripts\python.exe main.py    # 直接用 .venv 的 Python
```

## 如果 PyTorch 变成了 CPU 版本

重新安装 CUDA 版本：

```bash
cd c:\AA_GSE_DP\DR-2

# 1. 卸载 CPU 版本
uv pip uninstall torch torchvision

# 2. 安装 CUDA 版本
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 验证
.\.venv\Scripts\python.exe verify_torch.py
```

应该显示：
```
PyTorch: 2.6.0+cu124
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

## 验证 GPU 可用

```bash
.\.venv\Scripts\python.exe verify_torch.py
```

## 快速启动

```bash
cd c:\AA_GSE_DP\DR-2
.\.venv\Scripts\python.exe main.py
```