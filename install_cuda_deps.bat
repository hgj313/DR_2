@echo off
REM ============================================
REM 首次使用 CUDA 版本时执行此脚本
REM ============================================

echo ============================================
echo 安装 PyTorch CUDA 版本
echo ============================================

echo.
echo 1. 卸载 CPU 版本...
call uv pip uninstall torch torchvision -y 2>nul

echo.
echo 2. 安装 CUDA 版本...
call uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo 3. 验证安装...
python -c "import torch; print(f'版本: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo ============================================
echo 安装完成！
echo ============================================
pause