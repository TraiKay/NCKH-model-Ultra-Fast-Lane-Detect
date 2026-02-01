@echo off
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"

echo Checking nvcc...
nvcc --version
if errorlevel 1 (
    echo Error: nvcc not found!
    exit /b 1
)

echo Building my_interp...
cd my_interp
uv run python setup.py install
if errorlevel 1 (
    echo Error: Build failed!
    exit /b 1
)

echo Build successful!
