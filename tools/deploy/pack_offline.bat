@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ============================================================================
:: VisionHub1 Offline Package Builder
:: Run this on a PC WITH internet to download all dependencies.
:: Output: a self-contained folder ready to copy to the target machine.
:: ============================================================================

echo ============================================================
echo   VisionHub1 Offline Package Builder
echo ============================================================
echo.

:: ---------------------------------------------------------------------------
:: Configuration
:: ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
:: Navigate up two levels: tools\deploy\ -> VisionHub1 root
for %%I in ("%SCRIPT_DIR%\..\..\") do set "REPO_ROOT=%%~fI"

set "OUTPUT_DIR=%REPO_ROOT%VisionHub1_offline"
set "PKG_DIR=%OUTPUT_DIR%\packages"
set "PYTHON_VERSION=3.10.11"
set "PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%"

:: ---------------------------------------------------------------------------
:: Detect CUDA version
:: ---------------------------------------------------------------------------
echo [1/5] Detecting CUDA version...
set "CUDA_TAG=cu118"
where nvcc >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=5 delims= " %%v in ('nvcc --version ^| findstr /i "release"') do (
        set "NVCC_VER=%%v"
    )
    echo   Found CUDA: !NVCC_VER!
    echo !NVCC_VER! | findstr /i "12." >nul && set "CUDA_TAG=cu121"
    echo !NVCC_VER! | findstr /i "11." >nul && set "CUDA_TAG=cu118"
) else (
    echo   CUDA not found, defaulting to cu118.
    echo   If the target machine has CUDA 12.x, edit CUDA_TAG in this script.
)
echo   Using PyTorch index: %CUDA_TAG%
echo.

:: ---------------------------------------------------------------------------
:: Create output directory
:: ---------------------------------------------------------------------------
echo [2/5] Creating output directory: %OUTPUT_DIR%
if exist "%OUTPUT_DIR%" (
    echo   WARNING: Output directory already exists. Files may be overwritten.
)
mkdir "%OUTPUT_DIR%" 2>nul
mkdir "%PKG_DIR%" 2>nul
echo.

:: ---------------------------------------------------------------------------
:: Download Python installer
:: ---------------------------------------------------------------------------
echo [3/5] Downloading Python %PYTHON_VERSION% installer...
if exist "%OUTPUT_DIR%\%PYTHON_INSTALLER%" (
    echo   Already downloaded, skipping.
) else (
    echo   URL: %PYTHON_URL%
    curl -L -o "%OUTPUT_DIR%\%PYTHON_INSTALLER%" "%PYTHON_URL%"
    if !errorlevel! neq 0 (
        echo   ERROR: Failed to download Python installer.
        echo   Please download manually from: %PYTHON_URL%
        echo   and place it in: %OUTPUT_DIR%\
    ) else (
        echo   OK
    )
)
echo.

:: ---------------------------------------------------------------------------
:: Download PyTorch + torchvision wheels
:: ---------------------------------------------------------------------------
echo [4/5] Downloading Python packages (this may take 10+ minutes)...
echo   Downloading PyTorch + torchvision (%CUDA_TAG%)...
pip download torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_TAG% -d "%PKG_DIR%" --python-version 3.10 --only-binary=:all: --platform win_amd64
if !errorlevel! neq 0 (
    echo   WARNING: PyTorch download may have failed. Check the packages folder.
)

echo.
echo   Downloading other dependencies...
pip download fastapi "uvicorn[standard]" pyyaml pydantic pydantic-settings numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless faiss-cpu -d "%PKG_DIR%" --python-version 3.10 --only-binary=:all: --platform win_amd64
if !errorlevel! neq 0 (
    echo   WARNING: Some packages may have failed. Retrying without binary constraint...
    pip download fastapi "uvicorn[standard]" pyyaml pydantic pydantic-settings numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless faiss-cpu -d "%PKG_DIR%" --python-version 3.10 --platform win_amd64
)
echo.

:: ---------------------------------------------------------------------------
:: Copy VisionHub1 source code
:: ---------------------------------------------------------------------------
echo [5/5] Copying VisionHub1 source code...
set "CODE_DEST=%OUTPUT_DIR%\VisionHub1"
if exist "%CODE_DEST%" rmdir /s /q "%CODE_DEST%"
mkdir "%CODE_DEST%"

:: Copy relevant directories (skip .git, .ruff_cache, __pycache__)
robocopy "%REPO_ROOT%python_service" "%CODE_DEST%\python_service" /e /xd .git .ruff_cache __pycache__ .venv >nul
robocopy "%REPO_ROOT%config_templates" "%CODE_DEST%\config_templates" /e >nul
robocopy "%REPO_ROOT%tools" "%CODE_DEST%\tools" /e /xd __pycache__ >nul
if exist "%REPO_ROOT%csharp_ui" (
    robocopy "%REPO_ROOT%csharp_ui" "%CODE_DEST%\csharp_ui" /e /xd bin obj .vs >nul
)
if exist "%REPO_ROOT%README.md" copy "%REPO_ROOT%README.md" "%CODE_DEST%\" >nul

:: Copy the installer script
copy "%SCRIPT_DIR%install.bat" "%OUTPUT_DIR%\install.bat" >nul

echo.
echo ============================================================
echo   Package complete!
echo ============================================================
echo.
echo   Output: %OUTPUT_DIR%
echo.
echo   Contents:
echo     %PYTHON_INSTALLER%     - Python installer
echo     packages\              - All Python wheel packages
echo     VisionHub1\            - Source code
echo     install.bat            - One-click installer
echo.
echo   Copy the entire '%OUTPUT_DIR%' folder to the target
echo   machine (USB drive, network share, etc.) and run install.bat
echo.
echo   NOTE: You also need to copy your project data separately:
echo     E:\AIInspect\projects\  (project configs + trained models)
echo ============================================================

pause
