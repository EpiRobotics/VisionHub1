@echo off
REM ============================================================
REM VisionHub - Download offline packages for deployment
REM Run this on a machine WITH internet access.
REM It downloads all Python dependency wheels into offline_packages\
REM Then copy the entire VisionHub1 folder to the offline machine.
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  VisionHub Offline Package Downloader
echo ============================================================
echo.

REM Determine script directory (where this .bat lives)
set "SCRIPT_DIR=%~dp0"
REM Go up one level to VisionHub root
set "ROOT_DIR=%SCRIPT_DIR%.."
set "PKG_DIR=%ROOT_DIR%\offline_packages"

echo Root directory: %ROOT_DIR%
echo Package output: %PKG_DIR%
echo.

REM Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10 first.
    pause
    exit /b 1
)

REM Create output directory
if not exist "%PKG_DIR%" mkdir "%PKG_DIR%"

echo [1/3] Downloading PyTorch (CUDA 11.8)...
echo This may take a while (about 2.5 GB)...
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 --dest "%PKG_DIR%" --python-version 3.10 --platform win_amd64 --only-binary=:all:
if errorlevel 1 (
    echo [WARNING] PyTorch download may have issues. Retrying...
    pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 --dest "%PKG_DIR%" --python-version 3.10 --platform win_amd64 --only-binary=:all:
)

echo.
echo [2/3] Downloading FastAPI, Uvicorn, and core packages...
pip download fastapi uvicorn[standard] pydantic pydantic-settings pyyaml --dest "%PKG_DIR%" --python-version 3.10 --platform win_amd64 --only-binary=:all:

echo.
echo [3/3] Downloading ML and image processing packages...
pip download numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless --dest "%PKG_DIR%" --python-version 3.10 --platform win_amd64 --only-binary=:all:

echo.
echo ============================================================
echo  Download complete!
echo  Packages saved to: %PKG_DIR%
echo.
echo  Next steps:
echo  1. Copy the entire VisionHub1 folder to the offline machine
echo  2. On the offline machine, run setup.bat
echo ============================================================
pause
