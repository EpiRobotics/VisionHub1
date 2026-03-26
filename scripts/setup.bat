@echo off
REM ============================================================
REM VisionHub - One-click environment setup
REM Run this on a new machine to set up the Python environment.
REM Supports both OFFLINE (from offline_packages folder) and
REM ONLINE (download from internet) installation.
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo  VisionHub Environment Setup
echo ============================================================
echo.

REM Determine script directory and root
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "PYTHON_SERVICE=%ROOT_DIR%\python_service"
set "VENV_DIR=%PYTHON_SERVICE%\.venv"
set "PKG_DIR=%ROOT_DIR%\offline_packages"

echo Root directory:    %ROOT_DIR%
echo Python service:    %PYTHON_SERVICE%
echo Virtual env:       %VENV_DIR%
echo.

REM ---- Step 1: Check Python ----
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10 first:
    echo   1. Download from https://www.python.org/downloads/
    echo   2. During installation, CHECK "Add Python to PATH"
    echo   3. Re-run this setup.bat
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo   Python %PY_VER% found.
echo.

REM ---- Step 2: Create venv ----
echo [2/4] Creating virtual environment...
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   Virtual environment already exists. Skipping creation.
) else (
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
)
echo.

REM ---- Step 3: Activate venv ----
echo [3/4] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo   Activated.
echo.

REM ---- Step 4: Install packages ----
echo [4/4] Installing Python packages...

if exist "%PKG_DIR%" (
    echo   Found offline_packages folder. Installing OFFLINE...
    echo.

    REM Install PyTorch first (CUDA 11.8)
    echo   Installing PyTorch...
    pip install --no-index --find-links="%PKG_DIR%" torch torchvision 2>nul
    if errorlevel 1 (
        echo   [WARNING] Some PyTorch packages may need online download.
        echo   Trying online for PyTorch...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    )

    REM Install remaining packages
    echo   Installing remaining packages...
    pip install --no-index --find-links="%PKG_DIR%" fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml 2>nul
    pip install --no-index --find-links="%PKG_DIR%" numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless 2>nul

) else (
    echo   No offline_packages folder found. Installing ONLINE...
    echo   (This requires internet access)
    echo.

    echo   Installing PyTorch (CUDA 11.8)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch. Check internet connection.
        pause
        exit /b 1
    )

    echo   Installing remaining packages...
    pip install fastapi "uvicorn[standard]" pydantic pydantic-settings pyyaml
    pip install numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  You can now launch VisionHub by double-clicking VisionHubUI.exe
echo  The Python backend will start automatically.
echo ============================================================
pause
