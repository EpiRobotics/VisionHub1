@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ============================================================================
:: VisionHub1 One-Click Installer (Offline)
:: Run this on the TARGET machine (no internet required).
:: Expects the following files in the same directory:
::   python-3.10.11-amd64.exe   - Python installer
::   packages\                   - Wheel files from pack_offline.bat
::   VisionHub1\                 - Source code
:: ============================================================================

echo ============================================================
echo   VisionHub1 One-Click Installer
echo ============================================================
echo.

set "INSTALLER_DIR=%~dp0"
set "PYTHON_INSTALLER=%INSTALLER_DIR%python-3.10.11-amd64.exe"
set "PKG_DIR=%INSTALLER_DIR%packages"
set "SRC_DIR=%INSTALLER_DIR%VisionHub1"

:: ---------------------------------------------------------------------------
:: Default install paths (user can change these)
:: ---------------------------------------------------------------------------
set "INSTALL_DIR=D:\project\VisionHub1"
set "DATA_ROOT=E:\AIInspect"

:: ---------------------------------------------------------------------------
:: Prompt user for install paths
:: ---------------------------------------------------------------------------
echo   Default install paths:
echo     Code:    %INSTALL_DIR%
echo     Data:    %DATA_ROOT%
echo.
set /p "CONFIRM=Press ENTER to use defaults, or type 'c' to customize: "
if /i "!CONFIRM!"=="c" (
    set /p "INSTALL_DIR=Code install path [%INSTALL_DIR%]: " || set "INSTALL_DIR=%INSTALL_DIR%"
    set /p "DATA_ROOT=Data root path [%DATA_ROOT%]: " || set "DATA_ROOT=%DATA_ROOT%"
)
echo.
echo   Installing to: !INSTALL_DIR!
echo   Data root:     !DATA_ROOT!
echo.

:: ---------------------------------------------------------------------------
:: Step 1: Check / Install Python
:: ---------------------------------------------------------------------------
echo [1/6] Checking Python installation...
set "PYTHON_EXE="

:: Check if python 3.10 is already installed
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
    echo !PY_VER! | findstr /b "3.10" >nul
    if !errorlevel!==0 (
        echo   Python !PY_VER! found. Skipping installation.
        set "PYTHON_EXE=python"
        goto :python_ready
    )
)

:: Check common install locations
for %%P in (
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python310\python.exe"
) do (
    if exist %%P (
        echo   Found Python at %%P
        set "PYTHON_EXE=%%~P"
        goto :python_ready
    )
)

:: Install Python
if not exist "%PYTHON_INSTALLER%" (
    echo   ERROR: Python installer not found: %PYTHON_INSTALLER%
    echo   Please download python-3.10.11-amd64.exe and place it next to this script.
    pause
    exit /b 1
)
echo   Installing Python 3.10.11...
"%PYTHON_INSTALLER%" /passive InstallAllUsers=0 PrependPath=1 Include_test=0
if !errorlevel! neq 0 (
    echo   ERROR: Python installation failed.
    pause
    exit /b 1
)
echo   Python installed. You may need to restart this script for PATH to take effect.

:: Find the newly installed python
for %%P in (
    "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%P (
        set "PYTHON_EXE=%%~P"
        goto :python_ready
    )
)
set "PYTHON_EXE=python"

:python_ready
echo   Using Python: !PYTHON_EXE!
echo.

:: ---------------------------------------------------------------------------
:: Step 2: Copy source code
:: ---------------------------------------------------------------------------
echo [2/6] Copying VisionHub1 source code...
if not exist "%SRC_DIR%" (
    echo   ERROR: Source code not found at: %SRC_DIR%
    pause
    exit /b 1
)

if not exist "!INSTALL_DIR!" mkdir "!INSTALL_DIR!"
robocopy "%SRC_DIR%" "!INSTALL_DIR!" /e /xd .venv __pycache__ .ruff_cache >nul
echo   Code copied to: !INSTALL_DIR!
echo.

:: ---------------------------------------------------------------------------
:: Step 3: Create virtual environment
:: ---------------------------------------------------------------------------
echo [3/6] Creating Python virtual environment...
set "VENV_DIR=!INSTALL_DIR!\python_service\.venv"

if exist "!VENV_DIR!" (
    echo   Virtual environment already exists, reusing.
) else (
    "!PYTHON_EXE!" -m venv "!VENV_DIR!"
    if !errorlevel! neq 0 (
        echo   ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)
set "PIP=!VENV_DIR!\Scripts\pip.exe"
set "PYTHON_VENV=!VENV_DIR!\Scripts\python.exe"
echo   Virtual environment: !VENV_DIR!
echo.

:: ---------------------------------------------------------------------------
:: Step 4: Install packages from local wheels
:: ---------------------------------------------------------------------------
echo [4/6] Installing Python packages (offline)...
if not exist "%PKG_DIR%" (
    echo   ERROR: Packages directory not found: %PKG_DIR%
    pause
    exit /b 1
)

echo   Installing PyTorch + torchvision...
"!PIP!" install --no-index --find-links="%PKG_DIR%" torch torchvision 2>&1 | findstr /i "success error"
if !errorlevel! neq 0 (
    "!PIP!" install --no-index --find-links="%PKG_DIR%" torch torchvision
)

echo   Installing other dependencies...
"!PIP!" install --no-index --find-links="%PKG_DIR%" fastapi "uvicorn[standard]" pyyaml pydantic pydantic-settings numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless faiss-cpu 2>&1 | findstr /i "success error"
if !errorlevel! neq 0 (
    "!PIP!" install --no-index --find-links="%PKG_DIR%" fastapi "uvicorn[standard]" pyyaml pydantic pydantic-settings numpy Pillow matplotlib tqdm joblib scikit-learn opencv-python-headless faiss-cpu
)
echo   Packages installed.
echo.

:: ---------------------------------------------------------------------------
:: Step 5: Create data directory and .env
:: ---------------------------------------------------------------------------
echo [5/6] Setting up data directory and configuration...
if not exist "!DATA_ROOT!" mkdir "!DATA_ROOT!"
if not exist "!DATA_ROOT!\projects" mkdir "!DATA_ROOT!\projects"

:: Create .env file for the Python service
set "ENV_FILE=!INSTALL_DIR!\python_service\.env"
(
    echo DATA_ROOT=!DATA_ROOT!
) > "!ENV_FILE!"
echo   Created: !ENV_FILE!
echo.

:: ---------------------------------------------------------------------------
:: Step 6: Create startup shortcut
:: ---------------------------------------------------------------------------
echo [6/6] Creating startup scripts...

:: Create start_service.bat
set "START_SCRIPT=!INSTALL_DIR!\start_service.bat"
(
    echo @echo off
    echo chcp 65001 ^>nul 2^>^&1
    echo echo Starting VisionHub1 AI Service...
    echo cd /d "!INSTALL_DIR!\python_service"
    echo call ".venv\Scripts\activate.bat"
    echo python -m app.main
    echo pause
) > "!START_SCRIPT!"
echo   Created: !START_SCRIPT!

:: Create verify_install.bat
set "VERIFY_SCRIPT=!INSTALL_DIR!\verify_install.bat"
(
    echo @echo off
    echo chcp 65001 ^>nul 2^>^&1
    echo echo ============================================================
    echo echo   VisionHub1 Installation Verification
    echo echo ============================================================
    echo echo.
    echo cd /d "!INSTALL_DIR!\python_service"
    echo call ".venv\Scripts\activate.bat"
    echo echo [1] Python version:
    echo python --version
    echo echo.
    echo echo [2] PyTorch + CUDA:
    echo python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    echo echo.
    echo echo [3] Key packages:
    echo python -c "import fastapi; import cv2; import numpy; import yaml; print('  fastapi, opencv, numpy, pyyaml - OK')"
    echo echo.
    echo echo [4] VisionHub import:
    echo python -c "from app.plugins.panel_seg_plugin import PanelSegV1Plugin; print('  panel_seg_v1 plugin - OK')"
    echo echo.
    echo echo ============================================================
    echo pause
) > "!VERIFY_SCRIPT!"
echo   Created: !VERIFY_SCRIPT!

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------
echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo   Install location:  !INSTALL_DIR!
echo   Data root:         !DATA_ROOT!
echo   Python venv:       !VENV_DIR!
echo.
echo   Next steps:
echo     1. Copy project data to !DATA_ROOT!\projects\
echo        (project.yaml, models\, datasets\)
echo     2. Run verify_install.bat to check the installation
echo     3. Run start_service.bat to start the AI service
echo     4. Start the C# UI client
echo.
echo   Useful scripts:
echo     !START_SCRIPT!     - Start AI service
echo     !VERIFY_SCRIPT!    - Verify installation
echo ============================================================
echo.
pause
