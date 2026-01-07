@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================
rem EasyOCR GUI - start.bat (live console output + fresh log)
rem ============================================================

rem ---------- CONFIG ----------
set "VENV_DIR=.venv"
set "PYTHON=py"
set "PT_INDEX_CPU=https://download.pytorch.org/whl/cpu"
set "PT_INDEX_121=https://download.pytorch.org/whl/cu121"
set "PT_INDEX_128=https://download.pytorch.org/whl/cu128"

set "HOST=192.168.80.246"
set "PORT=7860"
set "RULE_NAME=EasyOCR_Flask_%HOST%_%PORT%"
set "BASE_DIR=%~dp0"
set "LOG_DIR=%BASE_DIR%Logs"
set "LOG_FILE=%LOG_DIR%\easyocr_server.log"

rem Env for app.py
set "EASYOCR_HOST=%HOST%"
set "EASYOCR_PORT=%PORT%"
set "EASYOCR_THREADS=8"
set "EASYOCR_LANGS=en,sk,cs,de"
set "EASYOCR_MAX_MB=64"
set "EASYOCR_UPLOAD_DIR=uploads"

rem Avoid accidental local shadowing
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="

echo.
echo ============================================================
echo [1/7] Detecting GPU via WMI...
echo ============================================================
for /f "usebackq tokens=*" %%G in (`powershell -NoProfile -Command ^
  "(Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }).Name"`) do (
  set "GPU_NAME=%%G"
)

if not defined GPU_NAME (
  echo No NVIDIA GPU detected. Using CPU mode.
  set "MODE=cpu"
) else (
  echo Detected GPU: %GPU_NAME%
  echo.
  echo Checking generation...
  echo %GPU_NAME% | findstr /I "5060 5090 5080 5070 5050" >nul && set "MODE=cu128"
  echo %GPU_NAME% | findstr /I "4060 4070 4080 4090 3060 3070 3080 3090" >nul && if not defined MODE set "MODE=cu121"
)
if not defined MODE set "MODE=cu121"

if /I "%MODE%"=="cpu" (
  set "PT_INDEX=%PT_INDEX_CPU%"
) else if /I "%MODE%"=="cu121" (
  set "PT_INDEX=%PT_INDEX_121%"
) else (
  set "PT_INDEX=%PT_INDEX_128%"
)

echo Selected mode: %MODE%
echo Using PyTorch index: %PT_INDEX%

echo.
echo ============================================================
echo [2/7] Checking virtual environment
echo ============================================================
if exist "%VENV_DIR%\Scripts\python.exe" (
  echo Found existing venv. Skipping setup.
  goto RUN_PHASE
)

echo No venv found. Proceeding with one-time setup...

echo.
echo ============================================================
echo [3/7] Creating virtual environment
echo ============================================================
%PYTHON% -m venv "%VENV_DIR%" || (echo ERROR creating venv & exit /b 1)
call "%VENV_DIR%\Scripts\activate.bat" || (echo ERROR activating venv & exit /b 1)

echo.
echo ============================================================
echo [4/7] Installing PyTorch
echo ============================================================
python -m pip install --upgrade pip setuptools wheel
if /I "%MODE%"=="cu128" (
  python -m pip install --index-url "%PT_INDEX%" torch torchvision torchaudio || (
    echo WARN cu128 stable failed, trying nightly
    python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision torchaudio || (echo ERROR installing torch cu128 & exit /b 1)
  )
) else (
  python -m pip install --index-url "%PT_INDEX%" torch torchvision torchaudio || (echo ERROR installing torch & exit /b 1)
)

echo.
echo ============================================================
echo [5/7] Installing EasyOCR stack
echo ============================================================
python -m pip install "numpy>=2.2,<2.3" easyocr opencv-python-headless "pillow==10.4.0" flask waitress pymupdf || (echo ERROR installing deps & exit /b 1)
python -m pip install --upgrade "python-bidi==0.6.7"

echo.
echo ============================================================
echo [6/7] Torch sanity
echo ============================================================
python -c "import torch; print('Torch', torch.__version__)" || (echo ERROR loading torch & exit /b 1)

echo.
echo ============================================================
echo [7/7] Setup complete
echo ============================================================

:RUN_PHASE
pushd "%BASE_DIR%"

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo ERROR .venv not found. Setup failed or missing.
  popd
  exit /b 1
)
call "%VENV_DIR%\Scripts\activate.bat" || (echo ERROR activating venv & exit /b 1)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "uploads" mkdir "uploads"

rem ==== Fresh log on every start ====
del "%LOG_FILE%" >nul 2>&1
echo -------------------------------------------------------------- > "%LOG_FILE%"
echo [%date% %time%] START >> "%LOG_FILE%"

echo ==== Freeing port %PORT% if needed ====
for /f "tokens=5" %%p in ('netstat -ano ^| find ":%PORT%" ^| find "LISTENING"') do (
  echo Killing PID %%p on port %PORT%
  taskkill /PID %%p /F >nul 2>&1
)

echo ==== Opening firewall rule ====
netsh advfirewall firewall delete rule name="%RULE_NAME%" >nul 2>&1
netsh advfirewall firewall add rule name="%RULE_NAME%" dir=in action=allow protocol=TCP localip=%HOST% localport=%PORT% profile=any >nul
if %ERRORLEVEL% NEQ 0 (
  echo WARN firewall add failed (need Administrator?)
) else (
  echo INFO Firewall rule added: %RULE_NAME% TCP %HOST%:%PORT%
)

echo.
echo INFO Starting EasyOCR GUI on http://%HOST%:%PORT%
echo INFO Logging to: "%LOG_FILE%"
echo (Press Ctrl+C to stop)
echo --------------------------------------------------------------

rem ==== Run with live output AND log ====
powershell -NoProfile -Command ^
  "& {python 'app.py' 2>&1 | Tee-Object -FilePath '%LOG_FILE%'}"

echo [%date% %time%] STOP >> "%LOG_FILE%"

netsh advfirewall firewall delete rule name="%RULE_NAME%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  echo INFO Firewall rule removed: %RULE_NAME%
) else (
  echo WARN Could not remove firewall rule
)

popd
exit /b 0
