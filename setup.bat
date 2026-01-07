@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ==============================
rem EasyOCR GUI - setup.bat (clean)
rem Creates .venv, installs deps.
rem Tries CUDA (cu121), verifies, else falls back to CPU.
rem ==============================

set "BASE_DIR=%~dp0"
pushd "%BASE_DIR%"

rem -- Python launcher detection
where python >nul 2>&1
if %ERRORLEVEL%==0 (
  set "PY=python"
) else (
  where py >nul 2>&1
  if %ERRORLEVEL%==0 (
    set "PY=py -3"
  ) else (
    echo [ERROR] Python was not found in PATH. Install Python 3.10+ and rerun.
    exit /b 1
  )
)

rem -- Create venv if missing
if not exist ".venv" (
  echo [INFO] Creating virtual environment .venv
  %PY% -m venv .venv || ( echo [ERROR] venv creation failed & exit /b 1 )
)

rem -- Activate venv
call ".venv\Scripts\activate.bat" || ( echo [ERROR] Failed to activate venv & exit /b 1 )

rem -- Upgrade pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel
if %ERRORLEVEL% NEQ 0 echo [WARN] Pip upgrade failed, continuing...

rem -- Base deps (common)
set "REQ_COMMON=flask waitress pymupdf easyocr"

rem -- Detect if nvidia-smi exists (presence only)
set "HAVE_NVIDIA=0"
where nvidia-smi >nul 2>&1 && set "HAVE_NVIDIA=1"
if "!HAVE_NVIDIA!"=="1" (
  for /f "usebackq tokens=* delims=" %%A in (`nvidia-smi -L 2^>nul`) do (
    set "GPU_LINE=%%~A"
    goto :got_gpu_line
  )
)
:got_gpu_line
if "!HAVE_NVIDIA!"=="1" (
  if defined GPU_LINE (
    echo [INFO] NVIDIA present: !GPU_LINE!
  ) else (
    echo [INFO] NVIDIA present.
  )
) else (
  echo [INFO] NVIDIA GPU not detected (nvidia-smi not found). Will prefer CPU build.
)

rem -- Try install CUDA build of PyTorch (cu121) if NVIDIA present
set "TORCH_OK=0"
if "!HAVE_NVIDIA!"=="1" (
  echo [INFO] Installing PyTorch CUDA 12.1 wheels...
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  if %ERRORLEVEL% EQU 0 (
    set "TMPPY=%TEMP%\torch_check_%RANDOM%%RANDOM%.py"
    >"%TMPPY%" echo import sys, torch
    >>"%TMPPY%" echo ok=torch.cuda.is_available()
    >>"%TMPPY%" echo success=False
    >>"%TMPPY%" echo try:
    >>"%TMPPY%" echo ^    if ok:
    >>"%TMPPY%" echo ^        x=torch.rand((1024,1024), device='cuda')
    >>"%TMPPY%" echo ^        y=x@x
    >>"%TMPPY%" echo ^        success=bool(y.is_cuda)
    >>"%TMPPY%" echo except Exception as e:
    >>"%TMPPY%" echo ^    print("CUDA_TEST_FAIL=", repr(e))
    >>"%TMPPY%" echo print("CUDA_AVAIL=", ok)
    >>"%TMPPY%" echo print("CUDA_TEST_OK=", success)
    >>"%TMPPY%" echo sys.exit(0 if (ok and success) else 2)

    python "%TMPPY%"
    if %ERRORLEVEL% EQU 0 (
      set "TORCH_OK=1"
      echo [INFO] CUDA build OK and runnable.
    ) else (
      echo [WARN] CUDA test failed — falling back to CPU build of PyTorch.
    )
    del /f /q "%TMPPY%" >nul 2>&1
  ) else (
    echo [WARN] PyTorch CUDA installation failed — will fallback to CPU build.
  )
)

rem -- If CUDA not OK, enforce CPU build
if "!TORCH_OK!"=="0" (
  echo [INFO] Installing CPU build of PyTorch...
  python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
  if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install CPU PyTorch.
    exit /b 1
  )
)

rem -- Install remaining dependencies
echo [INFO] Installing app dependencies...
python -m pip install %REQ_COMMON%
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Failed to install app dependencies.
  exit /b 1
)

rem -- Create uploads directory
if not exist "uploads" mkdir "uploads"

rem -- Final summary
set "TMPPY2=%TEMP%\torch_summary_%RANDOM%%RANDOM%.py"
>"%TMPPY2%" echo import torch
>>"%TMPPY2%" echo dev="cuda" if torch.cuda.is_available() else "cpu"
>>"%TMPPY2%" echo name=torch.cuda.get_device_name(0) if dev=="cuda" else "CPU"
>>"%TMPPY2%" echo print(f"[SUMMARY] Torch: {torch.__version__} ^| Device: {dev} ^| {name}")
python "%TMPPY2%"
del /f /q "%TMPPY2%" >nul 2>&1

echo [INFO] Setup finished.
popd
exit /b 0
