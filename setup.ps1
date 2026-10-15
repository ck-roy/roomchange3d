param([switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Log = Join-Path $Root 'setup.log'
Start-Transcript -Path $Log -Append | Out-Null
try {
  $Py = Get-Command py -ErrorAction SilentlyContinue
  if ($Py) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed. Close RoomChange3D if it is running, then retry START_HERE.bat.' }
  } else { throw 'Python Launcher not found. Install Python 3.12 from python.org and enable the launcher.' }
  $Python = Join-Path $Root '.venv\Scripts\python.exe'
  if (!(Test-Path $Python)) { throw 'Virtual environment creation failed.' }
  & $Python -m pip install --upgrade pip
  # CUDA 12.8 wheels support modern RTX GPUs when the installed driver is compatible.
  try {
    & $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel install failed' }
  } catch {
    Write-Warning 'CUDA PyTorch wheel could not be installed. Trying CPU wheels.'
    & $Python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed for both CUDA and CPU.' }
  }
  & $Python -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed; see setup.log.' }
  & $Python -m pip uninstall -y opencv-python-headless
  if (!$SkipModels) {
    if (!(Get-Command git -ErrorAction SilentlyContinue)) { throw 'Git for Windows is not installed. Install it from git-scm.com, then run START_HERE.bat again.' }
    $Vggt = Join-Path $Root 'vendor\vggt'
    if (!(Test-Path $Vggt)) {
      New-Item -ItemType Directory -Force (Split-Path $Vggt) | Out-Null
      git clone --depth 1 https://github.com/facebookresearch/vggt.git $Vggt
      if ($LASTEXITCODE -ne 0) { throw 'Could not download the official VGGT source.' }
    }
    & $Python -m pip install --no-deps -e $Vggt
    if ($LASTEXITCODE -ne 0) { throw 'VGGT install failed.' }
  }
  & $Python -m pip check
  if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed; see setup.log.' }
  & $Python -c "import torch, cv2, streamlit, plotly, pyarrow, numpy; pyarrow.Table.from_pydict({'setup_check':[1]}); print('PyTorch',torch.__version__,'CUDA available',torch.cuda.is_available()); print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback'); print('OpenCV',cv2.__version__,'NumPy',numpy.__version__,'PyArrow',pyarrow.__version__)"
  if ($LASTEXITCODE -ne 0) { throw 'Installation verification failed.' }
  Write-Host 'Setup completed. First analysis downloads the selected VGGT checkpoint and recognition weights locally. The default VGGT checkpoint is non-commercial.'
} catch { Write-Error $_; exit 1 }
finally { Stop-Transcript | Out-Null }

