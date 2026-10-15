$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host 'RoomChange3D system check'
Get-Command py,git -ErrorAction SilentlyContinue | Select-Object Name,Source
py -0p
nvidia-smi
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (Test-Path $Py) {
  & $Py -c "import torch; print('torch',torch.__version__,'CUDA',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
  & $Py -m pip check
  & $Py -c "import importlib.util as u; print('VGGT source installed:',bool(u.find_spec('vggt')))"
  $ModelCache = Join-Path $env:USERPROFILE '.cache\huggingface\hub\models--facebook--VGGT-1B\snapshots'
  Write-Host "VGGT checkpoint cached: $(Test-Path $ModelCache) ($ModelCache)"
} else { Write-Warning 'Virtual environment missing. Double-click START_HERE.bat.' }
Get-PSDrive -Name C | Select-Object Used,Free

