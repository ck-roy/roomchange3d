$ErrorActionPreference = 'Stop'

$root = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path).TrimEnd('\')
$destination = Join-Path $root 'github_upload'
$destinationFull = [IO.Path]::GetFullPath($destination)
if (!$destinationFull.StartsWith("$root\", [StringComparison]::OrdinalIgnoreCase) -or
    $destinationFull -ne [IO.Path]::GetFullPath((Join-Path $root 'github_upload'))) {
  throw "Refusing unsafe staging path: $destination"
}

if (Test-Path -LiteralPath $destination) {
  $existing = Get-Item -LiteralPath $destination -Force
  if ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw "Refusing to replace a linked staging path: $destination"
  }
  Remove-Item -LiteralPath $destination -Recurse -Force
}
New-Item -ItemType Directory -Path $destination | Out-Null

$rootFiles = @(
  '.gitignore', 'LICENSE', 'README.md', 'START_HERE.bat',
  'THIRD_PARTY_NOTICES.md', 'app.py', 'doctor.ps1',
  'pyproject.toml', 'requirements.txt', 'run_app.bat', 'setup.ps1'
)
foreach ($name in $rootFiles) {
  $source = Join-Path $root $name
  if (!(Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Required submission file is missing: $source"
  }
  Copy-Item -LiteralPath $source -Destination $destination
}

foreach ($directory in @('examples', 'roomchange', 'scripts', 'tests')) {
  $sourceRoot = Join-Path $root $directory
  $files = Get-ChildItem -LiteralPath $sourceRoot -File -Recurse | Where-Object {
    if ($directory -eq 'examples') { $_.Extension -eq '.md' }
    else { $_.Extension -in @('.py', '.ps1', '.md') }
  }
  foreach ($file in $files) {
    $relative = $file.FullName.Substring($sourceRoot.Length).TrimStart('\')
    $target = Join-Path (Join-Path $destination $directory) $relative
    $targetParent = Split-Path -Parent $target
    New-Item -ItemType Directory -Force -Path $targetParent | Out-Null
    Copy-Item -LiteralPath $file.FullName -Destination $target
  }
}

Write-Host "Clean GitHub upload folder prepared: $destination"
Write-Host 'Upload the contents of this folder, preserving the roomchange, scripts, tests, and examples directories.'
