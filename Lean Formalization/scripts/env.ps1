# Dot-source this file for this task's isolated Lean/cache environment.
$formalizationRoot = Split-Path -Parent $PSScriptRoot
$formalizationTools = Join-Path $formalizationRoot '.tools'
$formalizationLean = Join-Path $formalizationTools 'lean-4.36.0-rc2-windows/bin'
$env:PATH = "$formalizationLean;$(Join-Path $formalizationTools 'gh/bin');C:\Program Files\Git\bin;$env:PATH"
$env:UV_CACHE_DIR = Join-Path $formalizationTools 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $formalizationTools 'uv-python'
$env:XDG_CACHE_HOME = Join-Path $formalizationTools 'cache'
$env:MATHLIB_CACHE_DIR = Join-Path $formalizationTools 'mathlib-cache'
$env:LAKE_HOME = Join-Path $formalizationTools 'lake'
$env:LAKE_CONFIG = Join-Path $formalizationTools 'lake-cache.toml'
$env:TAUCETI_LOCAL_LINT_CACHE = Join-Path $formalizationTools 'local-lint-cache'
$env:TAUCETI_LOCAL_AXIOMS_CACHE = Join-Path $formalizationTools 'local-axioms-cache'
$env:TMP = Join-Path $formalizationTools 'tmp'
$env:TEMP = $env:TMP
$env:RUNNER_TEMP = $env:TMP
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
$env:PYTHONPATH = "$(Join-Path $formalizationTools 'python');$(Join-Path $formalizationTools 'TauCetiTools')"
$env:TAUCETI_VERIFY_PYTHON = (Get-Command python.exe).Source
foreach ($formalizationDir in @($env:TMP, $env:XDG_CACHE_HOME, $env:LAKE_HOME)) {
    New-Item -ItemType Directory -Force -Path $formalizationDir | Out-Null
}
