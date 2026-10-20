$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=================================================================" -ForegroundColor Magenta
Write-Host "   MULTIMODAL VLM ARCHITECTURE & SHAPE VERIFICATION SUITE       " -ForegroundColor Magenta
Write-Host "=================================================================" -ForegroundColor Magenta

Write-Host "`n>>> Running Multimodal Architecture & Gradient Flow Tests <<<" -ForegroundColor Cyan
python tests/test_multimodal_architecture.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Multimodal Architecture Tests Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n>>> Running End-to-End Pipeline & Autoregressive Inference Test <<<" -ForegroundColor Green
python test.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline Verification Test Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[SUCCESS] All Multimodal Vision-Language Model Tests Passed!" -ForegroundColor Green
$global:LASTEXITCODE = 0
exit 0
