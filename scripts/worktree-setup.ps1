# CS2 Analyzer Parallel Development Setup
# Based on Boris Cherny's "3-5 git worktrees at once" strategy
#
# Usage: .\scripts\worktree-setup.ps1
#
# This creates 5 parallel worktrees for independent development:
# - ai-coaching: LLM features (ai/coaching.py, ai/patterns.py)
# - api-endpoints: Backend work (api.py, infra/cache.py)
# - ui-frontend: Visualization (static/, visualization/)
# - bugfix-hotfix: Quick fixes (never blocks feature work)
# - performance: Optimization (metrics_optimized.py)

$WORKTREE_DIR = "..\cs2-worktrees"

Write-Host "CS2 Analyzer - Parallel Development Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Create worktree directory
if (-not (Test-Path $WORKTREE_DIR)) {
    New-Item -ItemType Directory -Force -Path $WORKTREE_DIR | Out-Null
    Write-Host "Created worktree directory: $WORKTREE_DIR" -ForegroundColor Green
}

# Function to create a worktree
function New-Worktree {
    param(
        [string]$Name,
        [string]$Branch,
        [string]$Description
    )

    $path = "$WORKTREE_DIR\$Name"

    if (Test-Path $path) {
        Write-Host "  [EXISTS] $Name - already exists" -ForegroundColor Yellow
        return
    }

    Write-Host "  Creating $Name ($Description)..." -ForegroundColor White

    # Check if branch exists
    $branchExists = git branch --list $Branch

    if ($branchExists) {
        git worktree add $path $Branch 2>$null
    } else {
        git worktree add -b $Branch $path 2>$null
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $Name created on branch $Branch" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] Failed to create $Name" -ForegroundColor Red
    }
}

Write-Host "Creating worktrees..." -ForegroundColor Cyan
Write-Host ""

# Create each worktree
New-Worktree -Name "ai-coaching" -Branch "feature/ai-coaching" -Description "LLM features"
New-Worktree -Name "api-endpoints" -Branch "feature/api-work" -Description "Backend work"
New-Worktree -Name "ui-frontend" -Branch "feature/ui-improvements" -Description "Visualization"
New-Worktree -Name "bugfix-hotfix" -Branch "bugfix/current" -Description "Quick fixes"
New-Worktree -Name "performance" -Branch "perf/optimization" -Description "Optimization"

Write-Host ""
Write-Host "Worktree setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Launch separate Claude sessions in each:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  cd $WORKTREE_DIR\ai-coaching && claude" -ForegroundColor White
Write-Host "  cd $WORKTREE_DIR\api-endpoints && claude" -ForegroundColor White
Write-Host "  cd $WORKTREE_DIR\ui-frontend && claude" -ForegroundColor White
Write-Host "  cd $WORKTREE_DIR\bugfix-hotfix && claude" -ForegroundColor White
Write-Host "  cd $WORKTREE_DIR\performance && claude" -ForegroundColor White
Write-Host ""
Write-Host "Or use PowerShell shortcuts (add to `$PROFILE):" -ForegroundColor Cyan
Write-Host ""
Write-Host '  function cs2-ai { cd ..\cs2-worktrees\ai-coaching }' -ForegroundColor Gray
Write-Host '  function cs2-api { cd ..\cs2-worktrees\api-endpoints }' -ForegroundColor Gray
Write-Host '  function cs2-ui { cd ..\cs2-worktrees\ui-frontend }' -ForegroundColor Gray
Write-Host '  function cs2-fix { cd ..\cs2-worktrees\bugfix-hotfix }' -ForegroundColor Gray
Write-Host '  function cs2-perf { cd ..\cs2-worktrees\performance }' -ForegroundColor Gray
Write-Host ""
Write-Host "List all worktrees:" -ForegroundColor Cyan
git worktree list
Write-Host ""
