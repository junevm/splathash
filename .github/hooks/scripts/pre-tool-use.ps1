# planning-with-files: Pre-tool-use hook for GitHub Copilot (PowerShell)
# Reads the first 30 lines of task_plan.md to keep goals in context.
# Always allows tool execution — this hook never blocks tools.
# Always exits 0 — outputs JSON to stdout.

# Read stdin (required — Copilot pipes JSON to stdin)
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$InputData = [Console]::In.ReadToEnd()

$PlanFile = "task_plan.md"

if (-not (Test-Path $PlanFile)) {
    Write-Output '{}'
    exit 0
}

$Context = (Get-Content $PlanFile -TotalCount 30 -ErrorAction SilentlyContinue) -join "`n"

if (-not $Context) {
    Write-Output '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'
    exit 0
}

$output = @{
    hookSpecificOutput = @{
        hookEventName = "PreToolUse"
        permissionDecision = "allow"
        additionalContext = $Context
    }
}
$output | ConvertTo-Json -Depth 3 -Compress
exit 0
