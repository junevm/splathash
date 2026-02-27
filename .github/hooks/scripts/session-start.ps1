# planning-with-files: Session start hook for GitHub Copilot (PowerShell)
# When task_plan.md exists: runs session-catchup or reads plan header.
# When task_plan.md doesn't exist: injects SKILL.md so Copilot knows the planning workflow.
# Always exits 0 — outputs JSON to stdout.

# Read stdin (required — Copilot pipes JSON to stdin)
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$InputData = [Console]::In.ReadToEnd()

$PlanFile = "task_plan.md"
$SkillDir = ".github/skills/planning-with-files"

if (Test-Path $PlanFile) {
    # Plan exists — try session catchup, fall back to reading plan header
    $Catchup = ""
    if (Test-Path "$SkillDir/scripts/session-catchup.py") {
        try {
            $PythonCmd = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { "python" }
            $Catchup = & $PythonCmd "$SkillDir/scripts/session-catchup.py" (Get-Location).Path 2>$null
        } catch {
            $Catchup = ""
        }
    }

    if ($Catchup) {
        $Context = $Catchup -join "`n"
    } else {
        $Context = (Get-Content $PlanFile -TotalCount 5 -ErrorAction SilentlyContinue) -join "`n"
    }
} else {
    # No plan yet — inject SKILL.md so Copilot knows the planning workflow and templates
    if (Test-Path "$SkillDir/SKILL.md") {
        $Context = Get-Content "$SkillDir/SKILL.md" -Raw -ErrorAction SilentlyContinue
    }
}

if (-not $Context) {
    Write-Output '{}'
    exit 0
}

$output = @{
    hookSpecificOutput = @{
        hookEventName = "SessionStart"
        additionalContext = $Context
    }
}
$output | ConvertTo-Json -Depth 3 -Compress
exit 0
