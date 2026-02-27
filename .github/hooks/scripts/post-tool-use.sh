#!/bin/bash
# planning-with-files: Post-tool-use hook for GitHub Copilot
# Reminds the agent to update task_plan.md after tool use.
# Always exits 0 — outputs JSON to stdout.

# Read stdin (required — Copilot pipes JSON to stdin)
INPUT=$(cat)

echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"[planning-with-files] File updated. If this completes a phase, update task_plan.md status."}}'
exit 0
