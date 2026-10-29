#!/bin/bash
# Run the Claude health-check agent against the deployed Mosaic HF Space.
#
# Runs scripts/hf_space_e2e.py (which factory-restarts the Space once on HF
# infrastructure failures), then hands report.json to a Claude agent that
# diagnoses failures and files/updates a Jira issue labelled "mosaic-hf-e2e"
# (via the Atlassian MCP server configured in Claude Code). The check runs
# outside the agent because it can take up to an hour, longer than an agent
# Bash call is allowed to run. The agent's instructions are in
# scripts/hf_space_e2e_agent_prompt.md.
#
# Schedule from cron by submitting it as a SLURM job (scrontab may be
# disabled), e.g. every 3 days at 09:00 local time:
#
#   0 9 */3 * * cd /path/to/mosaic && /usr/bin/sbatch --mail-type=FAIL \
#       --mail-user=you@mskcc.org scripts/hf_space_e2e_agent.sh
#
# cron does not read ~/.bashrc, so the script first sources
# $MOSAIC_E2E_ENV (default ~/.config/mosaic-hf-e2e.env) if it exists. Put
# PATH additions (claude, a python3 with playwright) and credentials there.
#
# Required environment:
#   JIRA_PROJECT  Jira project key for failure tickets
#   JIRA_ASSIGNEE Jira username new tickets are assigned to (optional)
#   HF_TOKEN      Token with write access to raylim/mosaic (for the restart);
#                 read from ~/.cache/huggingface/token if unset
#   JIRA_*        Credentials for the mcp-atlassian server, if it reads them
#                 from the environment
#
# Exits non-zero when the Space check fails, so `--mail-type=FAIL` emails you.

#SBATCH --job-name=mosaic-hf-e2e
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hf_e2e_%j.out

set -euo pipefail

# Under sbatch the script runs from a spool copy, so prefer the submit dir.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

ENV_FILE="${MOSAIC_E2E_ENV:-$HOME/.config/mosaic-hf-e2e.env}"
# shellcheck source=/dev/null
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

: "${JIRA_PROJECT:?Set JIRA_PROJECT to the Jira project key}"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
fi
: "${HF_TOKEN:?Set HF_TOKEN (needs write access to raylim/mosaic)}"

RUN_DIR="$REPO_DIR/hf_e2e_out/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
export RUN_DIR JIRA_PROJECT HF_TOKEN JIRA_ASSIGNEE="${JIRA_ASSIGNEE:-}"  # HF_TOKEN is withheld from the agent below

# Keep the comparison refs fresh for diagnosis; never touches the work tree.
git fetch --quiet origin main 2>/dev/null || true
git fetch --quiet hf main 2>/dev/null || true

echo "Mosaic HF Space e2e — run dir: $RUN_DIR"
E2E_STATUS=0
# E2E_EXTRA_ARGS: extra hf_space_e2e.py flags, e.g. to test the failure path.
# shellcheck disable=SC2086
python3 scripts/hf_space_e2e.py --out-dir "$RUN_DIR" --restart-on-infra-error \
    ${E2E_EXTRA_ARGS:-} 2>&1 | tee "$RUN_DIR/e2e.log" || E2E_STATUS=$?
if [[ ! -f "$RUN_DIR/report.json" ]]; then
    echo "hf_space_e2e.py produced no report.json (exit $E2E_STATUS)" >&2
    exit 1
fi

ALLOWED_TOOLS=(
    "Read" "Grep" "Glob"
    "Bash(curl -s https://huggingface.co/api/spaces/:*)"
    "Bash(git log:*)" "Bash(git show:*)"
    "Bash(ls:*)" "Bash(jq:*)"
    "mcp__atlassian__jira_search"
    "mcp__atlassian__jira_get_issue"
    "mcp__atlassian__jira_create_issue"
    "mcp__atlassian__jira_add_comment"
)

env -u HF_TOKEN claude -p "$(<scripts/hf_space_e2e_agent_prompt.md)" \
    --model "${CLAUDE_MODEL:-claude-sonnet-5}" \
    --allowedTools "${ALLOWED_TOOLS[@]}" \
    --output-format text | tee "$RUN_DIR/agent.log"

exit "$E2E_STATUS"
