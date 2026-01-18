# Ralph Setup Guide for IndexTTS2-Rust

## Quick Setup (Git Bash or WSL Required)

Ralph is a bash-based tool. On Windows, use **Git Bash** or **WSL**.

### Step 1: Install Ralph (One Time)

```bash
# In Git Bash or WSL
cd ~
git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code
./install.sh
```

This adds `ralph`, `ralph-monitor`, and `ralph-setup` commands globally.

### Step 2: Initialize This Project for Ralph

```bash
# Navigate to the project (in Git Bash)
cd /c/AI/indextts2-rust

# Ralph is already configured! Files created:
# - PROMPT.md (main instructions)
# - @fix_plan.md (task tracking)
# - @AGENT.md (build commands)
```

### Step 3: Run Ralph

```bash
# Start autonomous development with monitoring
ralph --monitor

# Or without tmux monitoring
ralph

# Check status
ralph --status
```

## How Ralph Works

1. **Reads PROMPT.md** - Your project instructions with MCP tool guidance
2. **Checks @fix_plan.md** - Current task to work on
3. **Runs Claude Code** - Executes the task
4. **Updates Progress** - Marks tasks complete
5. **Repeats** - Until all tasks done or limits reached

## Ralph Commands

```bash
ralph --monitor           # Start with tmux dashboard (recommended)
ralph --calls 50          # Limit API calls per hour
ralph --timeout 30        # 30-minute timeout per task
ralph --verbose           # Detailed progress output
ralph --status            # Check current status
ralph-monitor             # Standalone monitoring dashboard
```

## Exit Signals

Ralph stops when it sees:
- All tasks in `@fix_plan.md` marked `[x]`
- `PHASE_COMPLETE` or `PROJECT_COMPLETE` in output
- Rate limits or timeouts

## Important Notes

1. **Use Git Bash/WSL** - Ralph doesn't work in PowerShell
2. **tmux Required** - Install via `pacman -S tmux` (Git Bash) or `apt install tmux` (WSL)
3. **jq Required** - Install via `pacman -S jq` or `apt install jq`

## Monitoring Tips

```bash
# Detach from tmux (keeps Ralph running)
Ctrl+B then D

# Reattach to session
tmux attach

# List sessions
tmux list-sessions
```

## Troubleshooting

**Ralph not found?**
- Make sure `~/.ralph` is in your PATH
- Run `source ~/.bashrc` after install

**tmux not found?**
- Git Bash: `pacman -S tmux`
- WSL: `sudo apt install tmux`

**jq not found?**
- Git Bash: `pacman -S jq`  
- WSL: `sudo apt install jq`
