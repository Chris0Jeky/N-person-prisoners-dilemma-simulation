# Fix Phantom Changes (184 files showing as modified)

## The Problem
- Git shows 184 changed files in WSL
- GitHub Desktop shows no changes
- These are "phantom" changes caused by line ending differences

## Quick Fix (Do This Now)
```bash
git add . && git reset --hard HEAD
```

This discards all phantom changes immediately.

## Permanent Fix
```bash
# Run the fix script
chmod +x fix_git_config.sh
./fix_git_config.sh
```

Or manually:
```bash
# Configure git for WSL
git config core.autocrlf input
git config core.eol lf

# Refresh the index
git rm --cached -r .
git reset --hard
```

## Why This Happens
- Windows uses CRLF line endings
- Linux/WSL uses LF line endings
- Git sees every line as changed
- GitHub Desktop (Windows) filters these out

## Prevention
The `.gitattributes` file should prevent this, but sometimes Git needs to be refreshed to recognize it.

## Updated Scripts
Your scripts now detect this issue:
- `init_claude_session.sh` - Warns about phantom changes
- `git_commit.sh` - Refuses to commit phantom changes

## For Claude Code
Tell Claude: "There are phantom line ending changes. They're not real changes, just ignore them."
