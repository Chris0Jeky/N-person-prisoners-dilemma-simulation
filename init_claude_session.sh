#!/bin/bash

# Claude Code Session Initializer (No Claude Prefix Version)

echo "╔════════════════════════════════════════════╗"
echo "║      Claude Code Session Initializer       ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "./git_commit.sh" ]; then
    echo "⚠️  Warning: Not in project root directory"
    echo "   Please run from: /mnt/c/Users/jekyt/Desktop/IHateSoar"
    echo ""
fi

# Make scripts executable
chmod +x git_commit.sh fix_line_endings.sh 2>/dev/null

# Show current git status
echo "📊 Current Git Status:"
echo "   Branch: $(git branch --show-current 2>/dev/null || echo 'Not in git repo')"

# Count changed files
CHANGED_COUNT=$(git status --porcelain 2>/dev/null | wc -l || echo '0')
echo "   Changed files: $CHANGED_COUNT"

# Special warning for line endings issue
if [ "$CHANGED_COUNT" -gt 1000 ]; then
    echo ""
    echo "⚠️  WARNING: Over 1000 files changed!"
    echo "   This is likely a line endings issue."
    echo "   Run: git reset --hard HEAD"
    echo "   Then: git config core.autocrlf true"
    echo "   See FIX_GIT_MESS.md for details"
fi

# Check last commit
LAST_COMMIT=$(git log -1 --pretty=format:"%h - %s (%cr)" 2>/dev/null || echo "No commits yet")
echo "   Last commit: $LAST_COMMIT"
echo ""

# Show reminders
echo "🤖 Session Reminders:"
echo "   • Use ./git_commit.sh for commits (no Claude prefix)"
echo "   • Commits will look like you made them"
echo "   • Follow PEP 8 and add type hints"
echo ""

echo "📋 Quick Commands:"
echo "   ./git_commit.sh         - Smart auto-commit"
echo "   ./git_commit.sh quick   - Quick commit all"
echo "   ./git_commit.sh status  - Check status"
echo "   git log --oneline -5    - Recent commits"
echo ""

echo "📝 Session Start Template:"
echo "─────────────────────────────────────────────"
echo "I'm working on the IHateSoar project. Please:"
echo "1. Use ./git_commit.sh for all commits"
echo "2. Follow PEP 8 and add type hints"
echo "3. Update tests when changing code"
echo ""
echo "First, check status: ./git_commit.sh status"
echo "─────────────────────────────────────────────"
echo ""
echo "✅ Ready to start! What would you like to work on?"
