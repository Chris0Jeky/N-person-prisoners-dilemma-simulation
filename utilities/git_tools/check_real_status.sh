#!/bin/bash

# Check real Git status, distinguishing between real changes and line ending issues

echo "🔍 Checking Git Status..."
echo ""

# Get different types of changes
ALL_CHANGES=$(git status --porcelain 2>/dev/null | wc -l)
REAL_CHANGES=$(git diff --name-only 2>/dev/null | wc -l)
STAGED_CHANGES=$(git diff --cached --name-only 2>/dev/null | wc -l)
UNTRACKED=$(git ls-files -o --exclude-standard 2>/dev/null | wc -l)

echo "📊 Status Summary:"
echo "   Total reported changes: $ALL_CHANGES"
echo "   Real content changes: $REAL_CHANGES"
echo "   Staged changes: $STAGED_CHANGES"
echo "   Untracked files: $UNTRACKED"
echo ""

# Diagnose the situation
if [ "$ALL_CHANGES" -gt 100 ] && [ "$REAL_CHANGES" -eq 0 ]; then
    echo "⚠️  PHANTOM CHANGES DETECTED!"
    echo "   These are line ending changes, not real modifications."
    echo ""
    echo "   Quick fix:"
    echo "   $ git add . && git reset --hard HEAD"
    echo ""
elif [ "$ALL_CHANGES" -eq 0 ]; then
    echo "✅ Working directory is clean!"
elif [ "$ALL_CHANGES" -lt 10 ]; then
    echo "📝 You have a few real changes:"
    git status --short
else
    echo "📝 You have real changes:"
    echo "   Run 'git status' to see details"
fi

echo ""
echo "💡 Tips:"
echo "   - Use './git_commit.sh status' for basic status"
echo "   - Use 'git diff' to see actual content changes"
echo "   - GitHub Desktop filters out line ending changes automatically"
