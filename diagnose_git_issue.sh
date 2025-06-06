#!/bin/bash

# Diagnose what's really going on with Git

echo "🔍 Diagnosing Git Issues..."
echo ""

# Show a sample of what's changed
echo "📄 Sample of changed files (first 10):"
git status --porcelain | head -10
echo ""

# Check if files are actually different
echo "🔎 Checking if files have real content differences:"
echo "First file:"
FIRST_FILE=$(git status --porcelain | head -1 | awk '{print $2}')
if [ -n "$FIRST_FILE" ]; then
    echo "File: $FIRST_FILE"
    echo "Diff:"
    git diff "$FIRST_FILE" | head -20
    echo ""
fi

# Check git config
echo "⚙️  Git Configuration:"
echo "core.autocrlf = $(git config core.autocrlf)"
echo "core.eol = $(git config core.eol)"
echo ""

# Check if .gitattributes is being applied
echo "📋 .gitattributes status:"
if [ -f .gitattributes ]; then
    echo "✓ .gitattributes exists"
    # Check if it's being recognized
    git check-attr -a README.md | head -5
else
    echo "✗ No .gitattributes file"
fi
echo ""

# Show what git thinks is happening
echo "📊 Git's view:"
echo "Modified files: $(git ls-files -m | wc -l)"
echo "Deleted files: $(git ls-files -d | wc -l)"
echo "Untracked files: $(git ls-files -o --exclude-standard | wc -l)"
echo ""

echo "💡 Likely issues:"
if git ls-files -d | grep -q .; then
    echo "- Git thinks some tracked files are deleted"
fi
if git diff --check | grep -q .; then
    echo "- Whitespace or line ending issues detected"
fi
