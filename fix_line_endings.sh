#!/bin/bash

# Fix line endings issue between Windows and WSL

echo "Fixing line endings issue..."

# Configure git to handle line endings properly
git config core.autocrlf true

# Remove all files from the index (but not the working directory)
git rm --cached -r .

# Re-add all files with proper line endings
git add .

echo "Line endings fixed! Now you can commit normally."
echo ""
echo "To reset everything without committing these 7k files:"
echo "  git reset --hard HEAD"
echo ""
echo "Or to commit with a message explaining the fix:"
echo "  git commit -m 'Fix line endings for Windows/WSL compatibility'"
