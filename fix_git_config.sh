#!/bin/bash

# Fix Git configuration for Windows/WSL compatibility

echo "🔧 Fixing Git configuration for Windows/WSL..."
echo ""

# Set core.autocrlf to handle line endings
git config core.autocrlf input
git config core.eol lf

# Refresh the index to pick up the .gitattributes file
echo "📄 Refreshing Git index with .gitattributes settings..."
git rm --cached -r .
git reset --hard

echo ""
echo "✅ Git configuration fixed!"
echo ""
echo "📊 Current status:"
git status --short | wc -l | xargs -I {} echo "   Changed files: {}"
echo ""

if [ $(git status --short | wc -l) -eq 0 ]; then
    echo "🎉 Success! No phantom changes detected."
else
    echo "⚠️  Still showing changes. Try:"
    echo "   git add . && git reset --hard HEAD"
fi
