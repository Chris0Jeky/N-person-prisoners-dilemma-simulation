#!/bin/bash

# Check if the Git changes are just file permission issues

echo "🔍 Checking for permission issues (common WSL problem)..."
echo ""

# Check if core.filemode is set
FILEMODE=$(git config core.filemode)
echo "Current core.filemode: ${FILEMODE:-not set}"
echo ""

# Check what kind of changes git sees
echo "Types of changes:"
git status --porcelain | cut -c1-2 | sort | uniq -c
echo ""

# Check if it's just permission changes
echo "Checking first few files for permission-only changes:"
for file in $(git status --porcelain | head -5 | awk '{print $2}'); do
    if [ -f "$file" ]; then
        echo -n "$file: "
        if git diff "$file" | grep -q "^diff --git"; then
            if git diff "$file" | grep -q "mode"; then
                echo "permission change detected"
            else
                echo "content change detected"
            fi
        else
            echo "no diff?"
        fi
    fi
done

echo ""
echo "🔧 If these are permission changes, fix with:"
echo "git config core.filemode false"
echo ""
echo "Then:"
echo "git add . && git reset --hard HEAD"
