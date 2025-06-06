#!/bin/bash

# Nuclear option to fix Git issues

echo "☢️  Nuclear Git Fix - This will reset EVERYTHING"
echo "⚠️  Make sure you have no uncommitted work you want to keep!"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "🔧 Starting nuclear fix..."
echo ""

# Step 1: Stash any real changes (just in case)
echo "1️⃣ Stashing any changes..."
git stash push -m "Nuclear fix backup $(date +%Y%m%d_%H%M%S)" || true

# Step 2: Reset to the last commit
echo "2️⃣ Hard reset to last commit..."
git reset --hard HEAD

# Step 3: Clean everything
echo "3️⃣ Cleaning untracked files..."
git clean -fd

# Step 4: Set proper config
echo "4️⃣ Setting Git config..."
git config core.autocrlf input
git config core.eol lf
git config core.whitespace trailing-space,space-before-tab

# Step 5: Normalize line endings
echo "5️⃣ Normalizing line endings..."
git rm --cached -r . >/dev/null 2>&1
git reset --hard HEAD

# Step 6: Touch .gitattributes to ensure it's recognized
echo "6️⃣ Refreshing .gitattributes..."
touch .gitattributes
git add .gitattributes
git reset

# Step 7: Final check
echo ""
echo "📊 Final status:"
CHANGES=$(git status --porcelain | wc -l)
echo "Changed files: $CHANGES"

if [ "$CHANGES" -eq 0 ]; then
    echo "✅ Success! Git is clean."
else
    echo ""
    echo "😞 Still showing changes. This might be because:"
    echo "1. You have actual uncommitted changes"
    echo "2. Your Git installation has issues"
    echo "3. File permissions changed (WSL issue)"
    echo ""
    echo "Try: git status --porcelain | head -20"
    echo "To see what's still changed"
fi

echo ""
echo "📝 Your stashed changes (if any) can be recovered with:"
echo "git stash list"
echo "git stash pop"
