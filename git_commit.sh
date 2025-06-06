#!/bin/bash

# Smart Git Commit Script (no Claude prefix)
# This script makes commits look like they're from you

# Configuration
MAX_FILES_PER_COMMIT=10  # Limit files per commit for clarity

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to generate commit message based on changes
generate_commit_message() {
    local files=("$@")
    local msg=""
    
    # Analyze file types and generate appropriate message
    local py_files=0
    local config_files=0
    local doc_files=0
    local test_files=0
    
    for file in "${files[@]}"; do
        if [[ $file == *.py ]]; then
            ((py_files++))
        elif [[ $file == *.json || $file == *.yaml || $file == *.yml || $file == *.toml ]]; then
            ((config_files++))
        elif [[ $file == *.md || $file == *.txt || $file == *.rst ]]; then
            ((doc_files++))
        elif [[ $file == *test* ]]; then
            ((test_files++))
        fi
    done
    
    # Build message based on file types
    if [ $py_files -gt 0 ]; then
        msg="Update Python code"
    fi
    if [ $config_files -gt 0 ]; then
        [ -n "$msg" ] && msg="$msg and configuration" || msg="Update configuration"
    fi
    if [ $doc_files -gt 0 ]; then
        [ -n "$msg" ] && msg="$msg and documentation" || msg="Update documentation"
    fi
    if [ $test_files -gt 0 ]; then
        [ -n "$msg" ] && msg="$msg and tests" || msg="Update tests"
    fi
    
    [ -z "$msg" ] && msg="Update project files"
    
    echo "$msg"
}

# Function to commit specific files
commit_files() {
    local message="$1"
    shift
    local files=("$@")
    
    # Add files
    for file in "${files[@]}"; do
        git add "$file"
    done
    
    # Commit (no prefix, looks like your commit)
    if git commit -m "$message"; then
        echo -e "${GREEN}✓ Committed ${#files[@]} files: $message${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed to commit${NC}"
        return 1
    fi
}

# Main function
main() {
    # Check if in git repo
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}Error: Not in a git repository${NC}"
        exit 1
    fi
    
    # Check for line endings issue
    REAL_CHANGES=$(git diff --name-only 2>/dev/null | wc -l || echo '0')
    ALL_CHANGES=$(git status --porcelain 2>/dev/null | wc -l || echo '0')
    
    if [ "$ALL_CHANGES" -gt 100 ] && [ "$REAL_CHANGES" -eq 0 ]; then
        echo -e "${RED}Error: Detected phantom changes (likely line endings issue)${NC}"
        echo "Files show as modified but have no real changes."
        echo ""
        echo "Quick fix: git add . && git reset --hard HEAD"
        echo "Or run: ./fix_git_config.sh"
        exit 1
    fi
    
    # Get modified files (excluding results directory by default)
    mapfile -t modified_files < <(git ls-files -m | grep -v "^results/")
    mapfile -t untracked_files < <(git ls-files -o --exclude-standard | grep -v "^results/")
    
    all_files=("${modified_files[@]}" "${untracked_files[@]}")
    
    if [ ${#all_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}No changes to commit (excluding results/)${NC}"
        echo "To include results, use: $0 --include-results"
        exit 0
    fi
    
    echo -e "${BLUE}Found ${#all_files[@]} changed files${NC}"
    
    # Group files by directory for logical commits
    declare -A file_groups
    
    for file in "${all_files[@]}"; do
        dir=$(dirname "$file")
        file_groups[$dir]+="$file "
    done
    
    # Commit files in groups
    for dir in "${!file_groups[@]}"; do
        files_in_dir=(${file_groups[$dir]})
        
        # If too many files, split into chunks
        if [ ${#files_in_dir[@]} -gt $MAX_FILES_PER_COMMIT ]; then
            for ((i=0; i<${#files_in_dir[@]}; i+=MAX_FILES_PER_COMMIT)); do
                chunk=("${files_in_dir[@]:i:MAX_FILES_PER_COMMIT}")
                msg=$(generate_commit_message "${chunk[@]}")
                commit_files "$msg in $dir" "${chunk[@]}"
            done
        else
            msg=$(generate_commit_message "${files_in_dir[@]}")
            commit_files "$msg in $dir" "${files_in_dir[@]}"
        fi
    done
    
    # Show summary
    echo -e "\n${GREEN}Commit summary:${NC}"
    git log --oneline -5
}

# Handle command line arguments
case "${1:-auto}" in
    "auto")
        main
        ;;
    "--include-results")
        # Include all files including results
        mapfile -t all_files < <(git ls-files -m && git ls-files -o --exclude-standard)
        # Continue with normal flow...
        main
        ;;
    "quick")
        # Quick commit all changes with provided message
        git add -A
        git commit -m "${2:-Quick update}"
        ;;
    "status")
        # Just show status
        git status
        ;;
    *)
        # Custom message
        git add -A
        git commit -m "$*"
        ;;
esac
