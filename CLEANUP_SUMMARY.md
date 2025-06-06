# Project Cleanup Summary

## Date: June 6, 2025

### Overview
Major project reorganization to improve structure, remove outdated files, and consolidate documentation.

## Changes Made

### 1. Created Organized Folder Structure
- **`utilities/`** - Utility scripts
  - `git_tools/` - Git diagnostic and fix scripts
  - `diagnostics/` - Display setup and Chrome installation scripts
- **`scripts/`** - Main execution scripts  
  - `runners/` - Scenario generation and parameter sweep scripts
  - `analysis/` - Analysis scripts
  - `demos/` - Demonstration scripts
- **`docs/`** - Consolidated documentation
  - `implementation/` - Technical implementation docs
  - `testing/` - Test documentation
- **`archive/`** - Old/deprecated files (git-ignored)
  - `old_tests/` - Outdated test files
  - `git_fixes/` - Old git issue documentation
  - `old_docs/` - Outdated documentation
  - `deprecated_scripts/` - Old utility scripts
  - `coverage_reports/` - Coverage data
  - `3person_failed_attempt/` - Failed 3-person experiment
  - `old_backups/` - Old backup files

### 2. Removed/Archived Files
- Moved all git diagnostic scripts to `utilities/git_tools/`
- Archived failed 3-person experiment attempt
- Removed duplicate test files from root directory
- Archived old coverage reports and htmlcov
- Removed empty logs directory
- Consolidated duplicate analysis folders

### 3. Documentation Updates
- Created new `docs/README.md` overview
- Consolidated pairwise mode documentation into single file
- Removed outdated test status reports
- Updated implementation documentation

### 4. Fixed Issues
- Resolved git file permission tracking issue (set `core.filemode = false`)
- Fixed import paths for relocated scripts
- Updated .gitignore to exclude archive folder

## Project Status

### Working Components
✅ Core NPDL framework (`npdl/core/`)
✅ 3-Person experiments (`3-Person_Tragic_vs_Reciprocity/new_attempt/`)
✅ Browser-tools MCP functionality
✅ Git commit helper scripts

### Dependencies
- Python dependencies listed in `requirements.txt` (not all installed in current environment)
- Node.js MCP packages installed but version mismatch warning

### Key Scripts Preserved
- `main.py` - Main simulation runner
- `run.py` - Simple runner
- `git_commit.sh` - Commit helper
- `init_claude_session.sh` - Session initializer
- All scenario JSON files in `scenarios/`

## Next Steps
1. Install Python dependencies: `pip install -r requirements.txt`
2. Fix MCP package version in package.json
3. Run tests to ensure everything works: `pytest tests/`
4. Update README with new folder structure