#!/usr/bin/env python3
"""
Quick verification script to check that all critical functionality works
after project structure refactoring.

This script attempts to:
1. Load scenario files from their new locations
2. Run a minimal simulation
3. Verify the parameter sweep can find its config files
4. Check that analysis scripts can be imported and run

Exit code 0 means all checks passed, non-zero means something failed.
"""

import os
import sys
import subprocess
import importlib
import logging
import traceback

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    if exists:
        print(f"✅ {description} found: {filepath}")
    else:
        print(f"❌ ERROR: {description} NOT found: {filepath}")
    return exists

def run_command(command, description, capture_output=True):
    """Run a command and return success state."""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True,
            capture_output=capture_output,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {description} failed")
        print(f"    Command output: {e.stdout}")
        print(f"    Error output: {e.stderr}")
        return False

def try_import(module_name):
    """Attempt to import a module and return success state."""
    try:
        importlib.import_module(module_name)
        print(f"✅ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"❌ ERROR: Could not import {module_name}: {e}")
        return False

def main():
    # Setup basic logging to suppress noisy warnings during tests
    logging.basicConfig(level=logging.ERROR)
    
    # Track test successes
    checks_passed = 0
    checks_total = 0
    
    print("\n=== Verifying Project Refactoring ===\n")
    
    # 1. Check if critical files exist in their new locations
    print("--- Checking File Locations ---")
    files_to_check = [
        ("scenarios/scenarios.json", "Standard scenarios file"),
        ("scenarios/enhanced_scenarios.json", "Enhanced scenarios file"),
        ("scenarios/pairwise_scenarios.json", "Pairwise scenarios file"),
        ("configs/multi_sweep_config.json", "Multi-sweep config file"),
        ("configs/sweep_config.json", "Sweep config file"),
        ("tests/test_core_basic.py", "Core tests file"),
        ("analysis_scripts/run_statistical_analysis.py", "Statistical analysis script")
    ]
    
    for filepath, description in files_to_check:
        checks_total += 1
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # 2. Test imports
    print("\n--- Testing Critical Imports ---")
    modules_to_test = [
        "npdl.core.agents",
        "npdl.core.environment",
        "npdl.core.utils",
        "npdl.simulation.runner",
        "npdl.visualization.data_loader",
        "main"  # Make sure main.py can be imported
    ]
    
    for module in modules_to_test:
        checks_total += 1
        if try_import(module):
            checks_passed += 1
    
    # 3. Run a quick test simulation
    print("\n--- Running Minimal Simulation ---")
    checks_total += 1
    mini_run_success = run_command(
        "python main.py --enhanced --num_runs 1 --verbose",
        "Minimal simulation run with enhanced scenarios"
    )
    if mini_run_success:
        checks_passed += 1
    
    # 4. Test the parameter sweep can find its config
    print("\n--- Testing Parameter Sweep Config Access ---")
    # Just import and check if it can find the config, don't run the full sweep
    checks_total += 1
    try:
        with open("configs/multi_sweep_config.json", "r") as f:
            config_content = f.read()
            if len(config_content) > 0:
                print("✅ Successfully loaded sweep config")
                checks_passed += 1
            else:
                print("❌ Config file exists but is empty")
    except Exception as e:
        print(f"❌ ERROR: Failed to load sweep config: {e}")
    
    # 5. Run the dedicated refactoring test
    print("\n--- Running Dedicated Refactoring Tests ---")
    checks_total += 1
    test_success = run_command(
        "python -m unittest tests/test_refactored_paths.py",
        "Refactoring tests"
    )
    if test_success:
        checks_passed += 1
    
    # Print summary
    print("\n=== Verification Summary ===")
    print(f"Passed {checks_passed}/{checks_total} checks")
    
    if checks_passed == checks_total:
        print("\n✅ All checks passed! Refactoring seems successful.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("❌ ERROR: Unhandled exception during verification")
        traceback.print_exc()
        sys.exit(1)
