"""
Script to test the pairwise integration and help clean up files.

This script:
1. Runs the pairwise integration test to verify functionality
2. Prints a list of files that can be safely removed
3. Optionally moves test files to the tests directory

Usage:
    python test_and_cleanup.py [--cleanup]

Options:
    --cleanup     Actually delete/move files (default is just to list them)
"""

import os
import sys
import shutil
import unittest
import argparse

# Add current directory to path so we can import local modules
sys.path.insert(0, os.path.abspath('.'))

# Files to clean up after integration
FILES_TO_REMOVE = [
    "improved_environment.py",
    "improved_pairwise_implementation.py",
    "improved_pairwise_round.py",
    "improved_tft.py",
    "improved_tft_strategy.py",
    "modified_lra_q.py",
    "modified_q_learning.py",
    "pairwise_method.py",
    "pairwise_round.py",
    "apply_pairwise_fixes.py",
    "apply_permanent_fixes.py",
    "patch_pairwise_runtime.py",
    "run_patched_tests.py",
    "PAIRWISE_IMPLEMENTATION_SUMMARY.md",
    "PAIRWISE_README.md"
]

# Files to move to tests directory
FILES_TO_MOVE = {
    "test_pairwise.py": "tests/test_pairwise.py"
}

def run_tests():
    """Run the pairwise integration tests."""
    print("Running pairwise integration tests...")
    from test_pairwise_integration import TestPairwiseInteraction
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPairwiseInteraction)
    result = unittest.TextTestRunner().run(suite)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Tests failed!")
        return False

def check_files():
    """Check which files exist and can be cleaned up."""
    existing_files = []
    
    print("\nChecking files to remove:")
    for file_path in FILES_TO_REMOVE:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (not found)")
            
    print("\nChecking files to move:")
    files_to_move = {}
    for source, dest in FILES_TO_MOVE.items():
        if os.path.exists(source):
            files_to_move[source] = dest
            print(f"  ✓ {source} -> {dest}")
        else:
            print(f"  ✗ {source} (not found)")
    
    return existing_files, files_to_move

def cleanup_files(files_to_remove, files_to_move):
    """Actually remove and move files if requested."""
    # Remove files
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    # Move files
    for source, dest in files_to_move.items():
        try:
            # Create directory if needed
            dest_dir = os.path.dirname(dest)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            # Move file
            shutil.move(source, dest)
            print(f"Moved: {source} -> {dest}")
        except Exception as e:
            print(f"Error moving {source}: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test and clean up pairwise integration")
    parser.add_argument('--cleanup', action='store_true', help='Actually delete/move files')
    args = parser.parse_args()
    
    # Run tests first
    tests_passed = run_tests()
    
    # Check which files exist
    files_to_remove, files_to_move = check_files()
    
    if args.cleanup:
        if not tests_passed:
            print("\n⚠️  WARNING: Tests failed - cleanup aborted.")
            return
            
        print("\nPerforming cleanup...")
        cleanup_files(files_to_remove, files_to_move)
        print("\nCleanup complete. Don't forget to commit your changes!")
    else:
        print("\nTo clean up these files, run again with --cleanup flag.")
        
if __name__ == "__main__":
    main()
