#!/usr/bin/env python3
"""
Apply fixes for the pairwise interaction model issues.

This script will modify the following files:
1. npdl/core/environment.py - Replace the _run_pairwise_round method
2. npdl/core/agents.py - Update the TitForTatStrategy.choose_move method

Usage:
    python apply_pairwise_fixes.py

After running this script, you should be able to run:
    python test_pairwise.py
"""

import os
import re
import sys

def read_file(path):
    """Read a file and return its content."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    """Write content to a file."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def replace_method(file_content, method_name, new_method_content):
    """Replace a method in a class with new content."""
    # Find method definition using regex
    pattern = r'([ \t]*)def {}\([^)]*\):.*?(?=^[ \t]*def |\Z)'.format(method_name)
    match = re.search(pattern, file_content, re.DOTALL | re.MULTILINE)
    
    if not match:
        print(f"ERROR: Method {method_name} not found.")
        return file_content
    
    # Extract indentation
    indentation = match.group(1)
    
    # Indent the new method content
    new_lines = []
    for line in new_method_content.split('\n'):
        if line.strip():
            new_lines.append(indentation + line)
        else:
            new_lines.append('')
    
    # Replace the old method with the new one
    new_method = '\n'.join(new_lines)
    updated_content = file_content[:match.start()] + new_method + file_content[match.end():]
    
    return updated_content

def update_tft_strategy(agents_content):
    """Update the TitForTatStrategy class."""
    # Read the improved TFT implementation
    tft_path = "improved_tft.py"
    if not os.path.exists(tft_path):
        print(f"ERROR: {tft_path} not found.")
        return agents_content
    
    tft_code = read_file(tft_path)
    
    # Extract the method body
    method_body = re.search(r'def tft_choose_move\([^)]*\):(.*?)$', tft_code, 
                           re.DOTALL | re.MULTILINE).group(1)
    
    # Find the TitForTatStrategy class
    tft_pattern = r'class TitForTatStrategy\(Strategy\):(.*?)(?=class|$)'
    tft_match = re.search(tft_pattern, agents_content, re.DOTALL)
    
    if not tft_match:
        print("ERROR: TitForTatStrategy class not found.")
        return agents_content
    
    # Find the choose_move method in the TitForTatStrategy class
    choose_move_pattern = r'([ \t]*)def choose_move\([^)]*\):.*?(?=[ \t]*def|\Z)'
    choose_move_match = re.search(choose_move_pattern, tft_match.group(1), re.DOTALL)
    
    if not choose_move_match:
        print("ERROR: choose_move method not found in TitForTatStrategy.")
        return agents_content
    
    # Extract indentation
    indentation = choose_move_match.group(1)
    
    # Create the new method with correct indentation
    new_method_lines = ['def choose_move(self, agent, neighbors):']
    for line in method_body.split('\n'):
        if line.strip():
            new_method_lines.append(indentation + '    ' + line.strip())
        else:
            new_method_lines.append('')
    
    new_method = indentation + '\n'.join(new_method_lines)
    
    # Replace the old method with the new one
    old_method = choose_move_match.group(0)
    updated_tft_class = tft_match.group(0).replace(old_method, new_method)
    
    # Replace the TitForTatStrategy class in the file
    updated_content = agents_content.replace(tft_match.group(0), updated_tft_class)
    
    return updated_content

def update_environment_class(env_content):
    """Update the _run_pairwise_round method in the Environment class."""
    # Read the improved pairwise round implementation
    pairwise_path = "improved_pairwise_round.py"
    if not os.path.exists(pairwise_path):
        print(f"ERROR: {pairwise_path} not found.")
        return env_content
    
    pairwise_code = read_file(pairwise_path)
    
    # Extract the method body
    method_body = re.search(r'def _run_pairwise_round\([^)]*\):(.*?)$', pairwise_code, 
                           re.DOTALL | re.MULTILINE).group(1)
    
    # Find and replace the _run_pairwise_round method
    return replace_method(env_content, '_run_pairwise_round', method_body)

def main():
    """Apply fixes to the relevant files."""
    # Find the required files
    core_dir = os.path.join('npdl', 'core')
    if not os.path.exists(core_dir):
        print(f"ERROR: Core directory not found at {core_dir}")
        sys.exit(1)
    
    env_path = os.path.join(core_dir, 'environment.py')
    agents_path = os.path.join(core_dir, 'agents.py')
    
    if not os.path.exists(env_path):
        print(f"ERROR: Environment file not found at {env_path}")
        sys.exit(1)
    
    if not os.path.exists(agents_path):
        print(f"ERROR: Agents file not found at {agents_path}")
        sys.exit(1)
    
    # Read current files
    env_content = read_file(env_path)
    agents_content = read_file(agents_path)
    
    # Apply updates
    updated_env = update_environment_class(env_content)
    updated_agents = update_tft_strategy(agents_content)
    
    # Create backup files
    backup_dir = "backups"
    os.makedirs(backup_dir, exist_ok=True)
    
    env_backup = os.path.join(backup_dir, 'environment.py.bak')
    agents_backup = os.path.join(backup_dir, 'agents.py.bak')
    
    write_file(env_backup, env_content)
    write_file(agents_backup, agents_content)
    
    print(f"Created backup files in {backup_dir}")
    
    # Write updated files
    write_file(env_path, updated_env)
    write_file(agents_path, updated_agents)
    
    print("Successfully applied pairwise fixes!")
    print("You can now run: python test_pairwise.py")

if __name__ == "__main__":
    main()
