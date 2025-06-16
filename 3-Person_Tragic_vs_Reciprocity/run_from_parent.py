#!/usr/bin/env python3
"""
Helper script to set up Python path when running NPD simulator scripts.
Run this from the 3-Person_Tragic_vs_Reciprocity directory.
"""

import sys
import os
from pathlib import Path

# Add the parent of parent directory to Python path so npd_simulator can be imported
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(parent_dir))

# Now you can run any npd_simulator module
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_from_parent.py <module> [args...]")
        print("Examples:")
        print("  python run_from_parent.py npd_simulator.demo")
        print("  python run_from_parent.py npd_simulator.main single -c configs/example_config.json")
        print("  python run_from_parent.py npd_simulator.experiments.research_experiments")
        sys.exit(1)
    
    module_name = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove the script name
    
    # Import and run the module
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    
    module_name = module_name.replace('/', '.').replace('\\', '.')
    
    try:
        if '.' in module_name:
            # Import the module
            parts = module_name.split('.')
            module = __import__(module_name, fromlist=[parts[-1]])
            
            # If it has a main function or __main__ block, execute it
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, '__main__'):
                # Module has if __name__ == "__main__": block
                # Re-run the module as __main__
                import runpy
                runpy.run_module(module_name, run_name='__main__')
        else:
            # Run as a script
            import runpy
            runpy.run_module(module_name, run_name='__main__')
            
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running module {module_name}: {e}")
        sys.exit(1)