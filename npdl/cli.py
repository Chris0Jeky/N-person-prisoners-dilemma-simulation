"""
Command-line interface for N-Person Prisoner's Dilemma simulation.

This module provides a command-line interface for running simulations
and launching the visualization dashboard.
"""

import argparse
import sys
import os
import importlib.util


def check_dependencies(package_names):
    """Check if required dependencies are installed."""
    missing = []
    for package in package_names:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    return missing


def run_visualization():
    """Run the visualization dashboard."""
    # Check visualization dependencies
    required_packages = ['dash', 'dash_bootstrap_components', 'plotly', 'flask']
    missing = check_dependencies(required_packages)
    
    if missing:
        print(f"Missing required dependencies for visualization: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        return 1
    
    # Import the dashboard module
    try:
        from npdl.visualization.dashboard import run_dashboard
        print("Starting visualization dashboard...")
        print("Open your browser at http://127.0.0.1:8050/")
        run_dashboard(debug=True)
    except Exception as e:
        print(f"Error starting visualization dashboard: {e}")
        return 1
    
    return 0


def run_simulation(args):
    """Run simulation with the specified arguments."""
    try:
        from npdl.simulation.runner import run_simulation as sim_runner
        
        print("Starting simulation...")
        # Convert args to parameters for the simulation runner
        sim_config = {
            'enhanced': args.enhanced,
            'scenario_file': args.scenario_file if not args.enhanced else 'enhanced_scenarios.json',
            'results_dir': args.results_dir,
            'log_dir': args.log_dir,
            'analyze': args.analyze,
            'verbose': args.verbose
        }
        
        # Run the simulation with the given configuration
        sim_runner(**sim_config)
        return 0
    except ImportError:
        # Fall back to main.py if the simulation module is not found
        try:
            from main import main as orig_main
            
            print("Using legacy simulation runner from main.py")
            # Convert args back to a format expected by the original main function
            sys.argv = ['main.py']
            
            if args.enhanced:
                sys.argv.append('--enhanced')
            
            if args.scenario_file and not args.enhanced:
                sys.argv.extend(['--scenario_file', args.scenario_file])
            
            if args.results_dir:
                sys.argv.extend(['--results_dir', args.results_dir])
            
            if args.log_dir:
                sys.argv.extend(['--log_dir', args.log_dir])
            
            if args.analyze:
                sys.argv.append('--analyze')
            
            if args.verbose:
                sys.argv.append('--verbose')
            
            # Run the original main function
            orig_main()
            return 0
        except ImportError as e:
            print(f"Error: Could not import simulation module: {e}")
            print("Make sure the npdl package is properly installed.")
            return 1


def run_interactive():
    """Run the interactive gameplay mode."""
    try:
        from npdl.interactive.game import main as interactive_main
        
        print("Starting interactive game mode...")
        # Run the interactive game
        interactive_main()
        return 0
    except ImportError as e:
        print(f"Error: Could not import interactive game module: {e}")
        print("Make sure the npdl package is properly installed.")
        return 1


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="N-Person Prisoner's Dilemma Simulation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation")
    sim_parser.add_argument('--enhanced', action='store_true',
                          help='Use enhanced scenarios from enhanced_scenarios.json')
    sim_parser.add_argument('--scenario_file', type=str, default='scenarios.json',
                          help='Path to the JSON file containing scenario definitions.')
    sim_parser.add_argument('--results_dir', type=str, default='results',
                          help='Directory to save experiment results.')
    sim_parser.add_argument('--log_dir', type=str, default='logs',
                          help='Directory to save log files.')
    sim_parser.add_argument('--analyze', action='store_true',
                          help='Run analysis on results after experiments complete.')
    sim_parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging.')
    
    # Visualization command
    vis_parser = subparsers.add_parser("visualize", help="Run visualization dashboard")
    
    # Interactive command
    int_parser = subparsers.add_parser("interactive", help="Run interactive game")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, print help and exit
    if args.command is None:
        parser.print_help()
        return 0
    
    # Run the specified command
    if args.command == "simulate":
        return run_simulation(args)
    elif args.command == "visualize":
        return run_visualization()
    elif args.command == "interactive":
        return run_interactive()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
