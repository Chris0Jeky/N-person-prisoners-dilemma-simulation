"""
Main Orchestrator for N-Person Prisoner's Dilemma Simulator

This is the central entry point for running experiments, managing configurations,
and coordinating different types of simulations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from experiments.runners.experiment_runner import ExperimentRunner
from experiments.runners.batch_runner import BatchRunner
from analysis.visualizers.plot_generator import PlotGenerator
from analysis.reporters.report_generator import ReportGenerator
from utils.config_loader import load_config
from utils.logging import setup_logger


class NPDSimulator:
    """
    Main orchestrator for N-Person Prisoner's Dilemma simulations.
    
    Features:
    - Run single experiments
    - Run batch experiments
    - Comparative analysis across agent counts
    - Automatic visualization generation
    - Comprehensive reporting
    """
    
    def __init__(self, base_output_dir: str = "results"):
        """
        Initialize the simulator.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = setup_logger("NPDSimulator")
        
    def run_single_experiment(self, config: Dict[str, Any], experiment_type: str = "npd") -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            config: Experiment configuration
            experiment_type: Type of experiment ("npd" or "pairwise")
            
        Returns:
            Experiment results
        """
        # Determine output directory based on experiment type
        if experiment_type == "npd":
            output_dir = self.base_output_dir / "basic_runs"
        elif experiment_type == "pairwise":
            output_dir = self.base_output_dir / "basic_runs"
        elif "qlearning" in str(config.get('name', '')).lower():
            output_dir = self.base_output_dir / "qlearning_tests"
        else:
            output_dir = self.base_output_dir / "basic_runs"
        
        # Create runner
        runner = ExperimentRunner(output_dir, config.get('name'))
        
        # Run experiment
        if experiment_type == "npd":
            results = runner.run_npd_experiment(config)
        elif experiment_type == "pairwise":
            results = runner.run_pairwise_experiment(config)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Generate visualizations
        self._generate_visualizations(results, runner.figures_dir)
        
        return results
    
    def run_batch_experiments(self, configs: List[Dict[str, Any]], experiment_type: str = "npd") -> List[Dict[str, Any]]:
        """
        Run multiple experiments in batch.
        
        Args:
            configs: List of experiment configurations
            experiment_type: Type of experiments
            
        Returns:
            List of results
        """
        output_dir = self.base_output_dir / "batch_runs"
        batch_runner = BatchRunner(output_dir)
        
        results = batch_runner.run_batch(configs, experiment_type)
        
        # Generate comparative visualizations
        self._generate_batch_visualizations(results, batch_runner.figures_dir)
        
        return results
    
    def run_comparative_analysis(self, base_config: Dict[str, Any], agent_counts: List[int]) -> Dict[str, Any]:
        """
        Run comparative analysis across different agent counts.
        
        Args:
            base_config: Base experiment configuration
            agent_counts: List of agent counts to test
            
        Returns:
            Comparative analysis results
        """
        output_dir = self.base_output_dir / "comparative_analysis"
        results = []
        
        for n_agents in agent_counts:
            self.logger.info(f"Running experiment with {n_agents} agents")
            
            # Modify config for n agents
            config = base_config.copy()
            config['name'] = f"{base_config.get('name', 'experiment')}_{n_agents}_agents"
            
            # Scale up agent configuration
            if 'agents' in config:
                original_agents = config['agents']
                config['agents'] = self._scale_agents(original_agents, n_agents)
            
            # Run experiment
            runner = ExperimentRunner(output_dir, config['name'])
            result = runner.run_npd_experiment(config)
            result['n_agents'] = n_agents
            results.append(result)
        
        # Generate comparative analysis
        analysis = self._analyze_scaling(results)
        
        # Save comparative analysis
        analysis_path = output_dir / "comparative_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate comparative plots
        self._generate_comparative_plots(analysis, output_dir / "figures")
        
        return analysis
    
    def run_parameter_sweep(self, sweep_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run parameter sweep for optimization.
        
        Args:
            sweep_config: Parameter sweep configuration
            
        Returns:
            Sweep results
        """
        output_dir = self.base_output_dir / "parameter_sweeps"
        
        # Extract sweep parameters
        base_config = sweep_config['base_config']
        parameters = sweep_config['parameters']
        
        results = []
        param_combinations = self._generate_param_combinations(parameters)
        
        for i, param_set in enumerate(param_combinations):
            self.logger.info(f"Running parameter set {i+1}/{len(param_combinations)}: {param_set}")
            
            # Create config with parameters
            config = self._apply_parameters(base_config, param_set)
            config['name'] = f"sweep_{i+1}"
            
            # Run experiment
            runner = ExperimentRunner(output_dir, config['name'])
            result = runner.run_npd_experiment(config)
            result['parameters'] = param_set
            results.append(result)
        
        # Analyze sweep results
        sweep_analysis = self._analyze_sweep(results, parameters)
        
        # Save analysis
        analysis_path = output_dir / "sweep_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(sweep_analysis, f, indent=2)
        
        # Generate sweep visualizations
        self._generate_sweep_plots(sweep_analysis, output_dir / "figures")
        
        return sweep_analysis
    
    def _scale_agents(self, original_agents: List[Dict], n_agents: int) -> List[Dict]:
        """Scale agent configuration to n agents."""
        if not original_agents:
            return []
        
        # Simple scaling: repeat pattern
        scaled_agents = []
        for i in range(n_agents):
            agent_template = original_agents[i % len(original_agents)]
            agent = agent_template.copy()
            agent['id'] = i
            scaled_agents.append(agent)
        
        return scaled_agents
    
    def _generate_param_combinations(self, parameters: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations for sweep."""
        import itertools
        
        keys = list(parameters.keys())
        values = [parameters[k] for k in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _apply_parameters(self, base_config: Dict, params: Dict) -> Dict:
        """Apply parameters to base configuration."""
        config = json.loads(json.dumps(base_config))  # Deep copy
        
        for param_path, value in params.items():
            # Parse path (e.g., "agents.0.learning_rate")
            parts = param_path.split('.')
            current = config
            
            for part in parts[:-1]:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            # Set value
            last_part = parts[-1]
            if last_part.isdigit():
                current[int(last_part)] = value
            else:
                current[last_part] = value
        
        return config
    
    def _analyze_scaling(self, results: List[Dict]) -> Dict:
        """Analyze how metrics scale with agent count."""
        analysis = {
            'agent_counts': [],
            'avg_cooperation': [],
            'avg_score': [],
            'score_variance': [],
            'convergence_time': []
        }
        
        for result in results:
            n_agents = result['n_agents']
            analysis['agent_counts'].append(n_agents)
            analysis['avg_cooperation'].append(result.get('average_cooperation', 0))
            
            # Calculate average score
            scores = [stat['total_score'] for stat in result['agent_stats']]
            analysis['avg_score'].append(sum(scores) / len(scores))
            
            # Calculate score variance
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            analysis['score_variance'].append(variance ** 0.5)
            
            # Estimate convergence time (simplified)
            history = result.get('history', [])
            convergence_round = self._estimate_convergence(history)
            analysis['convergence_time'].append(convergence_round)
        
        return analysis
    
    def _estimate_convergence(self, history: List[Dict]) -> int:
        """Estimate when cooperation rate converges."""
        if len(history) < 10:
            return len(history)
        
        # Simple method: when variance becomes small
        window_size = 10
        threshold = 0.05
        
        for i in range(window_size, len(history)):
            window = [h['cooperation_rate'] for h in history[i-window_size:i]]
            avg = sum(window) / len(window)
            variance = sum((x - avg) ** 2 for x in window) / len(window)
            
            if variance < threshold:
                return i
        
        return len(history)
    
    def _analyze_sweep(self, results: List[Dict], parameters: Dict) -> Dict:
        """Analyze parameter sweep results."""
        analysis = {
            'parameters': parameters,
            'results': [],
            'best_params': {},
            'parameter_effects': {}
        }
        
        # Find best parameters
        best_score = -float('inf')
        best_cooperation = -float('inf')
        
        for result in results:
            params = result['parameters']
            scores = [stat['total_score'] for stat in result['agent_stats']]
            avg_score = sum(scores) / len(scores)
            avg_coop = result.get('average_cooperation', 0)
            
            analysis['results'].append({
                'parameters': params,
                'avg_score': avg_score,
                'avg_cooperation': avg_coop
            })
            
            if avg_score > best_score:
                best_score = avg_score
                analysis['best_params']['score'] = params
            
            if avg_coop > best_cooperation:
                best_cooperation = avg_coop
                analysis['best_params']['cooperation'] = params
        
        # Analyze parameter effects
        for param_name in parameters:
            analysis['parameter_effects'][param_name] = self._analyze_parameter_effect(
                results, param_name
            )
        
        return analysis
    
    def _analyze_parameter_effect(self, results: List[Dict], param_name: str) -> Dict:
        """Analyze the effect of a single parameter."""
        effect = {'values': [], 'avg_scores': [], 'avg_cooperation': []}
        
        # Group by parameter value
        grouped = {}
        for result in results:
            value = result['parameters'].get(param_name)
            if value not in grouped:
                grouped[value] = []
            grouped[value].append(result)
        
        # Calculate averages
        for value, group_results in sorted(grouped.items()):
            effect['values'].append(value)
            
            all_scores = []
            all_coop = []
            
            for result in group_results:
                scores = [stat['total_score'] for stat in result['agent_stats']]
                all_scores.extend(scores)
                all_coop.append(result.get('average_cooperation', 0))
            
            effect['avg_scores'].append(sum(all_scores) / len(all_scores))
            effect['avg_cooperation'].append(sum(all_coop) / len(all_coop))
        
        return effect
    
    def _generate_visualizations(self, results: Dict, output_dir: Path):
        """Generate standard visualizations for single experiment."""
        plot_gen = PlotGenerator(output_dir)
        plot_gen.generate_all_plots(results)
    
    def _generate_batch_visualizations(self, results: List[Dict], output_dir: Path):
        """Generate visualizations for batch experiments."""
        plot_gen = PlotGenerator(output_dir)
        plot_gen.generate_batch_comparison(results)
    
    def _generate_comparative_plots(self, analysis: Dict, output_dir: Path):
        """Generate plots for comparative analysis."""
        plot_gen = PlotGenerator(output_dir)
        plot_gen.generate_scaling_plots(analysis)
    
    def _generate_sweep_plots(self, analysis: Dict, output_dir: Path):
        """Generate plots for parameter sweep."""
        plot_gen = PlotGenerator(output_dir)
        plot_gen.generate_sweep_plots(analysis)


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(description="N-Person Prisoner's Dilemma Simulator")
    
    parser.add_argument('command', choices=['single', 'batch', 'compare', 'sweep'],
                       help='Type of experiment to run')
    parser.add_argument('-c', '--config', required=True,
                       help='Path to configuration file')
    parser.add_argument('-o', '--output', default='results',
                       help='Output directory')
    parser.add_argument('-t', '--type', choices=['npd', 'pairwise'], default='npd',
                       help='Game type')
    parser.add_argument('-n', '--agent-counts', nargs='+', type=int,
                       help='Agent counts for comparative analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create simulator
    simulator = NPDSimulator(args.output)
    
    # Run appropriate command
    if args.command == 'single':
        results = simulator.run_single_experiment(config, args.type)
        print(f"Experiment completed. Results saved to {args.output}")
        
    elif args.command == 'batch':
        if not isinstance(config, list):
            print("Batch mode requires a list of configurations")
            sys.exit(1)
        results = simulator.run_batch_experiments(config, args.type)
        print(f"Batch completed. {len(results)} experiments run.")
        
    elif args.command == 'compare':
        if not args.agent_counts:
            args.agent_counts = [3, 5, 10, 20]
        analysis = simulator.run_comparative_analysis(config, args.agent_counts)
        print(f"Comparative analysis completed for {args.agent_counts} agents")
        
    elif args.command == 'sweep':
        results = simulator.run_parameter_sweep(config)
        print(f"Parameter sweep completed. {len(results['results'])} combinations tested.")


if __name__ == "__main__":
    main()