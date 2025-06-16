"""
Batch Runner for running multiple experiments
"""

from typing import List, Dict, Any
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from npd_simulator.experiments.runners.experiment_runner import ExperimentRunner
from npd_simulator.utils.logging import setup_logger


class BatchRunner:
    """
    Runs multiple experiments in batch with optional parallelization.
    """
    
    def __init__(self, output_dir: str = "results/batch_runs"):
        """
        Initialize batch runner.
        
        Args:
            output_dir: Base directory for batch results
        """
        self.output_dir = Path(output_dir)
        self.batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = self.output_dir / f"batch_{self.batch_name}"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(f"BatchRunner_{self.batch_name}")
        self.figures_dir = self.batch_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
    def run_batch(self, 
                  configs: List[Dict[str, Any]], 
                  experiment_type: str = "npd",
                  parallel: bool = False,
                  max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Run a batch of experiments.
        
        Args:
            configs: List of experiment configurations
            experiment_type: Type of experiments ("npd" or "pairwise")
            parallel: Whether to run in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            List of results
        """
        self.logger.info(f"Starting batch of {len(configs)} experiments")
        
        if parallel:
            results = self._run_parallel(configs, experiment_type, max_workers)
        else:
            results = self._run_sequential(configs, experiment_type)
        
        # Save batch summary
        self._save_batch_summary(results)
        
        return results
    
    def _run_sequential(self, 
                       configs: List[Dict[str, Any]], 
                       experiment_type: str) -> List[Dict[str, Any]]:
        """Run experiments sequentially."""
        results = []
        
        for i, config in enumerate(configs):
            self.logger.info(f"Running experiment {i+1}/{len(configs)}: {config.get('name', 'unnamed')}")
            
            # Create experiment directory
            exp_name = config.get('name', f'experiment_{i+1}')
            exp_dir = self.batch_dir / exp_name
            
            # Run experiment
            runner = ExperimentRunner(str(exp_dir), exp_name)
            
            if experiment_type == "npd":
                result = runner.run_npd_experiment(config)
            else:
                result = runner.run_pairwise_experiment(config)
            
            results.append(result)
        
        return results
    
    def _run_parallel(self,
                     configs: List[Dict[str, Any]],
                     experiment_type: str,
                     max_workers: int) -> List[Dict[str, Any]]:
        """Run experiments in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {}
            
            for i, config in enumerate(configs):
                exp_name = config.get('name', f'experiment_{i+1}')
                exp_dir = self.batch_dir / exp_name
                
                future = executor.submit(
                    run_single_experiment,
                    config,
                    experiment_type,
                    str(exp_dir),
                    exp_name
                )
                future_to_config[future] = (i, config)
            
            # Collect results
            for future in as_completed(future_to_config):
                i, config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"Completed experiment {i+1}/{len(configs)}")
                except Exception as e:
                    self.logger.error(f"Experiment {i+1} failed: {e}")
                    results.append({'error': str(e), 'config': config})
        
        return results
    
    def _save_batch_summary(self, results: List[Dict[str, Any]]):
        """Save summary of batch results."""
        summary = {
            'batch_name': self.batch_name,
            'num_experiments': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'experiments': []
        }
        
        for result in results:
            exp_summary = {
                'name': result.get('config', {}).get('name', 'unnamed'),
                'num_agents': result.get('num_agents', 'N/A'),
                'num_rounds': result.get('num_rounds', 'N/A'),
                'avg_cooperation': result.get('average_cooperation', 'N/A')
            }
            
            if 'error' in result:
                exp_summary['error'] = result['error']
            else:
                # Add key metrics
                if 'agent_stats' in result:
                    scores = [stat['total_score'] for stat in result['agent_stats']]
                    exp_summary['avg_score'] = sum(scores) / len(scores) if scores else 0
                    exp_summary['score_range'] = max(scores) - min(scores) if scores else 0
            
            summary['experiments'].append(exp_summary)
        
        # Save summary
        summary_path = self.batch_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Batch summary saved to {summary_path}")


def run_single_experiment(config: Dict[str, Any],
                         experiment_type: str,
                         output_dir: str,
                         exp_name: str) -> Dict[str, Any]:
    """
    Helper function to run a single experiment (for parallel execution).
    """
    runner = ExperimentRunner(output_dir, exp_name)
    
    if experiment_type == "npd":
        return runner.run_npd_experiment(config)
    else:
        return runner.run_pairwise_experiment(config)