"""
Research Experiments Runner
Generates TFT-centric and QLearning-centric experiment data with aggregated results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from scipy import stats
import logging

from npd_simulator.experiments.runners.experiment_runner import ExperimentRunner
from npd_simulator.profiles.profile_loader import ProfileLoader
from npd_simulator.utils.results_manager import ResultsManager
from npd_simulator.static_style_visualizer import StaticStyleVisualizer

# Ensure agents are registered
import npd_simulator.agents


class ResearchExperiments:
    """
    Runs comprehensive experiments for TFT-centric and QLearning-centric research.
    Outputs data in the exact format as existing results.
    """
    
    def __init__(self, num_runs: int = 20, num_rounds: int = 200, output_dir: str = "results"):
        self.num_runs = num_runs
        self.num_rounds = num_rounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # TFT-centric configurations
        self.tft_configs = {
            "3_TFT": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.0},
                {"id": 1, "type": "TFT", "exploration_rate": 0.0},
                {"id": 2, "type": "TFT", "exploration_rate": 0.0}
            ],
            "2_TFT-E__plus__1_AllD": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.1},
                {"id": 1, "type": "TFT", "exploration_rate": 0.1},
                {"id": 2, "type": "AllD", "exploration_rate": 0.0}
            ],
            "2_TFT__plus__1_AllD": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.0},
                {"id": 1, "type": "TFT", "exploration_rate": 0.0},
                {"id": 2, "type": "AllD", "exploration_rate": 0.0}
            ],
            "2_TFT-E__plus__1_AllC": [
                {"id": 0, "type": "TFT", "exploration_rate": 0.1},
                {"id": 1, "type": "TFT", "exploration_rate": 0.1},
                {"id": 2, "type": "AllC", "exploration_rate": 0.0}
            ]
        }
        
        # QLearning configurations will be generated dynamically
        self.other_agent_types = ["AllD", "AllC", "TFT", "TFT-E", "Random"]
        
    def run_tft_centric_experiments(self) -> Dict[str, Any]:
        """Run TFT-centric experiments for both pairwise and neighbourhood."""
        results = {
            "pairwise": {},
            "neighbourhood": {}
        }
        
        for config_name, agent_config in self.tft_configs.items():
            self.logger.info(f"Running TFT-centric: {config_name}")
            
            # Run pairwise experiments
            pairwise_data = self._run_multiple_experiments(
                agent_config, "pairwise", config_name
            )
            results["pairwise"][config_name] = pairwise_data
            
            # Run neighbourhood (N-person) experiments
            neighbourhood_data = self._run_multiple_experiments(
                agent_config, "nperson", config_name
            )
            results["neighbourhood"][config_name] = neighbourhood_data
            
            # Save aggregated CSVs
            self._save_aggregated_csv(
                pairwise_data, "pairwise", "tft_cooperation", config_name
            )
            self._save_aggregated_csv(
                neighbourhood_data, "neighbourhood", "tft_cooperation", config_name
            )
        
        # Save summary CSVs
        self._save_summary_csv(results["pairwise"], "pairwise", "tft_cooperation")
        self._save_summary_csv(results["neighbourhood"], "neighbourhood", "tft_cooperation")
        
        return results
    
    def run_qlearning_centric_experiments(self) -> Dict[str, Any]:
        """Run QLearning-centric experiments with all combinations."""
        results = {
            "2QL": {},
            "1QL": {},
            "comparison": {}
        }
        
        # 2 QL agents experiments
        for other_type in self.other_agent_types:
            # Basic QLearning
            config_name = f"2_QL__plus__1_{other_type}"
            agent_config = self._create_2ql_config("QLearning", other_type)
            
            self.logger.info(f"Running 2QL experiment: {config_name}")
            data = self._run_multiple_experiments(agent_config, "nperson", config_name)
            results["2QL"][config_name] = data
            self._save_aggregated_csv(data, "qlearning", "cooperation", config_name)
            
            # Enhanced QLearning
            config_name = f"2_EQL__plus__1_{other_type}"
            agent_config = self._create_2ql_config("EnhancedQLearning", other_type)
            
            self.logger.info(f"Running 2EQL experiment: {config_name}")
            data = self._run_multiple_experiments(agent_config, "nperson", config_name)
            results["2QL"][config_name] = data
            self._save_aggregated_csv(data, "qlearning", "cooperation", config_name)
        
        # 1 QL agent experiments
        for ql_type in ["QLearning", "EnhancedQLearning"]:
            ql_prefix = "QL" if ql_type == "QLearning" else "EQL"
            
            # All combinations of 2 other agents
            for i, type1 in enumerate(self.other_agent_types):
                for type2 in self.other_agent_types[i:]:
                    config_name = f"1_{ql_prefix}__plus__1_{type1}__plus__1_{type2}"
                    agent_config = self._create_1ql_config(ql_type, type1, type2)
                    
                    self.logger.info(f"Running 1QL experiment: {config_name}")
                    data = self._run_multiple_experiments(agent_config, "nperson", config_name)
                    results["1QL"][config_name] = data
                    self._save_aggregated_csv(data, "qlearning", "cooperation", config_name)
        
        # Save summary CSVs
        self._save_summary_csv(results["2QL"], "qlearning", "2ql_cooperation")
        self._save_summary_csv(results["1QL"], "qlearning", "1ql_cooperation")
        
        return results
    
    def _create_2ql_config(self, ql_type: str, other_type: str) -> List[Dict]:
        """Create configuration for 2 QL agents + 1 other."""
        config = [
            self._get_ql_agent_config(ql_type, 0),
            self._get_ql_agent_config(ql_type, 1),
            self._get_other_agent_config(other_type, 2)
        ]
        return config
    
    def _create_1ql_config(self, ql_type: str, type1: str, type2: str) -> List[Dict]:
        """Create configuration for 1 QL agent + 2 others."""
        config = [
            self._get_ql_agent_config(ql_type, 0),
            self._get_other_agent_config(type1, 1),
            self._get_other_agent_config(type2, 2)
        ]
        return config
    
    def _get_ql_agent_config(self, ql_type: str, agent_id: int) -> Dict:
        """Get QLearning agent configuration."""
        if ql_type == "QLearning":
            return {
                "id": agent_id,
                "type": "QLearning",
                "learning_rate": 0.1,
                "discount_factor": 0.9,
                "epsilon": 0.1,
                "exploration_rate": 0.0
            }
        else:  # EnhancedQLearning
            return {
                "id": agent_id,
                "type": "EnhancedQLearning",
                "learning_rate": 0.1,
                "discount_factor": 0.95,
                "epsilon": 0.15,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "exploration_rate": 0.0
            }
    
    def _get_other_agent_config(self, agent_type: str, agent_id: int) -> Dict:
        """Get configuration for non-QL agents."""
        if agent_type == "TFT-E":
            return {"id": agent_id, "type": "TFT", "exploration_rate": 0.1}
        elif agent_type == "Random":
            return {"id": agent_id, "type": "Random", "cooperation_probability": 0.5}
        else:
            return {"id": agent_id, "type": agent_type, "exploration_rate": 0.0}
    
    def _run_multiple_experiments(self, agent_config: List[Dict], 
                                game_type: str, config_name: str) -> Dict[str, Any]:
        """Run multiple experiments and collect data."""
        runner = ExperimentRunner()
        all_cooperation_rates = []
        all_agent_scores = []
        
        for run in range(self.num_runs):
            # Create experiment config
            exp_config = {
                "name": f"{config_name}_run_{run}",
                "agents": agent_config,
                "num_rounds": self.num_rounds,
                "rounds_per_pair": self.num_rounds
            }
            
            # Run experiment
            if game_type == "pairwise":
                results = runner.run_pairwise_experiment(exp_config)
            else:  # nperson
                results = runner.run_npd_experiment(exp_config)
            
            # Extract cooperation rates per round
            cooperation_by_round = self._extract_cooperation_by_round(results)
            all_cooperation_rates.append(cooperation_by_round)
            
            # Extract final scores
            if "agent_stats" in results:
                scores = {stat["agent_id"]: stat["total_score"] 
                         for stat in results["agent_stats"]}
                all_agent_scores.append(scores)
        
        return {
            "cooperation_rates": all_cooperation_rates,
            "agent_scores": all_agent_scores,
            "config": agent_config,
            "game_type": game_type
        }
    
    def _extract_cooperation_by_round(self, results: Dict) -> List[float]:
        """Extract cooperation rate for each round from results."""
        cooperation_rates = []
        
        if "history" in results:
            # Sort by round number to ensure correct order
            sorted_history = sorted(results["history"], key=lambda x: x.get("round", 0))
            
            for round_data in sorted_history:
                # For TFT-centric, extract TFT agent cooperation
                if "agents" in round_data:
                    tft_coops = []
                    for agent_id, agent_data in round_data["agents"].items():
                        if agent_data.get("type") in ["TFT", "pTFT"]:
                            if "action" in agent_data:
                                tft_coops.append(1 - agent_data["action"])  # 0=coop, 1=defect
                    
                    if tft_coops:
                        cooperation_rates.append(np.mean(tft_coops))
                    else:
                        cooperation_rates.append(round_data.get("cooperation_rate", 0))
                else:
                    cooperation_rates.append(round_data.get("cooperation_rate", 0))
        
        # Ensure we have exactly num_rounds values
        while len(cooperation_rates) < self.num_rounds:
            cooperation_rates.append(0.0)
        
        return cooperation_rates[:self.num_rounds]
    
    def _save_aggregated_csv(self, data: Dict, exp_type: str, 
                           metric: str, config_name: str):
        """Save aggregated data in the exact format of existing results."""
        cooperation_data = np.array(data["cooperation_rates"])
        
        # Calculate statistics
        means = np.mean(cooperation_data, axis=0)
        stds = np.std(cooperation_data, axis=0)
        
        # Calculate 95% confidence intervals
        n = len(cooperation_data)
        ci_margin = 1.96 * stds / np.sqrt(n)
        lower_ci = means - ci_margin
        upper_ci = means + ci_margin
        
        # Create DataFrame
        df_data = {
            "Round": list(range(1, self.num_rounds + 1)),
            "Mean_Cooperation_Rate": means,
            "Std_Dev": stds,
            "Lower_95_CI": lower_ci,
            "Upper_95_CI": upper_ci
        }
        
        # Add individual run columns
        for i in range(self.num_runs):
            df_data[f"Run_{i+1}"] = cooperation_data[i]
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        filename = f"{exp_type}_{metric}_{config_name}_aggregated.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved aggregated data to {filepath}")
    
    def _save_summary_csv(self, all_data: Dict, exp_type: str, metric: str):
        """Save summary CSV comparing all experiments."""
        summary_data = {"Round": list(range(1, self.num_rounds + 1))}
        
        for config_name, data in all_data.items():
            cooperation_data = np.array(data["cooperation_rates"])
            means = np.mean(cooperation_data, axis=0)
            stds = np.std(cooperation_data, axis=0)
            
            summary_data[f"{config_name}_mean"] = means
            summary_data[f"{config_name}_std"] = stds
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        filename = f"{exp_type}_{metric}_all_experiments_summary.csv"
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved summary data to {filepath}")
    
    def generate_comparative_analysis(self, tft_results: Dict, ql_results: Dict):
        """Generate comparative analysis showing tragic valley vs reciprocity hill."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "tft_analysis": self._analyze_tft_results(tft_results),
            "ql_analysis": self._analyze_ql_results(ql_results),
            "comparative_insights": self._generate_insights(tft_results, ql_results)
        }
        
        # Save analysis
        analysis_path = self.output_dir / "comparative_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _analyze_tft_results(self, results: Dict) -> Dict:
        """Analyze TFT results for tragic valley vs reciprocity hill."""
        analysis = {}
        
        for game_type in ["pairwise", "neighbourhood"]:
            game_analysis = {}
            
            for config_name, data in results[game_type].items():
                cooperation_data = np.array(data["cooperation_rates"])
                final_cooperation = np.mean(cooperation_data[:, -50:], axis=1)  # Last 50 rounds
                
                game_analysis[config_name] = {
                    "mean_final_cooperation": float(np.mean(final_cooperation)),
                    "std_final_cooperation": float(np.std(final_cooperation)),
                    "convergence_round": self._find_convergence(np.mean(cooperation_data, axis=0))
                }
            
            analysis[game_type] = game_analysis
        
        # Calculate differences between pairwise and neighbourhood
        analysis["differences"] = {}
        for config in self.tft_configs.keys():
            if config in results["pairwise"] and config in results["neighbourhood"]:
                pw_coop = analysis["pairwise"][config]["mean_final_cooperation"]
                nb_coop = analysis["neighbourhood"][config]["mean_final_cooperation"]
                analysis["differences"][config] = {
                    "cooperation_difference": nb_coop - pw_coop,
                    "shows_tragic_valley": pw_coop > nb_coop,
                    "shows_reciprocity_hill": nb_coop > pw_coop
                }
        
        return analysis
    
    def _analyze_ql_results(self, results: Dict) -> Dict:
        """Analyze QLearning results."""
        analysis = {
            "2QL": {},
            "1QL": {},
            "learning_effectiveness": {}
        }
        
        # Analyze 2QL experiments
        for config_name, data in results["2QL"].items():
            cooperation_data = np.array(data["cooperation_rates"])
            
            # Check learning curve
            early_coop = np.mean(cooperation_data[:, :50], axis=1)
            late_coop = np.mean(cooperation_data[:, -50:], axis=1)
            
            analysis["2QL"][config_name] = {
                "mean_early_cooperation": float(np.mean(early_coop)),
                "mean_late_cooperation": float(np.mean(late_coop)),
                "learning_improvement": float(np.mean(late_coop) - np.mean(early_coop)),
                "final_scores": self._analyze_scores(data["agent_scores"])
            }
        
        # Similar analysis for 1QL
        for config_name, data in results["1QL"].items():
            cooperation_data = np.array(data["cooperation_rates"])
            early_coop = np.mean(cooperation_data[:, :50], axis=1)
            late_coop = np.mean(cooperation_data[:, -50:], axis=1)
            
            analysis["1QL"][config_name] = {
                "mean_early_cooperation": float(np.mean(early_coop)),
                "mean_late_cooperation": float(np.mean(late_coop)),
                "learning_improvement": float(np.mean(late_coop) - np.mean(early_coop))
            }
        
        return analysis
    
    def _find_convergence(self, mean_cooperation: np.ndarray, 
                         threshold: float = 0.05) -> int:
        """Find round where cooperation converges."""
        if len(mean_cooperation) < 20:
            return -1
        
        for i in range(20, len(mean_cooperation) - 10):
            window = mean_cooperation[i:i+10]
            if np.std(window) < threshold:
                return i
        
        return -1
    
    def _analyze_scores(self, all_scores: List[Dict]) -> Dict:
        """Analyze agent scores."""
        if not all_scores:
            return {}
        
        # Average scores by agent
        agent_scores = {}
        for scores in all_scores:
            for agent_id, score in scores.items():
                if agent_id not in agent_scores:
                    agent_scores[agent_id] = []
                agent_scores[agent_id].append(score)
        
        return {
            f"agent_{agent_id}": {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores))
            }
            for agent_id, scores in agent_scores.items()
        }
    
    def _generate_insights(self, tft_results: Dict, ql_results: Dict) -> Dict:
        """Generate comparative insights."""
        insights = {
            "tragic_valley_observed": False,
            "reciprocity_hill_observed": False,
            "ql_adaptation_success": {},
            "key_findings": []
        }
        
        # Check for tragic valley and reciprocity hill
        if "differences" in tft_results.get("tft_analysis", {}):
            for config, diff in tft_results["tft_analysis"]["differences"].items():
                if diff["shows_tragic_valley"]:
                    insights["tragic_valley_observed"] = True
                    insights["key_findings"].append(
                        f"Tragic valley observed in {config}: "
                        f"cooperation drops by {abs(diff['cooperation_difference']):.3f} "
                        "in neighbourhood vs pairwise"
                    )
                elif diff["shows_reciprocity_hill"]:
                    insights["reciprocity_hill_observed"] = True
                    insights["key_findings"].append(
                        f"Reciprocity hill observed in {config}: "
                        f"cooperation increases by {diff['cooperation_difference']:.3f} "
                        "in neighbourhood vs pairwise"
                    )
        
        # Analyze QL adaptation
        if "2QL" in ql_results.get("ql_analysis", {}):
            for config, analysis in ql_results["ql_analysis"]["2QL"].items():
                if analysis["learning_improvement"] > 0.1:
                    insights["ql_adaptation_success"][config] = "Successful adaptation"
                elif analysis["learning_improvement"] < -0.1:
                    insights["ql_adaptation_success"][config] = "Degraded performance"
                else:
                    insights["ql_adaptation_success"][config] = "Stable performance"
        
        return insights
    
    def run_all_experiments(self):
        """Run all experiments and generate complete results."""
        self.logger.info("Starting comprehensive experiment suite...")
        
        # Run TFT-centric experiments
        self.logger.info("Running TFT-centric experiments...")
        tft_results = self.run_tft_centric_experiments()
        
        # Run QLearning-centric experiments
        self.logger.info("Running QLearning-centric experiments...")
        ql_results = self.run_qlearning_centric_experiments()
        
        # Generate comparative analysis
        self.logger.info("Generating comparative analysis...")
        analysis = self.generate_comparative_analysis(tft_results, ql_results)
        
        # Generate visualizations
        self.logger.info("Generating visualizations...")
        self._generate_visualizations(tft_results, ql_results)
        
        self.logger.info("All experiments completed!")
        return {
            "tft_results": tft_results,
            "ql_results": ql_results,
            "analysis": analysis
        }
    
    def _generate_visualizations(self, tft_results: Dict, ql_results: Dict):
        """Generate visualization figures."""
        visualizer = StaticStyleVisualizer(self.output_dir)
        
        # Prepare data for static style visualization
        # Convert to format expected by visualizer
        experiment_results = {}
        
        # Add TFT pairwise results
        for config_name, data in tft_results["pairwise"].items():
            key = f"pairwise_{config_name}"
            experiment_results[key] = self._convert_to_visualizer_format(data)
        
        # Add TFT neighbourhood results
        for config_name, data in tft_results["neighbourhood"].items():
            key = f"neighbourhood_{config_name}"
            experiment_results[key] = self._convert_to_visualizer_format(data)
        
        # Generate standard figures
        visualizer.generate_all_figures(experiment_results)
    
    def _convert_to_visualizer_format(self, data: Dict) -> List[Dict]:
        """Convert data to format expected by StaticStyleVisualizer."""
        # Create mock results for each run
        results = []
        
        for i in range(self.num_runs):
            run_result = {
                "history": [],
                "num_rounds": self.num_rounds,
                "agent_stats": []
            }
            
            # Add history
            for round_num in range(self.num_rounds):
                round_data = {
                    "round": round_num,
                    "cooperation_rate": data["cooperation_rates"][i][round_num],
                    "agents": {}
                }
                run_result["history"].append(round_data)
            
            results.append(run_result)
        
        return results


def main():
    """Run all research experiments."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiments
    experiments = ResearchExperiments(
        num_runs=20,
        num_rounds=200,
        output_dir="results"
    )
    
    results = experiments.run_all_experiments()
    
    print("\nExperiments completed successfully!")
    print(f"Results saved to: {experiments.output_dir}")
    print("\nKey files generated:")
    print("- TFT cooperation CSVs (pairwise and neighbourhood)")
    print("- QLearning cooperation CSVs (2QL and 1QL variants)")
    print("- Summary comparison CSVs")
    print("- Comparative analysis JSON")
    print("- Visualization figures")


if __name__ == "__main__":
    main()