"""
Export N-Person IPD simulations to CSV format.
Compatible with the existing analysis tools.
"""

import os
import csv
import json
from typing import List, Dict, Any
from simple_npd import NPDAgent, Strategy, NPrisonersDilemmaGame, NPersonSimulation, Action


class NPDCSVExporter:
    """Export N-Person simulation results to CSV files."""
    
    def __init__(self, results_dir: str = "simple_results"):
        self.results_dir = results_dir
        
    def export_simulation(self, 
                         agents: List[NPDAgent],
                         game: NPrisonersDilemmaGame,
                         scenario_name: str,
                         num_rounds: int,
                         run_number: int = 0):
        """Export a complete N-Person simulation run to CSV files."""
        # Create directory structure
        scenario_dir = os.path.join(self.results_dir, scenario_name)
        run_dir = os.path.join(scenario_dir, f"run_{run_number:02d}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run simulation and collect data
        sim = NPersonSimulation(agents, game)
        round_data = self._run_and_collect_data(sim, scenario_name, run_number, num_rounds)
        
        # Export agents summary
        agents_file = os.path.join(run_dir, "experiment_results_agents.csv")
        self._export_agents(agents, scenario_name, run_number, agents_file)
        
        # Export rounds data
        rounds_file = os.path.join(run_dir, "experiment_results_rounds.csv")
        self._export_rounds(round_data, rounds_file)
        
        # Export network structure (fully connected for N-Person)
        network_file = os.path.join(run_dir, "experiment_results_network.json")
        self._export_network(agents, network_file)
        
        print(f"\nCSV files exported to: {run_dir}")
        
    def _run_and_collect_data(self, simulation: NPersonSimulation, scenario_name: str, 
                             run_number: int, num_rounds: int) -> List[Dict[str, Any]]:
        """Run simulation and collect round-by-round data."""
        round_data = []
        
        # Run simulation rounds
        for round_num in range(num_rounds):
            # All agents decide simultaneously
            actions = {}
            for agent in simulation.agents:
                actions[agent.name] = agent.decide_action()
            
            # Calculate payoffs
            payoffs = simulation.game.calculate_payoffs(actions)
            
            # Update agent histories and collect data
            for i, agent in enumerate(simulation.agents):
                agent_action = actions[agent.name]
                agent_payoff = payoffs[agent.name]
                
                # Update agent
                agent.update_history(agent_action, actions, agent_payoff)
                
                # Collect round data
                round_data.append({
                    'scenario_name': scenario_name,
                    'run_number': run_number,
                    'round': round_num,
                    'agent_id': i,
                    'move': 'cooperate' if agent_action == Action.COOPERATE else 'defect',
                    'payoff': agent_payoff,
                    'strategy': agent.strategy.value.lower().replace(" ", "_").replace("-", "_")
                })
        
        return round_data
    
    def _export_agents(self, agents: List[NPDAgent], scenario_name: str, 
                      run_number: int, filename: str):
        """Export agent summary data."""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scenario_name', 'run_number', 'agent_id', 'strategy', 
                'final_score', 'final_q_values_avg', 'full_q_values',
                'exploration_rate', 'cooperation_rate'
            ])
            writer.writeheader()
            
            for i, agent in enumerate(agents):
                # Calculate cooperation rate
                coop_rate = 0.0
                if len(agent.action_history) > 0:
                    coop_count = sum(1 for a in agent.action_history if a == Action.COOPERATE)
                    coop_rate = coop_count / len(agent.action_history)
                
                writer.writerow({
                    'scenario_name': scenario_name,
                    'run_number': run_number,
                    'agent_id': i,
                    'strategy': agent.strategy.value.lower().replace(" ", "_").replace("-", "_"),
                    'final_score': agent.total_score,
                    'final_q_values_avg': '',  # Empty for non-RL agents
                    'full_q_values': '',  # Empty for non-RL agents
                    'exploration_rate': agent.exploration_rate,
                    'cooperation_rate': coop_rate
                })
    
    def _export_rounds(self, round_data: List[Dict[str, Any]], filename: str):
        """Export round-by-round data."""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scenario_name', 'run_number', 'round', 'agent_id', 
                'move', 'payoff', 'strategy'
            ])
            writer.writeheader()
            writer.writerows(round_data)
    
    def _export_network(self, agents: List[NPDAgent], filename: str):
        """Export network structure (fully connected for N-Person)."""
        network_data = {
            "network_type": "fully_connected",
            "interaction_mode": "n_person",
            "num_nodes": len(agents),
            "edges": []
        }
        
        # Create fully connected network
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                network_data["edges"].append([i, j])
        
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)


def create_scenarios() -> List[Dict[str, Any]]:
    """Create a variety of N-Person scenarios to test."""
    scenarios = [
        {
            'name': 'NPD_Classic_NoExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.0},
                {'name': 'Bob', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.0},
                {'name': 'Charlie', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0}
            ],
            'num_rounds': 20,
            'payoffs': {}  # Use default payoffs
        },
        {
            'name': 'NPD_Classic_WithExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.1},
                {'name': 'Bob', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.1},
                {'name': 'Charlie', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1}
            ],
            'num_rounds': 20,
            'payoffs': {}
        },
        {
            'name': 'NPD_AllTFT_NoExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0},
                {'name': 'Bob', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0},
                {'name': 'Charlie', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0}
            ],
            'num_rounds': 20,
            'payoffs': {}
        },
        {
            'name': 'NPD_AllTFT_VaryingExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.0},
                {'name': 'Bob', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'Charlie', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.2}
            ],
            'num_rounds': 20,
            'payoffs': {}
        },
        {
            'name': 'NPD_TwoCoopOneDefect_LowExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.05},
                {'name': 'Bob', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.05},
                {'name': 'Charlie', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.05}
            ],
            'num_rounds': 20,
            'payoffs': {}
        },
        {
            'name': 'NPD_HighCoopReward',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'Bob', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.1},
                {'name': 'Charlie', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.1}
            ],
            'num_rounds': 20,
            'payoffs': {'R': 4.0, 'S': 0.5, 'T': 4.5, 'P': 1.0}
        },
        {
            'name': 'NPD_HighExploration',
            'agents': [
                {'name': 'Alice', 'strategy': Strategy.ALWAYS_COOPERATE, 'exploration_rate': 0.3},
                {'name': 'Bob', 'strategy': Strategy.ALWAYS_DEFECT, 'exploration_rate': 0.3},
                {'name': 'Charlie', 'strategy': Strategy.TIT_FOR_TAT, 'exploration_rate': 0.3}
            ],
            'num_rounds': 20,
            'payoffs': {}
        }
    ]
    return scenarios


def main():
    """Run batch export of N-Person simulations."""
    exporter = NPDCSVExporter()
    scenarios = create_scenarios()
    num_runs = 10  # Number of runs per scenario
    
    print(f"Running {len(scenarios)} N-Person scenarios with {num_runs} runs each...")
    print(f"Total simulations: {len(scenarios) * num_runs}")
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Create agents
            agents = []
            for agent_config in scenario['agents']:
                agent = NPDAgent(
                    name=agent_config['name'],
                    strategy=agent_config['strategy'],
                    exploration_rate=agent_config['exploration_rate']
                )
                agents.append(agent)
            
            # Create game
            game = NPrisonersDilemmaGame(N=3, **scenario['payoffs'])
            
            # Export this run (without verbose output)
            exporter.export_simulation(
                agents, 
                game,
                scenario['name'], 
                scenario['num_rounds'],
                run_number=run
            )
    
    print(f"\n\nAll N-Person simulations exported to: {exporter.results_dir}")
    print("You can now run analyze_results.py to analyze the results!")


if __name__ == "__main__":
    main()
