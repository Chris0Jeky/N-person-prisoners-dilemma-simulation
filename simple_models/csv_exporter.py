"""
CSV Export functionality for Simple IPD Simulation
=================================================
Exports simulation results in a format compatible with the main npdl framework.
"""

import csv
import os
from typing import List, Dict, Any
from datetime import datetime
from simple_ipd import Agent, Action, Simulation


class CSVExporter:
    """Export simulation results to CSV files compatible with npdl analysis tools."""
    
    def __init__(self, results_dir: str = "simple_results"):
        self.results_dir = results_dir
        
    def export_simulation(self, 
                         simulation: Simulation,
                         scenario_name: str,
                         num_rounds: int,
                         run_number: int = 0):
        """Export a complete simulation run to CSV files."""
        # Create directory structure
        scenario_dir = os.path.join(self.results_dir, scenario_name)
        run_dir = os.path.join(scenario_dir, f"run_{run_number:02d}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Collect round-by-round data
        round_data = self._collect_round_data(simulation, scenario_name, run_number, num_rounds)
        
        # Export agents summary
        agents_file = os.path.join(run_dir, "experiment_results_agents.csv")
        self._export_agents(simulation.agents, scenario_name, run_number, agents_file)
        
        # Export rounds data
        rounds_file = os.path.join(run_dir, "experiment_results_rounds.csv")
        self._export_rounds(round_data, rounds_file)
        
        # Export network structure (for pairwise, it's fully connected)
        network_file = os.path.join(run_dir, "experiment_results_network.json")
        self._export_network(simulation.agents, network_file)
        
        print(f"\nCSV files exported to: {run_dir}")
        
    def _collect_round_data(self, simulation: Simulation, scenario_name: str, 
                           run_number: int, num_rounds: int) -> List[Dict[str, Any]]:
        """Collect round-by-round data by simulating the game."""
        round_data = []
        
        # Reset agents for fresh simulation
        for agent in simulation.agents:
            agent.history.clear()
            agent.opponent_history.clear()
            agent.score = 0
        
        # Run simulation and collect data
        for round_num in range(num_rounds):
            # Store current round moves
            round_moves = {}
            round_payoffs = {}
            
            # Generate all pairs
            pairs = []
            for i in range(len(simulation.agents)):
                for j in range(i + 1, len(simulation.agents)):
                    pairs.append((simulation.agents[i], simulation.agents[j]))
            
            # Play each pair
            for agent1, agent2 in pairs:
                # Get actions
                action1 = agent1.decide_action(agent2.name)
                action2 = agent2.decide_action(agent1.name)
                
                # Get payoffs
                payoff1, payoff2 = simulation.game.get_payoffs(action1, action2)
                
                # Update scores
                agent1.score += payoff1
                agent2.score += payoff2
                
                # Update histories
                agent1.update_history(agent2.name, action1, action2)
                agent2.update_history(agent1.name, action2, action1)
                
                # Store moves and payoffs for this round
                if agent1.name not in round_moves:
                    round_moves[agent1.name] = action1
                    round_payoffs[agent1.name] = payoff1
                else:
                    round_payoffs[agent1.name] += payoff1
                    
                if agent2.name not in round_moves:
                    round_moves[agent2.name] = action2
                    round_payoffs[agent2.name] = payoff2
                else:
                    round_payoffs[agent2.name] += payoff2
            
            # Add to round data
            for agent in simulation.agents:
                round_data.append({
                    'scenario_name': scenario_name,
                    'run_number': run_number,
                    'round': round_num,
                    'agent_id': simulation.agents.index(agent),  # Use index as ID
                    'move': 'cooperate' if round_moves[agent.name] == Action.COOPERATE else 'defect',
                    'payoff': round_payoffs[agent.name],
                    'strategy': agent.strategy.value.lower().replace(" ", "_").replace("-", "_")
                })
        
        return round_data
    
    def _export_agents(self, agents: List[Agent], scenario_name: str, 
                      run_number: int, filename: str):
        """Export agent summary data."""
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scenario_name', 'run_number', 'agent_id', 'strategy', 
                'final_score', 'final_q_values_avg', 'full_q_values'
            ])
            writer.writeheader()
            
            for i, agent in enumerate(agents):
                writer.writerow({
                    'scenario_name': scenario_name,
                    'run_number': run_number,
                    'agent_id': i,
                    'strategy': agent.strategy.value.lower().replace(" ", "_").replace("-", "_"),
                    'final_score': agent.score,
                    'final_q_values_avg': '',  # Empty for non-RL agents
                    'full_q_values': ''  # Empty for non-RL agents
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
    
    def _export_network(self, agents: List[Agent], filename: str):
        """Export network structure (fully connected for simple IPD)."""
        import json
        
        network_data = {
            "network_type": "fully_connected",
            "num_nodes": len(agents),
            "edges": []
        }
        
        # Create fully connected network
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                network_data["edges"].append([i, j])
        
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)


def export_batch_simulations(scenarios: List[Dict[str, Any]], 
                           num_runs: int = 10,
                           results_dir: str = "simple_results"):
    """Export multiple scenarios with multiple runs each."""
    exporter = CSVExporter(results_dir)
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario['name']}")
        print(f"{'='*60}")
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs}")
            
            # Create agents
            agents = []
            for agent_config in scenario['agents']:
                agent = Agent(
                    name=agent_config['name'],
                    strategy=agent_config['strategy'],
                    exploration_rate=agent_config.get('exploration_rate', 0.0)
                )
                agents.append(agent)
            
            # Create game and simulation
            from simple_ipd import PrisonersDilemmaGame
            game = PrisonersDilemmaGame(**scenario.get('payoffs', {}))
            sim = Simulation(agents, game)
            
            # Export this run
            exporter.export_simulation(
                sim, 
                scenario['name'], 
                scenario['num_rounds'],
                run_number=run
            )
    
    print(f"\n\nAll simulations exported to: {results_dir}")
