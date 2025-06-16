"""
Enhanced Q-Learning Experiment Runner

This script systematically tests the four improvements to Q-learning:
1. Decaying epsilon
2. State representation excluding self
3. Longer training
4. Opponent modeling

It runs experiments for individual factors and combined configurations.
"""

import random
import time
import json
from collections import defaultdict

# Import game structures
from main_neighbourhood import NPersonPrisonersDilemma, NPersonAgent, NPERSON_COOPERATE

# Import extended agents
from extended_agents import (
    ExtendedNPersonAgent, QLearningNPersonWrapper
)

# Import enhanced Q-learning
from enhanced_qlearning_agents import EnhancedQLearningAgent, OpponentModelingQLearning


class ExperimentRunner:
    """Runs systematic experiments on Q-learning improvements."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
    
    def create_agents(self, strategies, ql_config=None):
        """Create agents with specified Q-learning configuration."""
        agents = []
        for i, strategy in enumerate(strategies):
            if strategy == "QL":
                if ql_config is None:
                    ql_config = {}
                
                # Create enhanced Q-learning agent
                ql_agent = EnhancedQLearningAgent(i, **ql_config)
                agents.append(QLearningNPersonWrapper(i, ql_agent))
            else:
                agents.append(ExtendedNPersonAgent(i, strategy, exploration_rate=0.01))
        return agents
    
    def run_single_experiment(self, scenario_name, strategies, ql_config, num_rounds=1000):
        """Run a single experiment with given configuration."""
        if self.verbose:
            print(f"  Running: {scenario_name} with {ql_config}")
        
        # Create agents
        agents = self.create_agents(strategies, ql_config)
        
        # Track metrics
        metrics = {
            'cooperation_history': {i: [] for i in range(len(agents))},
            'scores': {i: [] for i in range(len(agents))},
            'group_cooperation': [],
            'epsilon_history': [],
            'q_table_size': [],
            'state_visits': defaultdict(int)
        }
        
        # Find Q-learning agent
        ql_idx = None
        for i, strategy in enumerate(strategies):
            if strategy == "QL":
                ql_idx = i
                break
        
        # Run simulation
        prev_coop_ratio = None
        
        for round_num in range(num_rounds):
            # Collect actions
            actions = {}
            for agent in agents:
                _, actual = agent.choose_action(prev_coop_ratio, round_num)
                actions[agent.agent_id] = actual
            
            # Calculate cooperation
            num_coops = sum(1 for action in actions.values() if action == NPERSON_COOPERATE)
            prev_coop_ratio = num_coops / len(agents)
            metrics['group_cooperation'].append(prev_coop_ratio)
            
            # Calculate payoffs
            for agent in agents:
                my_action = actions[agent.agent_id]
                others_coop = num_coops - (1 if my_action == NPERSON_COOPERATE else 0)
                
                # Payoff calculation
                if my_action == NPERSON_COOPERATE:
                    payoff = 0 + 3 * (others_coop / (len(agents) - 1))
                else:
                    payoff = 1 + 4 * (others_coop / (len(agents) - 1))
                
                agent.record_round_outcome(my_action, payoff)
                
                # Track metrics
                is_coop = 1 if my_action == NPERSON_COOPERATE else 0
                metrics['cooperation_history'][agent.agent_id].append(is_coop)
                metrics['scores'][agent.agent_id].append(payoff)
            
            # Track Q-learning specific metrics
            if ql_idx is not None:
                ql_agent = agents[ql_idx].qlearning_agent
                metrics['epsilon_history'].append(ql_agent.epsilon)
                metrics['q_table_size'].append(len(ql_agent.q_table))
                
                # Track state visits
                if hasattr(ql_agent, 'last_state') and ql_agent.last_state:
                    metrics['state_visits'][ql_agent.last_state] += 1
        
        # Calculate final statistics
        final_stats = {}
        for i, agent in enumerate(agents):
            final_stats[i] = {
                'strategy': strategies[i],
                'total_score': agent.total_score,
                'cooperation_rate': agent.get_cooperation_rate(),
                'avg_score_per_round': agent.total_score / num_rounds
            }
        
        # Q-learning specific analysis
        if ql_idx is not None:
            ql_agent = agents[ql_idx].qlearning_agent
            final_stats[ql_idx]['q_table'] = dict(ql_agent.q_table) if hasattr(ql_agent, 'q_table') else {}
            final_stats[ql_idx]['final_epsilon'] = ql_agent.epsilon
            final_stats[ql_idx]['step_count'] = ql_agent.step_count
        
        return {
            'final_stats': final_stats,
            'metrics': metrics,
            'ql_idx': ql_idx,
            'config': ql_config
        }
    
    def test_epsilon_decay_impact(self, num_rounds=1000):
        """Test impact of different epsilon decay strategies."""
        print("Testing Epsilon Decay Impact...")
        
        scenario = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        scenario_name, strategies = scenario
        
        epsilon_configs = [
            {"name": "no_decay", "epsilon": 0.1, "epsilon_decay": 1.0},
            {"name": "slow_decay", "epsilon": 0.1, "epsilon_decay": 0.999, "epsilon_min": 0.01},
            {"name": "fast_decay", "epsilon": 0.1, "epsilon_decay": 0.99, "epsilon_min": 0.01},
            {"name": "very_fast_decay", "epsilon": 0.1, "epsilon_decay": 0.95, "epsilon_min": 0.01},
            {"name": "low_constant", "epsilon": 0.01, "epsilon_decay": 1.0},
        ]
        
        results = {}
        for config in epsilon_configs:
            name = config.pop("name")
            result = self.run_single_experiment(scenario_name, strategies, config, num_rounds)
            results[name] = result
            
            if self.verbose:
                ql_stats = result['final_stats'][result['ql_idx']]
                print(f"    {name}: Coop={ql_stats['cooperation_rate']:.3f}, "
                      f"Score={ql_stats['total_score']:.1f}, "
                      f"Final ε={ql_stats['final_epsilon']:.4f}")
        
        return results
    
    def test_state_representation_impact(self, num_rounds=1000):
        """Test impact of different state representations."""
        print("Testing State Representation Impact...")
        
        # Test on multiple scenarios
        scenarios = [
            ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
            ("QL vs AllD vs AllD", ["QL", "AllD", "AllD"]),
        ]
        
        state_configs = [
            {"name": "include_self_basic", "exclude_self": False, "state_type": "basic"},
            {"name": "exclude_self_basic", "exclude_self": True, "state_type": "basic"},
            {"name": "include_self_fine", "exclude_self": False, "state_type": "fine"},
            {"name": "exclude_self_fine", "exclude_self": True, "state_type": "fine"},
            {"name": "exclude_self_coarse", "exclude_self": True, "state_type": "coarse"},
        ]
        
        all_results = {}
        for scenario_name, strategies in scenarios:
            scenario_results = {}
            
            for config in state_configs:
                name = config.pop("name")
                result = self.run_single_experiment(scenario_name, strategies, config, num_rounds)
                scenario_results[name] = result
                
                if self.verbose:
                    ql_stats = result['final_stats'][result['ql_idx']]
                    print(f"    {scenario_name} - {name}: "
                          f"Coop={ql_stats['cooperation_rate']:.3f}, "
                          f"Score={ql_stats['total_score']:.1f}")
            
            all_results[scenario_name] = scenario_results
        
        return all_results
    
    def test_training_duration_impact(self):
        """Test impact of different training durations."""
        print("Testing Training Duration Impact...")
        
        scenario = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        scenario_name, strategies = scenario
        
        duration_configs = [
            {"rounds": 100, "name": "short"},
            {"rounds": 1000, "name": "medium"},
            {"rounds": 10000, "name": "long"},
            {"rounds": 50000, "name": "very_long"},
        ]
        
        # Use exclude_self configuration for better learning
        base_config = {"exclude_self": True, "epsilon_decay": 0.999, "epsilon_min": 0.01}
        
        results = {}
        for config in duration_configs:
            rounds = config["rounds"]
            name = config["name"]
            
            result = self.run_single_experiment(scenario_name, strategies, base_config, rounds)
            results[name] = result
            
            if self.verbose:
                ql_stats = result['final_stats'][result['ql_idx']]
                print(f"    {name} ({rounds} rounds): "
                      f"Coop={ql_stats['cooperation_rate']:.3f}, "
                      f"Score={ql_stats['total_score']:.1f}")
        
        return results
    
    def test_opponent_modeling_impact(self, num_rounds=1000):
        """Test impact of opponent modeling."""
        print("Testing Opponent Modeling Impact...")
        
        scenarios = [
            ("QL vs TFT vs AllD", ["QL", "pTFT-Threshold", "AllD"]),
            ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
        ]
        
        modeling_configs = [
            {"name": "no_modeling", "opponent_modeling": False},
            {"name": "with_modeling", "opponent_modeling": True},
        ]
        
        all_results = {}
        for scenario_name, strategies in scenarios:
            scenario_results = {}
            
            for config in modeling_configs:
                name = config.pop("name")
                # Use good base configuration
                full_config = {"exclude_self": True, "epsilon_decay": 0.999, "epsilon_min": 0.01}
                full_config.update(config)
                
                result = self.run_single_experiment(scenario_name, strategies, full_config, num_rounds)
                scenario_results[name] = result
                
                if self.verbose:
                    ql_stats = result['final_stats'][result['ql_idx']]
                    print(f"    {scenario_name} - {name}: "
                          f"Coop={ql_stats['cooperation_rate']:.3f}, "
                          f"Score={ql_stats['total_score']:.1f}")
            
            all_results[scenario_name] = scenario_results
        
        return all_results
    
    def test_combined_configurations(self, num_rounds=1000):
        """Test combined configurations of all improvements."""
        print("Testing Combined Configurations...")
        
        scenario = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        scenario_name, strategies = scenario
        
        combined_configs = [
            {
                "name": "baseline",
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": False,
                "state_type": "basic"
            },
            {
                "name": "exclude_self_only",
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": False,
                "state_type": "basic"
            },
            {
                "name": "epsilon_decay_only",
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 0.999,
                "epsilon_min": 0.01,
                "opponent_modeling": False,
                "state_type": "basic"
            },
            {
                "name": "optimal_exploitation",
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.001,
                "opponent_modeling": False,
                "state_type": "basic"
            },
            {
                "name": "robust_learning",
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "opponent_modeling": True,
                "state_type": "fine"
            },
            {
                "name": "all_improvements",
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.999,
                "epsilon_min": 0.001,
                "opponent_modeling": True,
                "state_type": "fine"
            }
        ]
        
        results = {}
        for config in combined_configs:
            name = config.pop("name")
            result = self.run_single_experiment(scenario_name, strategies, config, num_rounds)
            results[name] = result
            
            if self.verbose:
                ql_stats = result['final_stats'][result['ql_idx']]
                print(f"    {name}: Coop={ql_stats['cooperation_rate']:.3f}, "
                      f"Score={ql_stats['total_score']:.1f}, "
                      f"Final ε={ql_stats.get('final_epsilon', 'N/A')}")
        
        return results
    
    def run_full_experiment_suite(self):
        """Run the complete experiment suite."""
        print("Running Full Enhanced Q-Learning Experiment Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Individual factor testing
        print("\n" + "=" * 40)
        print("PHASE 1: Individual Factor Testing")
        print("=" * 40)
        
        self.results['epsilon_decay'] = self.test_epsilon_decay_impact()
        self.results['state_representation'] = self.test_state_representation_impact()
        self.results['training_duration'] = self.test_training_duration_impact()
        self.results['opponent_modeling'] = self.test_opponent_modeling_impact()
        
        # Phase 2: Combined configurations
        print("\n" + "=" * 40)
        print("PHASE 2: Combined Configuration Testing")
        print("=" * 40)
        
        self.results['combined'] = self.test_combined_configurations()
        
        # Save results
        timestamp = int(time.time())
        filename = f"enhanced_qlearning_results_{timestamp}.json"
        
        # Convert any non-serializable objects for JSON
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert objects to string
            else:
                return obj
        
        serializable_results = convert_for_json(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        elapsed = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"Experiment suite completed in {elapsed:.1f} seconds")
        print(f"Results saved to {filename}")
        print("=" * 60)
        
        return self.results
    
    def analyze_exploitation_effectiveness(self):
        """Analyze how well different configurations exploit AllC opponents."""
        print("\nAnalyzing Exploitation Effectiveness...")
        print("-" * 40)
        
        if 'combined' not in self.results:
            print("No combined results available. Run experiments first.")
            return
        
        combined_results = self.results['combined']
        
        print("Configuration Analysis (vs AllC vs AllC):")
        print(f"{'Configuration':<20} {'Coop Rate':<10} {'Score':<8} {'Exploitation':<12}")
        print("-" * 52)
        
        theoretical_max_score = 1000 * (1 + 4 * 1.0)  # Always defect vs 2 cooperators
        
        for config_name, result in combined_results.items():
            ql_stats = result['final_stats'][result['ql_idx']]
            coop_rate = ql_stats['cooperation_rate']
            score = ql_stats['total_score']
            exploitation = (1 - coop_rate) * 100  # % defection
            
            print(f"{config_name:<20} {coop_rate:<10.3f} {score:<8.1f} {exploitation:<12.1f}%")
        
        print(f"\nTheoretical maximum score: {theoretical_max_score:.1f}")
        print(f"(Achieved by always defecting against 2 cooperators)")


def main():
    """Run the enhanced Q-learning experiments."""
    random.seed(42)  # For reproducibility
    
    runner = ExperimentRunner(verbose=True)
    results = runner.run_full_experiment_suite()
    runner.analyze_exploitation_effectiveness()
    
    return results


if __name__ == "__main__":
    main()