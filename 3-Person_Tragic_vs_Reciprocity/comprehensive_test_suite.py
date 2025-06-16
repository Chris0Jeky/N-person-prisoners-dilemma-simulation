"""
Comprehensive Test Suite for Enhanced Q-Learning

This script systematically tests all improvements to Q-learning exploitation:
1. Original implementations (Simple QL and NPDL QL) 
2. Enhanced Q-learning with individual improvements
3. Enhanced Q-learning with combined improvements
4. Different scenarios and training durations

Provides detailed comparison and analysis of results.
"""

import random
import time
from collections import defaultdict

# Import original implementations
from qlearning_agents import SimpleQLearningAgent, NPDLQLearningAgent

# Import enhanced implementations
from enhanced_qlearning_agents import EnhancedQLearningAgent

# Import supporting classes
from extended_agents import (
    ExtendedNPersonAgent, QLearningNPersonWrapper
)
from main_neighbourhood import NPERSON_COOPERATE


class ComprehensiveTestSuite:
    """Comprehensive testing of all Q-learning variants."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        self.test_scenarios = [
            ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"]),
            ("QL vs AllD vs AllD", ["QL", "AllD", "AllD"]),
            ("QL vs TFT vs AllD", ["QL", "pTFT-Threshold", "AllD"]),
            ("QL vs AllC vs AllD", ["QL", "AllC", "AllD"])
        ]
    
    def create_agents(self, strategies, ql_type="enhanced", ql_config=None):
        """Create agents with specified Q-learning configuration."""
        agents = []
        for i, strategy in enumerate(strategies):
            if strategy == "QL":
                if ql_type == "simple":
                    ql_agent = SimpleQLearningAgent(i, learning_rate=0.15, epsilon=0.1)
                elif ql_type == "npdl":
                    ql_agent = NPDLQLearningAgent(i, learning_rate=0.15, epsilon=0.1,
                                                state_type="proportion_discretized")
                elif ql_type == "enhanced":
                    if ql_config is None:
                        ql_config = {}
                    ql_agent = EnhancedQLearningAgent(i, **ql_config)
                else:
                    raise ValueError(f"Unknown QL type: {ql_type}")
                
                agents.append(QLearningNPersonWrapper(i, ql_agent))
            else:
                agents.append(ExtendedNPersonAgent(i, strategy, exploration_rate=0.01))
        return agents
    
    def run_single_test(self, test_name, strategies, ql_type, ql_config, num_rounds):
        """Run a single test configuration."""
        if self.verbose:
            config_str = f"{ql_type}" + (f" {ql_config}" if ql_config else "")
            print(f"  Running: {test_name} - {config_str} ({num_rounds} rounds)")
        
        # Create agents
        agents = self.create_agents(strategies, ql_type, ql_config)
        
        # Track metrics
        cooperation_history = []
        q_table_evolution = []
        epsilon_history = []
        
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
            cooperation_history.append(prev_coop_ratio)
            
            # Track Q-learning specific metrics
            if ql_idx is not None:
                ql_agent = agents[ql_idx].qlearning_agent
                if hasattr(ql_agent, 'epsilon'):
                    epsilon_history.append(ql_agent.epsilon)
                
                # Sample Q-table at regular intervals
                if round_num % 200 == 0 and hasattr(ql_agent, 'q_table'):
                    q_table_snapshot = {
                        'round': round_num,
                        'q_table': dict(ql_agent.q_table) if ql_agent.q_table else {}
                    }
                    q_table_evolution.append(q_table_snapshot)
            
            # Calculate payoffs
            for agent in agents:
                my_action = actions[agent.agent_id]
                others_coop = num_coops - (1 if my_action == NPERSON_COOPERATE else 0)
                
                if my_action == NPERSON_COOPERATE:
                    payoff = 0 + 3 * (others_coop / (len(agents) - 1))
                else:
                    payoff = 1 + 4 * (others_coop / (len(agents) - 1))
                
                agent.record_round_outcome(my_action, payoff)
        
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
            final_stats[ql_idx]['final_q_table'] = dict(ql_agent.q_table) if hasattr(ql_agent, 'q_table') else {}
            final_stats[ql_idx]['final_epsilon'] = getattr(ql_agent, 'epsilon', 'N/A')
            final_stats[ql_idx]['step_count'] = getattr(ql_agent, 'step_count', 'N/A')
        
        return {
            'test_name': test_name,
            'ql_type': ql_type,
            'ql_config': ql_config,
            'num_rounds': num_rounds,
            'final_stats': final_stats,
            'ql_idx': ql_idx,
            'cooperation_history': cooperation_history,
            'epsilon_history': epsilon_history,
            'q_table_evolution': q_table_evolution
        }
    
    def test_original_implementations(self, num_rounds=1000):
        """Test original Simple and NPDL Q-learning implementations."""
        print("\n" + "="*60)
        print("TESTING ORIGINAL IMPLEMENTATIONS")
        print("="*60)
        
        results = {}
        
        for scenario_name, strategies in self.test_scenarios:
            scenario_results = {}
            
            # Test Simple Q-learning
            result = self.run_single_test(scenario_name, strategies, "simple", None, num_rounds)
            scenario_results['simple'] = result
            
            # Test NPDL Q-learning  
            result = self.run_single_test(scenario_name, strategies, "npdl", None, num_rounds)
            scenario_results['npdl'] = result
            
            results[scenario_name] = scenario_results
        
        return results
    
    def test_individual_improvements(self, num_rounds=1000):
        """Test each improvement individually."""
        print("\n" + "="*60)
        print("TESTING INDIVIDUAL IMPROVEMENTS")
        print("="*60)
        
        # Focus on key exploitation scenario
        scenario_name, strategies = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        
        improvements = [
            ("baseline", {
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": False,
                "state_type": "basic"
            }),
            ("exclude_self", {
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": False,
                "state_type": "basic"
            }),
            ("epsilon_decay", {
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "opponent_modeling": False,
                "state_type": "basic"
            }),
            ("fine_state", {
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": False,
                "state_type": "fine"
            }),
            ("opponent_modeling", {
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": True,
                "state_type": "basic"
            })
        ]
        
        results = {}
        for improvement_name, config in improvements:
            result = self.run_single_test(scenario_name, strategies, "enhanced", config, num_rounds)
            results[improvement_name] = result
        
        return results
    
    def test_combined_improvements(self, num_rounds=1000):
        """Test combinations of improvements."""
        print("\n" + "="*60)
        print("TESTING COMBINED IMPROVEMENTS")
        print("="*60)
        
        scenario_name, strategies = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        
        combinations = [
            ("exclude_self + decay", {
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "opponent_modeling": False,
                "state_type": "basic"
            }),
            ("exclude_self + modeling", {
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 1.0,
                "opponent_modeling": True,
                "state_type": "basic"
            }),
            ("decay + modeling", {
                "exclude_self": False,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "opponent_modeling": True,
                "state_type": "basic"
            }),
            ("optimal_exploitation", {
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.001,
                "opponent_modeling": False,
                "state_type": "basic"
            }),
            ("all_improvements", {
                "exclude_self": True,
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.001,
                "opponent_modeling": True,
                "state_type": "fine"
            })
        ]
        
        results = {}
        for combination_name, config in combinations:
            result = self.run_single_test(scenario_name, strategies, "enhanced", config, num_rounds)
            results[combination_name] = result
        
        return results
    
    def test_training_duration_effects(self):
        """Test the effect of different training durations."""
        print("\n" + "="*60)
        print("TESTING TRAINING DURATION EFFECTS")
        print("="*60)
        
        scenario_name, strategies = ("QL vs AllC vs AllC", ["QL", "AllC", "AllC"])
        
        # Use optimal configuration
        optimal_config = {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "opponent_modeling": False,
            "state_type": "basic"
        }
        
        durations = [100, 500, 1000, 2000, 5000]
        results = {}
        
        for duration in durations:
            result = self.run_single_test(scenario_name, strategies, "enhanced", optimal_config, duration)
            results[f"{duration}_rounds"] = result
        
        return results
    
    def test_multiple_scenarios(self, num_rounds=1000):
        """Test optimal configuration across multiple scenarios."""
        print("\n" + "="*60)
        print("TESTING ACROSS MULTIPLE SCENARIOS")
        print("="*60)
        
        optimal_config = {
            "exclude_self": True,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "opponent_modeling": False,
            "state_type": "basic"
        }
        
        results = {}
        for scenario_name, strategies in self.test_scenarios:
            result = self.run_single_test(scenario_name, strategies, "enhanced", optimal_config, num_rounds)
            results[scenario_name] = result
        
        return results
    
    def analyze_exploitation_effectiveness(self, results, test_name):
        """Analyze how effectively different configurations exploit opponents."""
        print(f"\n{test_name} - Exploitation Analysis:")
        print("-" * 50)
        
        print(f"{'Configuration':<25} {'Coop Rate':<10} {'Score':<8} {'Exploit %':<10}")
        print("-" * 50)
        
        for config_name, result in results.items():
            if result['ql_idx'] is not None:
                ql_stats = result['final_stats'][result['ql_idx']]
                coop_rate = ql_stats['cooperation_rate']
                score = ql_stats['total_score']
                exploit_rate = (1 - coop_rate) * 100
                
                print(f"{config_name:<25} {coop_rate:<10.3f} {score:<8.0f} {exploit_rate:<10.1f}%")
    
    def analyze_q_table_evolution(self, result):
        """Analyze how Q-table evolves over time."""
        if not result['q_table_evolution']:
            return
        
        print(f"\nQ-Table Evolution for {result['test_name']}:")
        print("-" * 40)
        
        for snapshot in result['q_table_evolution']:
            round_num = snapshot['round']
            q_table = snapshot['q_table']
            print(f"\nRound {round_num}:")
            for state, values in q_table.items():
                if isinstance(values, dict):
                    coop_val = values.get('cooperate', 0)
                    defect_val = values.get('defect', 0)
                    preference = "DEFECT" if defect_val > coop_val else "COOPERATE"
                    print(f"  {state}: C={coop_val:.2f}, D={defect_val:.2f} -> {preference}")
    
    def run_comprehensive_suite(self):
        """Run the complete test suite."""
        print("COMPREHENSIVE Q-LEARNING TEST SUITE")
        print("="*60)
        print("Testing all improvements to Q-learning exploitation")
        print(f"Random seed: 42 (for reproducibility)")
        print("="*60)
        
        random.seed(42)  # For reproducibility
        start_time = time.time()
        
        # Store all results
        all_results = {}
        
        # 1. Test original implementations
        all_results['original'] = self.test_original_implementations(1000)
        
        # 2. Test individual improvements
        all_results['individual'] = self.test_individual_improvements(1000)
        self.analyze_exploitation_effectiveness(all_results['individual'], "Individual Improvements")
        
        # 3. Test combined improvements
        all_results['combined'] = self.test_combined_improvements(1000)
        self.analyze_exploitation_effectiveness(all_results['combined'], "Combined Improvements")
        
        # 4. Test training duration effects
        all_results['duration'] = self.test_training_duration_effects()
        self.analyze_exploitation_effectiveness(all_results['duration'], "Training Duration Effects")
        
        # 5. Test across scenarios
        all_results['scenarios'] = self.test_multiple_scenarios(1000)
        self.analyze_exploitation_effectiveness(all_results['scenarios'], "Multiple Scenarios")
        
        # Summary analysis
        self.print_comprehensive_summary(all_results)
        
        elapsed = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"Comprehensive test suite completed in {elapsed:.1f} seconds")
        print("="*60)
        
        return all_results
    
    def print_comprehensive_summary(self, all_results):
        """Print comprehensive summary of all tests."""
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY")
        print("="*70)
        
        # Find best performing configurations
        best_configs = {}
        
        # From combined improvements
        for config_name, result in all_results['combined'].items():
            if result['ql_idx'] is not None:
                ql_stats = result['final_stats'][result['ql_idx']]
                exploit_rate = (1 - ql_stats['cooperation_rate']) * 100
                best_configs[config_name] = {
                    'exploit_rate': exploit_rate,
                    'score': ql_stats['total_score'],
                    'config': result['ql_config']
                }
        
        # Sort by exploitation rate
        sorted_configs = sorted(best_configs.items(), key=lambda x: x[1]['exploit_rate'], reverse=True)
        
        print("\nTOP PERFORMING CONFIGURATIONS:")
        print(f"{'Rank':<5} {'Configuration':<20} {'Exploit %':<10} {'Score':<8}")
        print("-" * 50)
        
        for i, (config_name, data) in enumerate(sorted_configs[:5], 1):
            print(f"{i:<5} {config_name:<20} {data['exploit_rate']:<10.1f} {data['score']:<8.0f}")
        
        if sorted_configs:
            best_config = sorted_configs[0]
            print(f"\nBEST CONFIGURATION: {best_config[0]}")
            print(f"Exploitation Rate: {best_config[1]['exploit_rate']:.1f}%")
            print(f"Score: {best_config[1]['score']:.0f}")
            print(f"Config: {best_config[1]['config']}")
        
        # Compare with original implementations
        print(f"\nCOMPARISON WITH ORIGINAL IMPLEMENTATIONS:")
        if 'original' in all_results:
            original_results = all_results['original'].get('QL vs AllC vs AllC', {})
            
            for impl_type in ['simple', 'npdl']:
                if impl_type in original_results:
                    result = original_results[impl_type]
                    if result['ql_idx'] is not None:
                        ql_stats = result['final_stats'][result['ql_idx']]
                        exploit_rate = (1 - ql_stats['cooperation_rate']) * 100
                        print(f"  {impl_type.upper()} QL: {exploit_rate:.1f}% exploitation, score {ql_stats['total_score']:.0f}")
        
        if sorted_configs:
            improvement = sorted_configs[0][1]['exploit_rate'] - 63.3  # Approximate baseline
            print(f"\nIMPROVEMENT: +{improvement:.1f}% exploitation over typical baseline")
        
        print(f"\nTHEORETICAL MAXIMUM: 100% exploitation (always defect vs AllC)")


def main():
    """Run the comprehensive test suite."""
    suite = ComprehensiveTestSuite(verbose=True)
    results = suite.run_comprehensive_suite()
    
    # Optionally save results
    timestamp = int(time.time())
    filename = f"comprehensive_test_results_{timestamp}.json"
    
    try:
        import json
        # Simplified serialization (just key metrics)
        summary_results = {}
        for test_type, test_results in results.items():
            summary_results[test_type] = {}
            for config_name, result in test_results.items():
                if result['ql_idx'] is not None:
                    ql_stats = result['final_stats'][result['ql_idx']]
                    summary_results[test_type][config_name] = {
                        'cooperation_rate': ql_stats['cooperation_rate'],
                        'total_score': ql_stats['total_score'],
                        'exploitation_rate': (1 - ql_stats['cooperation_rate']) * 100,
                        'config': result.get('ql_config', {})
                    }
        
        with open(filename, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print(f"\nSummary results saved to {filename}")
    except Exception as e:
        print(f"\nCould not save results: {e}")
    
    return results


if __name__ == "__main__":
    main()