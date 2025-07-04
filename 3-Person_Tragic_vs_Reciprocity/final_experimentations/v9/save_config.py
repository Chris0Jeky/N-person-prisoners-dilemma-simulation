#!/usr/bin/env python3
"""Utility to save configuration parameters with detailed formatting"""

import json
from datetime import datetime
from config import SIMULATION_CONFIG, VANILLA_PARAMS, ADAPTIVE_PARAMS, HYSTERETIC_PARAMS, LEGACY_PARAMS, LEGACY_3ROUND_PARAMS

def save_detailed_config(output_dir, scenario_descriptions=None):
    """
    Save configuration parameters in both human-readable and JSON formats
    
    Args:
        output_dir: Directory to save the config files
        scenario_descriptions: Optional dict describing what each scenario tests
    """
    import os
    
    # Save human-readable version
    txt_path = os.path.join(output_dir, "simulation_config.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "SIMULATION CONFIGURATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        # Simulation parameters
        f.write("SIMULATION PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write(f"Number of rounds per simulation: {SIMULATION_CONFIG['num_rounds']:,}\n")
        f.write(f"Number of runs to average: {SIMULATION_CONFIG['num_runs']}\n")
        f.write(f"Total simulations per scenario: {SIMULATION_CONFIG['num_rounds'] * SIMULATION_CONFIG['num_runs']:,}\n\n")
        
        # Vanilla Q-learner
        f.write("VANILLA Q-LEARNER (Fixed Parameters)\n")
        f.write("-"*40 + "\n")
        f.write(f"Learning Rate (α): {VANILLA_PARAMS['lr']}\n")
        f.write(f"Discount Factor (γ): {VANILLA_PARAMS['df']}\n")
        f.write(f"Exploration Rate (ε): {VANILLA_PARAMS['eps']}\n\n")
        
        # Adaptive Q-learner
        f.write("ADAPTIVE Q-LEARNER (Dynamic Parameters)\n")
        f.write("-"*40 + "\n")
        f.write(f"Initial Learning Rate: {ADAPTIVE_PARAMS['initial_lr']}\n")
        f.write(f"Initial Exploration Rate: {ADAPTIVE_PARAMS['initial_eps']}\n")
        f.write(f"Learning Rate Range: [{ADAPTIVE_PARAMS['min_lr']}, {ADAPTIVE_PARAMS['max_lr']}]\n")
        f.write(f"Exploration Rate Range: [{ADAPTIVE_PARAMS['min_eps']}, {ADAPTIVE_PARAMS['max_eps']}]\n")
        f.write(f"Adaptation Factor: {ADAPTIVE_PARAMS['adaptation_factor']}\n")
        f.write(f"Reward Window Size: {ADAPTIVE_PARAMS['reward_window_size']}\n")
        f.write(f"Discount Factor (γ): {ADAPTIVE_PARAMS['df']}\n\n")
        
        # Hysteretic Q-learner
        f.write("HYSTERETIC Q-LEARNER (Asymmetric Learning)\n")
        f.write("-"*40 + "\n")
        f.write(f"Learning Rate (α) for good news: {HYSTERETIC_PARAMS['lr']}\n")
        f.write(f"Learning Rate (β) for bad news: {HYSTERETIC_PARAMS['beta']}\n")
        f.write(f"Discount Factor (γ): {HYSTERETIC_PARAMS['df']}\n")
        f.write(f"Exploration Rate (ε): {HYSTERETIC_PARAMS['eps']}\n\n")
        
        # Legacy Q-learner
        f.write("LEGACY Q-LEARNER (Sophisticated State Representation)\n")
        f.write("-"*40 + "\n")
        f.write(f"Learning Rate (α): {LEGACY_PARAMS['lr']}\n")
        f.write(f"Discount Factor (γ): {LEGACY_PARAMS['df']}\n")
        f.write(f"Initial Exploration Rate (ε): {LEGACY_PARAMS['eps']}\n")
        f.write(f"Epsilon Decay Rate: {LEGACY_PARAMS['epsilon_decay']}\n")
        f.write(f"Minimum Epsilon: {LEGACY_PARAMS['epsilon_min']}\n")
        f.write(f"Initial Q-values: {LEGACY_PARAMS['optimistic_init']}\n")
        f.write("Features: 2-round history, cooperation trends\n\n")
        
        # Legacy 3-Round Q-learner
        f.write("LEGACY 3-ROUND Q-LEARNER (Extended History)\n")
        f.write("-"*40 + "\n")
        f.write(f"Learning Rate (α): {LEGACY_3ROUND_PARAMS['lr']}\n")
        f.write(f"Discount Factor (γ): {LEGACY_3ROUND_PARAMS['df']}\n")
        f.write(f"Initial Exploration Rate (ε): {LEGACY_3ROUND_PARAMS['eps']}\n")
        f.write(f"Epsilon Decay Rate: {LEGACY_3ROUND_PARAMS['epsilon_decay']}\n")
        f.write(f"Minimum Epsilon: {LEGACY_3ROUND_PARAMS['epsilon_min']}\n")
        f.write(f"Initial Q-values: {LEGACY_3ROUND_PARAMS['optimistic_init']}\n")
        f.write(f"History Length: {LEGACY_3ROUND_PARAMS['history_length']} rounds\n")
        f.write("Features: 3-round history, complex trend analysis, pessimistic initialization\n\n")
        
        # Scenario descriptions if provided
        if scenario_descriptions:
            f.write("SCENARIO DESCRIPTIONS\n")
            f.write("-"*40 + "\n")
            for scenario, desc in scenario_descriptions.items():
                f.write(f"{scenario}: {desc}\n")
            f.write("\n")
        
        # Game parameters
        f.write("GAME PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write("Prisoner's Dilemma Payoffs:\n")
        f.write("  T (Temptation): 5\n")
        f.write("  R (Reward): 3\n")
        f.write("  P (Punishment): 1\n")
        f.write("  S (Sucker): 0\n\n")
        
        f.write("Agent Strategies:\n")
        f.write("  AllC: Always Cooperate\n")
        f.write("  AllD: Always Defect\n")
        f.write("  Random: 50/50 random choice\n")
        f.write("  TFT: Tit-for-Tat\n")
        f.write("    - Pairwise: Copies opponent's last move\n")
        f.write("    - N-Person: Cooperates with probability = cooperation ratio\n")
        f.write("  TFT-E: TFT with 10% error rate\n\n")
        
        f.write("="*80 + "\n")
    
    # Save JSON version for programmatic access
    json_path = os.path.join(output_dir, "simulation_config.json")
    config_dict = {
        "timestamp": datetime.now().isoformat(),
        "output_directory": output_dir,
        "simulation": SIMULATION_CONFIG,
        "vanilla_params": VANILLA_PARAMS,
        "adaptive_params": ADAPTIVE_PARAMS,
        "hysteretic_params": HYSTERETIC_PARAMS,
        "legacy_params": LEGACY_PARAMS,
        "legacy_3round_params": LEGACY_3ROUND_PARAMS,
        "game_params": {
            "payoffs": {"T": 5, "R": 3, "P": 1, "S": 0},
            "strategies": {
                "AllC": "Always Cooperate",
                "AllD": "Always Defect",
                "Random": "50/50 random",
                "TFT": "Tit-for-Tat",
                "TFT-E": "TFT with 10% error"
            }
        }
    }
    
    if scenario_descriptions:
        config_dict["scenarios"] = scenario_descriptions
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Configuration saved to:")
    print(f"  - {txt_path} (human-readable)")
    print(f"  - {json_path} (JSON format)")

if __name__ == "__main__":
    # Test the function
    save_detailed_config(".", {
        "2QL_vs_1AllC": "Two Q-learners vs one Always Cooperate",
        "2QL_vs_1AllD": "Two Q-learners vs one Always Defect",
        "1QL_vs_2TFT": "One Q-learner vs two Tit-for-Tat"
    })