# --- Part 3: Contents of experiment_runner.py (main logic) ---

import main_pairwise
import main_neighbourhood

EXPLORATION_RATES_TO_TEST = [0.0, 0.10] 

AGENT_COMPOSITIONS_TO_TEST = [
    {
        "name": "3_TFTs",
        "config_templates": [ 
            {"id_prefix": "TFT", "strategy": "TFT"}, {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "TFT", "strategy": "TFT"},
        ]
    },
    {
        "name": "2_TFTs_1_AllD",
        "config_templates": [
            {"id_prefix": "TFT", "strategy": "TFT"}, {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "AllD", "strategy": "AllD"},
        ]
    },
    {
        "name": "2_TFTs_3_AllD", # New 5-agent composition
        "config_templates": [
            {"id_prefix": "TFT", "strategy": "TFT"}, {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "AllD", "strategy": "AllD"}, {"id_prefix": "AllD", "strategy": "AllD"},
            {"id_prefix": "AllD", "strategy": "AllD"},
        ]
    }
]

NUM_ROUNDS_PAIRWISE_TOTAL_PER_PAIR = 100 
NUM_ROUNDS_N_PERSON_TOTAL = 200 

# New parameters for experiment runner
PAIRWISE_EPISODIC_MODES = [False, True]
PAIRWISE_NUM_EPISODES_IF_EPISODIC = 10 # results in 10 rounds per episode if total is 100

NPERSON_TFT_VARIANTS = ["pTFT", "pTFT-Threshold"]


def generate_agent_configs_runner(composition_template_entry, exploration_rate_param):
    configs = []
    id_counts = {}
    for template in composition_template_entry["config_templates"]:
        prefix = template["id_prefix"]
        id_counts[prefix] = id_counts.get(prefix, 0) + 1
        agent_id = f"{prefix}_{id_counts[prefix]}"
        configs.append({
            "id": agent_id, "strategy": template["strategy"], "exploration_rate": exploration_rate_param
        })
    return configs

def run_all_experiments():
    print("Starting Extended Experiment Runner...")

    for exp_rate in EXPLORATION_RATES_TO_TEST:
        print(f"\n\n{'='*40}")
        print(f"  RUNNING EXPERIMENTS: EXPLORATION RATE = {exp_rate*100:.0f}%")
        print(f"{'='*40}")

        for composition_entry in AGENT_COMPOSITIONS_TO_TEST:
            composition_name = composition_entry['name']
            num_agents = len(composition_entry["config_templates"])
            print(f"\n\n{'-'*35}")
            print(f"  Agent Composition: {composition_name} ({num_agents} agents)")
            print(f"{'-'*35}")

            current_agent_configs = generate_agent_configs_runner(composition_entry, exp_rate)

            # --- Pairwise Model Experiments ---
            print(f"\n{'#'*15} MODEL TYPE: PAIRWISE {'#'*15}")
            for is_episodic in PAIRWISE_EPISODIC_MODES:
                print(f"\n{'*'*10} PAIRWISE Sub-Experiment {'*'*10}")
                print(f"Config: {composition_name}, Exploration: {exp_rate*100:.0f}%, Episodic: {is_episodic}")
                
                pairwise_configs_to_run = [dict(cfg) for cfg in current_agent_configs]
                main_pairwise.run_pairwise_experiment(
                    agent_configurations=pairwise_configs_to_run, 
                    total_rounds_per_pair=NUM_ROUNDS_PAIRWISE_TOTAL_PER_PAIR,
                    episodic_mode=is_episodic,
                    num_episodes_if_episodic=PAIRWISE_NUM_EPISODES_IF_EPISODIC
                )

            # --- N-Person Model Experiments ---
            # Skip N-Person if only 1 agent, though our compositions are >=3
            if num_agents <= 1 and composition_name != "1_Agent_Test_Placeholder": # Avoid issues with N-1 divisor
                 print(f"\n{'#'*15} MODEL TYPE: N-PERSON (SKIPPED for <2 agents) {'#'*15}")
                 continue # Skip N-person for single agent scenarios if any were added

            print(f"\n{'#'*15} MODEL TYPE: N-PERSON {'#'*15}")
            for tft_variant in NPERSON_TFT_VARIANTS:
                print(f"\n{'*'*10} N-PERSON Sub-Experiment {'*'*10}")
                print(f"Config: {composition_name}, Exploration: {exp_rate*100:.0f}%, pTFT Variant: {tft_variant}")
                
                n_person_configs_to_run = [dict(cfg) for cfg in current_agent_configs]
                main_neighbourhood.run_n_person_experiment(
                    agent_configurations=n_person_configs_to_run, 
                    num_rounds_total=NUM_ROUNDS_N_PERSON_TOTAL,
                    tft_variant_for_tft_agents=tft_variant
                )
    
    print("\n\nAll extended experiments complete.")

if __name__ == "__main__":
    run_all_experiments()