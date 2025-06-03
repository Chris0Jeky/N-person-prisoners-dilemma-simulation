# experiment_runner.py
import main_pairwise
import main_neighbourhood

# --- Experiment Parameters ---
EXPLORATION_RATES_TO_TEST = [0.0, 0.10]  # 0% and 10%

AGENT_COMPOSITIONS_TO_TEST = [
    {
        "name": "3_TFTs",
        "config_templates": [  # Using 'TFT' as a generic reciprocal strategy type
            {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "TFT", "strategy": "TFT"},
        ]
    },
    {
        "name": "2_TFTs_1_AllD",
        "config_templates": [
            {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "TFT", "strategy": "TFT"},
            {"id_prefix": "AllD", "strategy": "AllD"},
        ]
    }
]

NUM_ROUNDS_PAIRWISE_DEFAULT = 100
NUM_ROUNDS_N_PERSON_DEFAULT = 200


def generate_agent_configs(composition_template, exploration_rate):
    """Generates unique agent IDs and applies exploration rate."""
    configs = []
    counts = {}  # To make IDs unique like TFT_1, TFT_2
    for i, template in enumerate(composition_template["config_templates"]):
        strat_type = template["strategy"]
        counts[strat_type] = counts.get(strat_type, 0) + 1
        agent_id = f"{template['id_prefix']}_{counts[strat_type]}"

        configs.append({
            "id": agent_id,
            "strategy": template["strategy"],  # This is "TFT" or "AllD"
            "exploration_rate": exploration_rate
        })
    return configs


def main():
    print("Starting Experiment Runner...")

    for exp_rate in EXPLORATION_RATES_TO_TEST:
        print(f"\n\n{'=' * 30}")
        print(f"  RUNNING EXPERIMENTS WITH EXPLORATION RATE: {exp_rate * 100:.0f}%")
        print(f"{'=' * 30}")

        for composition in AGENT_COMPOSITIONS_TO_TEST:
            print(f"\n\n{'-' * 25}")
            print(f"  Agent Composition: {composition['name']}")
            print(f"{'-' * 25}")

            current_agent_base_configs = generate_agent_configs(composition, exp_rate)

            # --- Run Pairwise Model ---
            print(f"\n{'*' * 10} MODEL: PAIRWISE {'*' * 10}")
            print(f"Configuration: {composition['name']}, Exploration: {exp_rate * 100:.0f}%")
            # Pairwise model uses "TFT" and "AllD" as defined in agent_configs
            pairwise_configs_run = [dict(cfg) for cfg in current_agent_base_configs]  # Fresh copy
            main_pairwise.run_pairwise_experiment(
                agent_configurations=pairwise_configs_run,
                num_rounds_pairwise_param=NUM_ROUNDS_PAIRWISE_DEFAULT
            )

            # --- Run N-Person Model ---
            print(f"\n{'*' * 10} MODEL: N-PERSON (NEIGHBOURHOOD) {'*' * 10}")
            print(f"Configuration: {composition['name']}, Exploration: {exp_rate * 100:.0f}%")
            # N-Person model's run function will map "TFT" to "pTFT"
            n_person_configs_run = [dict(cfg) for cfg in current_agent_base_configs]  # Fresh copy
            main_neighbourhood.run_n_person_experiment(
                agent_configurations=n_person_configs_run,
                num_rounds_n_person_param=NUM_ROUNDS_N_PERSON_DEFAULT
            )

    print("\n\nAll experiments complete.")


if __name__ == "__main__":
    main()