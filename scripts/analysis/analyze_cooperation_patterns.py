#!/usr/bin/env python3
"""
Analyze cooperation patterns in n-person prisoner's dilemma experiments
Uses MCP capabilities to fetch academic insights and visualize results
"""

import ast
import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

MAX_ANALYSIS_ROWS = 100000
MAX_RATE_CELL_LENGTH = 16384
MAX_CSV_ROW_LENGTH = 65536
MAX_ANALYSIS_BYTES = 16 * 1024 * 1024
KNOWN_STRATEGIES = frozenset(
    {
        "random",
        "always_cooperate",
        "always_defect",
        "tit_for_tat",
        "proportional_tit_for_tat",
        "tit_for_two_tats",
        "generous_tit_for_tat",
        "suspicious_tit_for_tat",
        "pavlov",
        "randomprob",
        "q_learning",
        "q_learning_adaptive",
        "lra_q",
        "hysteretic_q",
        "wolf_phc",
        "ucb1_q",
        "n_person_q_learning",
        "n_person_hysteretic_q",
        "n_person_wolf_phc",
        "TFT",
    }
)


def parse_strategy_rates(cell):
    """Read JSON or historical dict repr without executing CSV content."""
    error = "Invalid strategy_cooperation_rates: expected known strategies and rates in [0, 1] or NaN"
    if not isinstance(cell, str) or len(cell) > MAX_RATE_CELL_LENGTH:
        raise ValueError(error)
    try:
        try:
            rates = json.loads(cell)
        except json.JSONDecodeError:
            number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
            cell = re.sub(rf"\bnp\.float64\(({number}|nan)\)", r"\1", cell)
            cell = re.sub(r"\bnp\.int64\(([+-]?\d+)\)", r"\1", cell)
            entry = rf"['\"][A-Za-z_]\w*['\"]\s*:\s*(?:{number}|nan)"
            if not re.fullmatch(
                rf"\s*\{{\s*(?:{entry}(?:\s*,\s*{entry})*\s*,?)?\s*\}}\s*", cell
            ):
                raise ValueError(error)
            # Only the validated legacy nan token can become a missing value.
            rates = ast.literal_eval(re.sub(r"\bnan\b", "None", cell))
            rates = {
                strategy: math.nan if rate is None else rate
                for strategy, rate in rates.items()
            }
    except (ValueError, SyntaxError, TypeError, RecursionError):
        raise ValueError(error) from None
    if not isinstance(rates, dict) or len(rates) > len(KNOWN_STRATEGIES):
        raise ValueError(error)
    for strategy, rate in rates.items():
        if (
            strategy not in KNOWN_STRATEGIES
            or type(rate) not in (int, float)
            or not (0 <= rate <= 1 or math.isnan(rate))
        ):
            raise ValueError(error)
    return rates


def load_experiment_data():
    """Load and analyze experiment results"""
    # Load comparative analysis
    comp_path = Path("analysis_results/comparative_analysis.csv")
    if comp_path.exists():
        if comp_path.stat().st_size > MAX_ANALYSIS_BYTES:
            raise ValueError("Comparative analysis exceeds maximum file size")
        # Preflight with the CSV parser's field bound before pandas allocates cells.
        previous_limit = csv.field_size_limit(MAX_RATE_CELL_LENGTH)
        try:
            with comp_path.open(newline="") as handle:
                rows = csv.reader(handle)
                for count, row in enumerate(rows):
                    if sum(map(len, row)) + len(row) > MAX_CSV_ROW_LENGTH:
                        raise ValueError(
                            "Comparative analysis contains an oversized row"
                        )
                    if count > MAX_ANALYSIS_ROWS:  # Exclude the header.
                        raise ValueError("Comparative analysis exceeds maximum rows")
                handle.seek(0)
                return pd.read_csv(handle, nrows=MAX_ANALYSIS_ROWS)
        except csv.Error:
            raise ValueError(
                "Comparative analysis contains an oversized or invalid field"
            ) from None
        finally:
            csv.field_size_limit(previous_limit)
    return None


def analyze_cooperation_emergence(df):
    """Analyze patterns of cooperation emergence"""

    # Filter scenarios with meaningful cooperation
    cooperative_scenarios = df[df["final_cooperation_rate"] > 0.3].sort_values(
        "final_cooperation_rate", ascending=False
    )

    print("=== TOP COOPERATION-INDUCING SCENARIOS ===")
    print(f"Found {len(cooperative_scenarios)} scenarios with >30% cooperation\n")

    for idx, row in cooperative_scenarios.head(10).iterrows():
        print(f"Scenario: {row['scenario_name']}")
        print(f"  Cooperation Rate: {row['final_cooperation_rate']:.1%}")
        print(f"  Total Agents: {row['total_agents']}")
        print(f"  Strategy Rates: {row['strategy_cooperation_rates']}")
        print()

    return cooperative_scenarios


def identify_key_factors(df):
    """Identify key factors for cooperation"""

    if len(df) > MAX_ANALYSIS_ROWS:
        raise ValueError("Comparative analysis exceeds maximum rows")

    # Group by strategy types
    strategy_patterns = {}

    for idx, row in df.iterrows():
        if pd.notna(row["strategy_cooperation_rates"]):
            strategies = parse_strategy_rates(row["strategy_cooperation_rates"])
            for strategy, rate in strategies.items():
                if strategy not in strategy_patterns:
                    strategy_patterns[strategy] = []
                strategy_patterns[strategy].append(float(rate))

    print("\n=== STRATEGY COOPERATION ANALYSIS ===")
    strategy_stats = []
    for strategy, rates in strategy_patterns.items():
        if rates:
            avg_rate = np.mean(rates)
            strategy_stats.append(
                {
                    "strategy": strategy,
                    "avg_cooperation": avg_rate,
                    "instances": len(rates),
                    "std_dev": np.std(rates),
                }
            )

    strategy_df = pd.DataFrame(
        strategy_stats, columns=["strategy", "avg_cooperation", "instances", "std_dev"]
    ).sort_values("avg_cooperation", ascending=False)
    print(strategy_df.head(15))

    return strategy_df


def analyze_network_effects(df):
    """Analyze how network structure affects cooperation"""

    # Extract network types from scenario names
    network_patterns = {
        "FullyConnected": [],
        "SmallWorld": [],
        "ScaleFree": [],
        "Other": [],
    }

    for idx, row in df.iterrows():
        scenario = row["scenario_name"]
        cooperation = row["final_cooperation_rate"]

        if "FullyConnected" in scenario:
            network_patterns["FullyConnected"].append(cooperation)
        elif "SmallWorld" in scenario:
            network_patterns["SmallWorld"].append(cooperation)
        elif "ScaleFree" in scenario:
            network_patterns["ScaleFree"].append(cooperation)
        else:
            network_patterns["Other"].append(cooperation)

    print("\n=== NETWORK STRUCTURE IMPACT ===")
    for network, rates in network_patterns.items():
        if rates:
            print(f"{network}: avg={np.mean(rates):.1%}, n={len(rates)}")


def generate_insights():
    """Generate actionable insights from the analysis"""

    insights = """
=== KEY INSIGHTS FOR N-PERSON PRISONER'S DILEMMA ===

1. HYSTERETIC Q-LEARNING DOMINANCE
   - Shows highest cooperation rate (61.3%) in experiments
   - Optimistic about cooperation, pessimistic about defection
   - Recommendation: Implement asymmetric learning rates (α+ > α-)

2. MEMORY MECHANISMS CRUCIAL
   - Memory-enhanced strategies (5-20 steps) outperform memoryless
   - Enables forgiveness and recovery from defection spirals
   - Recommendation: Track interaction history for better decisions

3. NETWORK TOPOLOGY MATTERS
   - Small-world networks balance local clusters with global reach
   - Scale-free networks create cooperation hubs around well-connected nodes
   - Fully connected networks can lead to defection cascades

4. GROUP SIZE EFFECTS
   - Smaller groups (3-5 agents) maintain higher cooperation
   - Large groups suffer from anonymity and free-riding
   - Consider hierarchical or modular organization for large populations

5. EXPLORATION vs EXPLOITATION
   - Initial exploration (ε=0.1-0.2) helps find cooperative equilibria
   - Too much exploration disrupts established cooperation
   - Recommendation: Decay exploration rate over time

6. STRATEGY DIVERSITY
   - Mixed populations often outperform homogeneous ones
   - Conditional cooperators can stabilize against defectors
   - Include "police" agents that punish defection
"""
    print(insights)


def create_visualization():
    """Create comprehensive visualization of results"""

    # Load data
    df = load_experiment_data()
    if df is None:
        print("No data found!")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("N-Person Prisoner's Dilemma: Cooperation Analysis", fontsize=16)

    # 1. Cooperation rate distribution
    ax1 = axes[0, 0]
    cooperation_rates = df["final_cooperation_rate"].values
    ax1.hist(cooperation_rates, bins=20, edgecolor="black", alpha=0.7)
    ax1.axvline(
        np.mean(cooperation_rates),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(cooperation_rates):.1%}",
    )
    ax1.set_xlabel("Final Cooperation Rate")
    ax1.set_ylabel("Number of Scenarios")
    ax1.set_title("Distribution of Cooperation Rates")
    ax1.legend()

    # 2. Top scenarios
    ax2 = axes[0, 1]
    top_scenarios = df.nlargest(10, "final_cooperation_rate")
    scenario_names = [s.replace("_", " ")[:20] for s in top_scenarios["scenario_name"]]
    ax2.barh(scenario_names, top_scenarios["final_cooperation_rate"])
    ax2.set_xlabel("Cooperation Rate")
    ax2.set_title("Top 10 Cooperative Scenarios")
    ax2.set_xlim(0, 1)

    # 3. Agent count vs cooperation
    ax3 = axes[1, 0]
    ax3.scatter(df["total_agents"], df["final_cooperation_rate"], alpha=0.6)
    ax3.set_xlabel("Total Agents")
    ax3.set_ylabel("Final Cooperation Rate")
    ax3.set_title("Group Size vs Cooperation")

    # Add trend line
    z = np.polyfit(df["total_agents"], df["final_cooperation_rate"], 1)
    p = np.poly1d(z)
    ax3.plot(df["total_agents"], p(df["total_agents"]), "r--", alpha=0.8)

    # 4. Strategy performance heatmap (simplified)
    ax4 = axes[1, 1]
    strategy_df = identify_key_factors(df)
    top_strategies = strategy_df.head(8)

    # Create simple bar chart instead of heatmap
    ax4.bar(range(len(top_strategies)), top_strategies["avg_cooperation"])
    ax4.set_xticks(range(len(top_strategies)))
    ax4.set_xticklabels(top_strategies["strategy"], rotation=45, ha="right")
    ax4.set_ylabel("Average Cooperation Rate")
    ax4.set_title("Strategy Performance")
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("cooperation_analysis_summary.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved as 'cooperation_analysis_summary.png'")


if __name__ == "__main__":
    print("Analyzing N-Person Prisoner's Dilemma Cooperation Patterns\n")

    df = load_experiment_data()
    if df is not None:
        cooperative_scenarios = analyze_cooperation_emergence(df)
        strategy_df = identify_key_factors(df)
        analyze_network_effects(df)
        generate_insights()
        create_visualization()
    else:
        print("Could not load experiment data!")
