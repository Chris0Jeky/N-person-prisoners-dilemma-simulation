import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def visualize_sweep_results(csv_file, output_dir):
    """Loads sweep results and generates plots."""

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}")
        return

    if df.empty:
        print(f"Warning: CSV file {csv_file} is empty.")
        return

    print(f"Visualizing results from: {csv_file}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Identify parameters and metrics ---
    metric_cols = [
        col for col in df.columns if col.startswith("avg_") or col.startswith("std_")
    ]
    param_cols = [col for col in df.columns if col not in metric_cols]

    # Target metrics for main plots
    target_coop = "avg_final_coop_rate_overall"
    target_score = "avg_avg_final_score_target"  # Adjusted based on CSV headers

    if target_coop not in df.columns:
        print(
            f"Warning: Primary metric '{target_coop}' not found in {csv_file}. Skipping some plots."
        )
        target_coop = None  # Disable plots relying on it

    # --- Generate Plots ---

    # 1. Pair Plot (Good overview, can be slow/dense for many params)
    try:
        if target_coop and len(param_cols) > 1:
            print("  Generating pair plot...")
            pair_plot_vars = param_cols + [target_coop]
            # Limit number of parameters for readability if too many
            if len(pair_plot_vars) > 6:
                print(f"    (Limiting pair plot to first {6 - 1} params + coop rate)")
                pair_plot_vars = param_cols[: (6 - 1)] + [target_coop]

            # Select a param with few unique values for hue if possible
            hue_param = None
            for p in param_cols:
                if df[p].nunique() < 6:  # Example threshold for hue categories
                    hue_param = p
                    break

            sns_plot = sns.pairplot(
                df[pair_plot_vars], height=2.5, hue=hue_param, diag_kind="kde"
            )
            plt.suptitle(
                f"Parameter Relationships vs Coop Rate ({os.path.basename(csv_file)})",
                y=1.02,
            )
            plot_path = os.path.join(
                output_dir,
                f"{os.path.basename(csv_file).replace('.csv', '_pairplot.png')}",
            )
            sns_plot.savefig(plot_path, dpi=150)
            plt.close()

    except Exception as e:
        print(f"  Error generating pair plot: {e}")

    # 2. Individual Parameter vs. Metrics Plots
    print("  Generating individual parameter plots...")
    num_params = len(param_cols)
    # Determine grid size dynamically
    cols = 3
    rows = math.ceil(num_params / cols)

    fig_coop, axes_coop = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
    )
    fig_score, axes_score = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
    )
    axes_coop_flat = axes_coop.flatten()
    axes_score_flat = axes_score.flatten()

    for i, param in enumerate(param_cols):
        # Cooperation Rate vs Parameter
        if target_coop:
            ax_c = axes_coop_flat[i]
            # Use boxplot for clearer distribution, especially if other params vary
            sns.boxplot(x=param, y=target_coop, data=df, ax=ax_c, palette="viridis")
            sns.stripplot(
                x=param, y=target_coop, data=df, ax=ax_c, color=".3", alpha=0.5, size=3
            )  # Show individual points
            ax_c.set_title(f"{target_coop} vs {param}")
            ax_c.tick_params(axis="x", rotation=45)
            ax_c.grid(True, axis="y", linestyle="--", alpha=0.6)

        # Target Score vs Parameter
        if target_score in df.columns:
            ax_s = axes_score_flat[i]
            sns.boxplot(x=param, y=target_score, data=df, ax=ax_s, palette="viridis")
            sns.stripplot(
                x=param, y=target_score, data=df, ax=ax_s, color=".3", alpha=0.5, size=3
            )
            ax_s.set_title(f"{target_score} vs {param}")
            ax_s.tick_params(axis="x", rotation=45)
            ax_s.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        fig_coop.delaxes(axes_coop_flat[j])
        fig_score.delaxes(axes_score_flat[j])

    if target_coop:
        fig_coop.suptitle(
            f"Cooperation Rate vs Individual Parameters ({os.path.basename(csv_file)})",
            fontsize=16,
            y=1.03,
        )
        fig_coop.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout
        plot_path_coop = os.path.join(
            output_dir,
            f"{os.path.basename(csv_file).replace('.csv', '_param_vs_coop.png')}",
        )
        fig_coop.savefig(plot_path_coop, dpi=150)
        plt.close(fig_coop)

    if target_score in df.columns:
        fig_score.suptitle(
            f"Target Score vs Individual Parameters ({os.path.basename(csv_file)})",
            fontsize=16,
            y=1.03,
        )
        fig_score.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust layout
        plot_path_score = os.path.join(
            output_dir,
            f"{os.path.basename(csv_file).replace('.csv', '_param_vs_score.png')}",
        )
        fig_score.savefig(plot_path_score, dpi=150)
        plt.close(fig_score)

    print(f"  Visualizations saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parameter sweep results.")
    parser.add_argument(
        "csv_file", type=str, help="Path to the sweep results CSV file."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to same directory as CSV.",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.csv_file)

    visualize_sweep_results(args.csv_file, args.out_dir)
