"""
Plotting script — generates every figure needed for the report from
the CSV outputs of train.py and test_and_evaluate.py.

Figures produced:
  1) Training curves (reward, cost, comfort, alpha)
  2) Validation curves with error bands
  3) Policy comparison bar charts
  4) Cost / comfort distributions
  5) Cost-vs-comfort scatter (Pareto trade-off)
  6) Per-season box plots
  7) 24-hour trajectory comparisons

Box plots suppress fliers for readability — see the histograms if you
want the full spread including extreme episodes.

Usage:
    python plot_results.py
    # or specify files explicitly:
    python plot_results.py --train training_log_...csv --eval eval_results_...csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Matplotlib defaults
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.figsize": (8, 4),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUTPUT_DIR = "plots"
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
POLICY_COLOURS = {
    "naive": "tab:gray",
    "sac": "tab:blue",
    "rule": "tab:orange",
    "random": "tab:red",
}
POLICY_LABELS = {
    "naive": "Naive thermostat",
    "sac": "SAC (AI)",
    "rule": "Rule-based",
    "random": "Random",
}


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_latest(pattern):
    """Return most recently modified file matching a glob pattern."""
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None


def rolling_mean(series, window=50):
    return series.rolling(window=window, min_periods=1).mean()


# =====================================================================
# 1. Training curves
# =====================================================================
def plot_training_curves(csv_path):
    """Reward / cost / comfort / alpha over episodes, with rolling mean overlay."""
    df = pd.read_csv(csv_path)
    window = 50

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(df["episode"], df["total_reward"], alpha=0.15, color="tab:blue")
    ax.plot(df["episode"], rolling_mean(df["total_reward"], window),
            color="tab:blue", linewidth=1.5, label=f"{window}-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("(a) Episode reward during training")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["episode"], df["total_energy_cost_p"], alpha=0.15, color="tab:green")
    ax.plot(df["episode"], rolling_mean(df["total_energy_cost_p"], window),
            color="tab:green", linewidth=1.5, label=f"{window}-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Daily cost (pence)")
    ax.set_title("(b) Daily energy cost during training")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["episode"], df["avg_comfort_penalty"], alpha=0.15, color="tab:red")
    ax.plot(df["episode"], rolling_mean(df["avg_comfort_penalty"], window),
            color="tab:red", linewidth=1.5, label=f"{window}-ep rolling mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg comfort penalty")
    ax.set_title("(c) Comfort penalty during training")
    ax.legend()

    if "alpha" in df.columns:
        ax = axes[1, 1]
        ax.plot(df["episode"], df["alpha"], color="tab:purple", linewidth=1.0)
        ax.set_xlabel("Episode")
        ax.set_ylabel("α (entropy coefficient)")
        ax.set_title("(d) Entropy coefficient α during training")
    else:
        axes[1, 1].text(0.5, 0.5, "No alpha data\n(fixed α used)",
                         ha="center", va="center", transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("(d) Entropy coefficient")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 2. Validation curves
# =====================================================================
def plot_validation_curves(csv_path):
    """Validation reward and cost over training, plus per-season cost trends."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(df["episode"], df["mean_reward"], color="tab:blue", marker=".", markersize=3)
    ax.fill_between(
        df["episode"],
        df["mean_reward"] - df["std_reward"],
        df["mean_reward"] + df["std_reward"],
        alpha=0.2, color="tab:blue",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward")
    ax.set_title("(a) Validation reward")

    ax = axes[1]
    ax.plot(df["episode"], df["mean_cost"], color="tab:green", marker=".", markersize=3)
    ax.fill_between(
        df["episode"],
        df["mean_cost"] - df["std_cost"],
        df["mean_cost"] + df["std_cost"],
        alpha=0.2, color="tab:green",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean cost (pence)")
    ax.set_title("(b) Validation cost")

    ax = axes[2]
    season_cols = ["season_0_cost", "season_1_cost", "season_2_cost", "season_3_cost"]
    if all(c in df.columns for c in season_cols):
        for s_id, col in enumerate(season_cols):
            ax.plot(df["episode"], df[col], label=SEASON_NAMES[s_id], marker=".", markersize=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean cost (pence)")
        ax.set_title("(c) Cost by season over training")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No per-season data", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "validation_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 3. Comparison bar charts
# =====================================================================
def plot_comparison_bars(csv_path):
    """Bar charts per policy for cost, comfort penalty, and violations."""
    df = pd.read_csv(csv_path)

    policies = ["naive", "sac", "rule", "random"]
    labels = [POLICY_LABELS[p] for p in policies]
    colours = [POLICY_COLOURS[p] for p in policies]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    means = [df[df["policy"] == p]["total_energy_cost_p"].mean() for p in policies]
    sems = [df[df["policy"] == p]["total_energy_cost_p"].sem() * 1.96 for p in policies]
    max_sem = max((s for s in sems if not np.isnan(s)), default=1.0)
    bars = ax.bar(labels, means, yerr=sems, color=colours, capsize=4, alpha=0.85)
    ax.set_ylabel("Daily cost (pence)")
    ax.set_title("(a) Average daily energy cost")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_sem * 0.3,
                f"{m:.0f}p", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    means = [df[df["policy"] == p]["avg_comfort_penalty"].mean() for p in policies]
    sems = [df[df["policy"] == p]["avg_comfort_penalty"].sem() * 1.96 for p in policies]
    bars = ax.bar(labels, means, yerr=sems, color=colours, capsize=4, alpha=0.85)
    ax.set_ylabel("Avg comfort penalty")
    ax.set_title("(b) Average comfort penalty")
    max_sem = max((s for s in sems if not np.isnan(s)), default=1.0)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_sem * 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    viols_policies = ["naive", "sac", "rule"]
    viols_labels = [POLICY_LABELS[p] for p in viols_policies]
    viols_colours = [POLICY_COLOURS[p] for p in viols_policies]
    means = [df[df["policy"] == p]["comfort_violations"].mean() for p in viols_policies]
    sems = [df[df["policy"] == p]["comfort_violations"].sem() * 1.96 for p in viols_policies]
    max_sem = max((s for s in sems if not np.isnan(s)), default=1.0)
    bars = ax.bar(viols_labels, means, yerr=sems, color=viols_colours, capsize=4, alpha=0.85)
    ax.set_ylabel("Avg violations per day")
    ax.set_title("(c) Comfort violations (>1°C when occupied)")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_sem * 0.3,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_bars.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 4. Distributions
# =====================================================================
def plot_distributions(csv_path):
    """Overlapping histograms of daily cost and comfort penalty per policy."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for policy in ["naive", "sac", "rule", "random"]:
        subset = df[df["policy"] == policy]["total_energy_cost_p"]
        if len(subset) > 0:
            ax.hist(subset, bins=25, alpha=0.4, label=POLICY_LABELS[policy],
                    color=POLICY_COLOURS[policy])
    ax.set_xlabel("Daily cost (pence)")
    ax.set_ylabel("Number of episodes")
    ax.set_title("(a) Distribution of daily energy cost")
    ax.legend()

    ax = axes[1]
    for policy in ["naive", "sac", "rule"]:
        subset = df[df["policy"] == policy]["avg_comfort_penalty"]
        if len(subset) > 0:
            ax.hist(subset, bins=25, alpha=0.4, label=POLICY_LABELS[policy],
                    color=POLICY_COLOURS[policy])
    ax.set_xlabel("Avg comfort penalty")
    ax.set_ylabel("Number of episodes")
    ax.set_title("(b) Distribution of comfort penalty")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 5. Cost-vs-comfort scatter
# =====================================================================
def plot_cost_vs_comfort(csv_path):
    """Scatter of cost vs comfort — ideal corner is bottom-left."""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 5))

    for policy in ["naive", "sac", "rule"]:
        subset = df[df["policy"] == policy]
        ax.scatter(
            subset["total_energy_cost_p"],
            subset["avg_comfort_penalty"],
            alpha=0.3, s=15, label=POLICY_LABELS[policy],
            color=POLICY_COLOURS[policy],
        )
        # Diamond marker for the policy mean
        ax.scatter(
            subset["total_energy_cost_p"].mean(),
            subset["avg_comfort_penalty"].mean(),
            s=120, color=POLICY_COLOURS[policy],
            edgecolors="black", linewidths=1.5, zorder=5,
            marker="D",
        )

    ax.set_xlabel("Daily cost (pence)")
    ax.set_ylabel("Avg comfort penalty")
    ax.set_title("Cost–comfort trade-off (diamonds = policy means)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cost_vs_comfort.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 6. Per-season box plots
# =====================================================================
def plot_season_boxplots(csv_path):
    """Box plots of cost per policy, split by season."""
    df = pd.read_csv(csv_path)

    if "season" not in df.columns:
        print("No season column in eval CSV — skipping season box plots.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for i, (s_id, s_name) in enumerate(SEASON_NAMES.items()):
        ax = axes[i]
        season_data = []
        tick_labels = []

        for policy in ["naive", "sac", "rule"]:
            subset = df[(df["policy"] == policy) & (df["season"] == s_id)]
            season_data.append(subset["total_energy_cost_p"].values)
            tick_labels.append(POLICY_LABELS[policy])

        bp = ax.boxplot(season_data, labels=tick_labels, patch_artist=True,
                        widths=0.6, showfliers=False)

        colours = [POLICY_COLOURS["naive"], POLICY_COLOURS["sac"], POLICY_COLOURS["rule"]]
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)

        ax.set_title(f"{s_name} (n={len(season_data[1])})")
        ax.tick_params(axis="x", rotation=25)

        if i == 0:
            ax.set_ylabel("Daily cost (pence)")

    plt.suptitle("Daily energy cost by season and policy", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "season_boxplots.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# =====================================================================
# 7. 24-hour trajectory comparison
# =====================================================================
def plot_trajectories(traj_df, seed=None):
    """
    Side-by-side 24-hour comparison for one scenario.
    Shows temperature, SOC, grid power, and HVAC actions across policies —
    this is probably the most impactful figure for the report since it
    shows *how* SAC saves money, not just that it does.
    """
    df = traj_df

    available_seeds = df["seed"].unique()
    if seed is None:
        seed = available_seeds[0]
    elif seed not in available_seeds:
        print(f"Seed {seed} not in trajectory data. Available: {available_seeds}")
        return

    seed_data = df[df["seed"] == seed]
    season_id = seed_data["season"].iloc[0]
    season_name = SEASON_NAMES.get(season_id, "Unknown")

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for policy in ["naive", "sac", "rule"]:
        pdata = seed_data[seed_data["policy"] == policy]
        if len(pdata) == 0:
            continue

        hours = pdata["hour"].values
        colour = POLICY_COLOURS[policy]
        label = POLICY_LABELS[policy]

        axes[0].plot(hours, pdata["T_in"].values, color=colour, label=label, linewidth=1.2)
        axes[1].plot(hours, pdata["SOC"].values, color=colour, label=label, linewidth=1.2)
        axes[2].plot(hours, pdata["p_grid_kw"].values, color=colour, label=label, linewidth=1.2)
        axes[3].plot(hours, pdata["a_hvac"].values, color=colour, label=label,
                    linewidth=1.0, alpha=0.8)

    # (a) Temperature
    ax = axes[0]
    naive_data = seed_data[seed_data["policy"] == "naive"]
    if len(naive_data) > 0:
        ax.plot(naive_data["hour"].values, naive_data["T_out"].values,
                color="black", linestyle="--", alpha=0.4, label="Outdoor temp")
    ax.axhline(21.0, color="green", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.axhspan(20.0, 22.0, color="green", alpha=0.07)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"24-hour trajectory comparison — Seed {seed} ({season_name})")
    ax.legend(loc="upper right", fontsize=8)

    # Shade when home is occupied
    if len(naive_data) > 0:
        occ = naive_data["occupancy"].values
        hours_arr = naive_data["hour"].values
        for j in range(len(occ) - 1):
            if occ[j] > 0.5:
                ax.axvspan(hours_arr[j], hours_arr[j + 1],
                           color="yellow", alpha=0.06)

    # (b) SOC
    ax = axes[1]
    ax.set_ylabel("Battery SOC")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.2, color="red", linestyle=":", alpha=0.4, linewidth=0.7)
    ax.axhline(0.8, color="red", linestyle=":", alpha=0.4, linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8)

    # (c) Grid power with price overlay
    ax = axes[2]
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Grid power (kW)")
    ax.legend(loc="upper right", fontsize=8)

    if len(naive_data) > 0:
        ax2 = ax.twinx()
        ax2.plot(naive_data["hour"].values, naive_data["price"].values,
                 color="tab:orange", linestyle=":", alpha=0.5, linewidth=0.8)
        ax2.set_ylabel("Price (p/kWh)", color="tab:orange", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=8)

    # (d) HVAC action
    ax = axes[3]
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("HVAC action")
    ax.set_xlabel("Hour of day")
    ax.set_ylim(-1.1, 1.1)
    ax.text(0.01, 0.95, "Heating ↑", transform=ax.transAxes, fontsize=7,
            va="top", color="tab:red")
    ax.text(0.01, 0.05, "Cooling ↓", transform=ax.transAxes, fontsize=7,
            va="bottom", color="tab:cyan")
    ax.legend(loc="upper right", fontsize=8)

    axes[3].set_xlim(0, 24)
    axes[3].xaxis.set_major_locator(mticker.MultipleLocator(2))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"trajectory_seed{seed}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate all SmartFlex report plots")
    parser.add_argument("--train", default=None, help="Training log CSV")
    parser.add_argument("--val", default=None, help="Validation log CSV")
    parser.add_argument("--eval", default=None, help="Evaluation results CSV")
    parser.add_argument("--traj", default=None, help="Trajectory data CSV")
    args = parser.parse_args()

    ensure_dir()

    # Auto-pick the most recent CSVs if not specified
    train_csv = args.train or find_latest("training_log_*.csv")
    val_csv = args.val or find_latest("validation_log_*.csv")
    eval_csv = args.eval or find_latest("eval_results_*.csv")
    traj_csv = args.traj or find_latest("trajectories_*.csv")

    if train_csv and os.path.exists(train_csv):
        print(f"\n[1/7] Training curves from: {train_csv}")
        plot_training_curves(train_csv)
    else:
        print("[1/7] No training log found — skipping training curves.")

    if val_csv and os.path.exists(val_csv):
        print(f"[2/7] Validation curves from: {val_csv}")
        plot_validation_curves(val_csv)
    else:
        print("[2/7] No validation log found — skipping validation curves.")

    if eval_csv and os.path.exists(eval_csv):
        print(f"[3/7] Comparison bars from: {eval_csv}")
        plot_comparison_bars(eval_csv)

        print(f"[4/7] Distributions from: {eval_csv}")
        plot_distributions(eval_csv)

        print(f"[5/7] Cost-vs-comfort scatter from: {eval_csv}")
        plot_cost_vs_comfort(eval_csv)

        print(f"[6/7] Season box plots from: {eval_csv}")
        plot_season_boxplots(eval_csv)
    else:
        print("[3-6/7] No evaluation CSV found — skipping evaluation plots.")

    if traj_csv and os.path.exists(traj_csv):
        traj_df = pd.read_csv(traj_csv)
        seeds = traj_df["seed"].unique()
        print(f"[7/7] Trajectory plots from: {traj_csv} (seeds: {seeds})")
        for seed in seeds:
            plot_trajectories(traj_df, seed=seed)
    else:
        print("[7/7] No trajectory CSV found — skipping trajectory plots.")

    print(f"\nAll plots saved to '{OUTPUT_DIR}/' directory.")


if __name__ == "__main__":
    main()
