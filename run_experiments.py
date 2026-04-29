"""
Extra experiments for the report:
  1) Failure case analysis — where does SAC lose to the rule-based baseline?
  2) Stress tests — price spikes, extreme weather, high base load
  3) Seed sensitivity (optional) — train multiple seeds and compare
  4) Structured results summary dumped to a text file

The failure analysis and stress tests are the most valuable for the
Discussion section because they demonstrate honest critical evaluation
rather than just reporting the best-case numbers.

Usage:
    python run_experiments.py
"""

import csv
import os
from datetime import datetime

import numpy as np
import torch

from smart_home_env import SmartHomeEnv
from sac_agent import SACAgent, ReplayBuffer
from test_and_evaluate import (
    rule_based_policy, naive_policy, run_episode,
    run_episode_with_trajectory, cohens_d, confidence_interval_95,
    STATE_DIM, SEASON_NAMES,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "experiment_results"


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_agent(model_path="best_model_actor.pth"):
    env = SmartHomeEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    agent.actor.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    agent.actor.eval()
    return agent


# ================================================================
# 1. Failure cases
# ================================================================
def find_failure_cases(agent, num_episodes=500, base_seed=42):
    """
    Find scenarios where SAC performs worse than the rule-based baseline.
    Good material for the Discussion: honest acknowledgement of weaknesses.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Failure Case Analysis")
    print("When does SAC perform worse than rule-based?")
    print("=" * 60)

    env = SmartHomeEnv()
    seeds = [base_seed + i for i in range(num_episodes)]

    sac_worse_cost = []
    sac_worse_comfort = []
    sac_worse_both = []

    all_comparisons = []

    for seed in seeds:
        sac_r = run_episode(env, "sac", sac_agent=agent, seed=seed)
        rule_r = run_episode(env, "rule", seed=seed)

        cost_diff = sac_r["total_energy_cost"] - rule_r["total_energy_cost"]
        comfort_diff = sac_r["avg_comfort_penalty"] - rule_r["avg_comfort_penalty"]

        record = {
            "seed": seed,
            "season": sac_r["season"],
            "sac_cost": sac_r["total_energy_cost"],
            "rule_cost": rule_r["total_energy_cost"],
            "cost_diff": cost_diff,
            "sac_comfort": sac_r["avg_comfort_penalty"],
            "rule_comfort": rule_r["avg_comfort_penalty"],
            "comfort_diff": comfort_diff,
            "sac_violations": sac_r["comfort_violations"],
            "rule_violations": rule_r["comfort_violations"],
            "sac_max_dev": sac_r["max_temp_dev"],
            "rule_max_dev": rule_r["max_temp_dev"],
        }
        all_comparisons.append(record)

        if cost_diff > 0:
            sac_worse_cost.append(record)
        if comfort_diff > 0:
            sac_worse_comfort.append(record)
        if cost_diff > 0 and comfort_diff > 0:
            sac_worse_both.append(record)

    print(f"\nOut of {num_episodes} episodes:")
    print(f"  SAC costs MORE than rule-based: {len(sac_worse_cost)} "
          f"({len(sac_worse_cost)/num_episodes*100:.1f}%)")
    print(f"  SAC has WORSE comfort:          {len(sac_worse_comfort)} "
          f"({len(sac_worse_comfort)/num_episodes*100:.1f}%)")
    print(f"  SAC worse on BOTH:              {len(sac_worse_both)} "
          f"({len(sac_worse_both)/num_episodes*100:.1f}%)")

    print("\nFailure rate by season (SAC costs more):")
    for s_id, s_name in SEASON_NAMES.items():
        total_season = sum(1 for r in all_comparisons if r["season"] == s_id)
        fails_season = sum(1 for r in sac_worse_cost if r["season"] == s_id)
        if total_season > 0:
            print(f"  {s_name}: {fails_season}/{total_season} "
                  f"({fails_season/total_season*100:.1f}%)")

    if sac_worse_both:
        worst = max(sac_worse_both, key=lambda r: r["cost_diff"])
        print(f"\nWorst case (seed={worst['seed']}, "
              f"{SEASON_NAMES.get(worst['season'], '?')}):")
        print(f"  SAC cost: {worst['sac_cost']:.1f}p vs "
              f"Rule: {worst['rule_cost']:.1f}p "
              f"(SAC {worst['cost_diff']:.1f}p more)")
        print(f"  SAC comfort: {worst['sac_comfort']:.3f} vs "
              f"Rule: {worst['rule_comfort']:.3f}")

    csv_path = os.path.join(OUTPUT_DIR, "failure_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_comparisons[0].keys())
        writer.writeheader()
        writer.writerows(all_comparisons)
    print(f"\nSaved to {csv_path}")

    return all_comparisons


# ================================================================
# 2. Stress tests
# ================================================================
def stress_tests(agent):
    """
    Evaluate the agent under extreme / unusual conditions.
    Good for a "robustness" paragraph in the Discussion.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Stress Tests")
    print("=" * 60)

    env = SmartHomeEnv()
    results = []

    # Scan seeds to bucket them by scenario type
    stress_scenarios = {
        "price_spike": [],
        "extreme_cold": [],
        "extreme_heat": [],
        "cheap_night": [],
        "high_base_load": [],
    }

    for seed in range(2000):
        env.reset(seed=seed)
        if env.price_spike and len(stress_scenarios["price_spike"]) < 20:
            stress_scenarios["price_spike"].append(seed)
        if env.extreme_weather and env.season == 0 and len(stress_scenarios["extreme_cold"]) < 20:
            stress_scenarios["extreme_cold"].append(seed)
        if env.extreme_weather and env.season == 2 and len(stress_scenarios["extreme_heat"]) < 20:
            stress_scenarios["extreme_heat"].append(seed)
        if env.cheap_night and len(stress_scenarios["cheap_night"]) < 20:
            stress_scenarios["cheap_night"].append(seed)
        if env.base_load_event == 1 and len(stress_scenarios["high_base_load"]) < 20:
            stress_scenarios["high_base_load"].append(seed)

    print("\nStress test results:")
    print(f"{'Scenario':<20} {'N':>4}  {'SAC cost':>10} {'Rule cost':>10} "
          f"{'SAC comfort':>12} {'Rule comfort':>12} {'SAC better?':>12}")
    print("-" * 85)

    for scenario_name, seeds in stress_scenarios.items():
        if not seeds:
            continue

        sac_costs = []
        rule_costs = []
        sac_comforts = []
        rule_comforts = []

        for seed in seeds:
            sac_r = run_episode(env, "sac", sac_agent=agent, seed=seed)
            rule_r = run_episode(env, "rule", seed=seed)
            sac_costs.append(sac_r["total_energy_cost"])
            rule_costs.append(rule_r["total_energy_cost"])
            sac_comforts.append(sac_r["avg_comfort_penalty"])
            rule_comforts.append(rule_r["avg_comfort_penalty"])

        sac_better = np.mean(sac_costs) < np.mean(rule_costs)

        print(f"{scenario_name:<20} {len(seeds):>4}  "
              f"{np.mean(sac_costs):>10.1f} {np.mean(rule_costs):>10.1f} "
              f"{np.mean(sac_comforts):>12.3f} {np.mean(rule_comforts):>12.3f} "
              f"{'YES' if sac_better else 'NO':>12}")

        results.append({
            "scenario": scenario_name,
            "n_episodes": len(seeds),
            "sac_mean_cost": np.mean(sac_costs),
            "rule_mean_cost": np.mean(rule_costs),
            "sac_mean_comfort": np.mean(sac_comforts),
            "rule_mean_comfort": np.mean(rule_comforts),
            "sac_cost_advantage": sac_better,
        })

    csv_path = os.path.join(OUTPUT_DIR, "stress_tests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")

    return results


# ================================================================
# 3. Seed sensitivity (optional — requires retraining)
# ================================================================
def seed_sensitivity(num_seeds=3, steps_per_seed=300_000):
    """
    Train the same architecture under several random seeds to check
    whether the reported results are reproducible or lucky.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT: Training Seed Sensitivity")
    print(f"Training {num_seeds} agents with different seeds")
    print("=" * 60)

    eval_seeds = list(range(9000, 9050))
    results = []

    for train_seed in range(num_seeds):
        print(f"\n--- Training seed {train_seed} ---")
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)

        env = SmartHomeEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        agent = SACAgent(state_dim=state_dim, action_dim=action_dim,
                         lr=3e-4, auto_alpha=True)
        buffer = ReplayBuffer(capacity=300_000, state_dim=state_dim,
                              action_dim=action_dim)

        state, _ = env.reset()
        episode = 0
        ep_reward = 0.0

        for step in range(steps_per_seed):
            if step < 5000:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, 0.0)

            if step >= 5000:
                agent.update(buffer, 128)

            ep_reward += reward
            state = next_state

            if done:
                if episode % 300 == 0:
                    print(f"  Seed {train_seed}, ep {episode}: reward={ep_reward:.1f}")
                episode += 1
                ep_reward = 0.0
                state, _ = env.reset()

        eval_env = SmartHomeEnv()
        costs = []
        comforts = []
        for es in eval_seeds:
            r = run_episode(eval_env, "sac", sac_agent=agent, seed=es)
            costs.append(r["total_energy_cost"])
            comforts.append(r["avg_comfort_penalty"])

        results.append({
            "train_seed": train_seed,
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs),
            "mean_comfort": np.mean(comforts),
            "std_comfort": np.std(comforts),
        })
        print(f"  Eval: cost={np.mean(costs):.1f}p ± {np.std(costs):.1f}, "
              f"comfort={np.mean(comforts):.3f}")

    cost_means = [r["mean_cost"] for r in results]
    print(f"\nCost across seeds: {[f'{c:.1f}' for c in cost_means]}")
    print(f"  Range: {max(cost_means) - min(cost_means):.1f}p")
    print(f"  Coefficient of variation: "
          f"{np.std(cost_means)/np.mean(cost_means)*100:.1f}%")

    csv_path = os.path.join(OUTPUT_DIR, "seed_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {csv_path}")

    return results


# ================================================================
# 4. Report-ready summary
# ================================================================
def generate_report_summary(agent, num_episodes=500, base_seed=42):
    """Write a structured text file with every number the report needs."""
    print("\n" + "=" * 60)
    print("Generating structured report summary...")
    print("=" * 60)

    env = SmartHomeEnv()
    seeds = [base_seed + i for i in range(num_episodes)]

    all_results = {}
    for policy in ["naive", "sac", "rule", "random"]:
        all_results[policy] = []
        for seed in seeds:
            r = run_episode(
                env, policy,
                sac_agent=agent if policy == "sac" else None,
                seed=seed,
            )
            all_results[policy].append(r)

    summary_path = os.path.join(OUTPUT_DIR, "report_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SMARTFLEX RESULTS SUMMARY — FOR REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Episodes per policy: {num_episodes}\n")
        f.write("=" * 70 + "\n\n")

        sac_costs = [r["total_energy_cost"] for r in all_results["sac"]]
        naive_costs = [r["total_energy_cost"] for r in all_results["naive"]]
        rule_costs = [r["total_energy_cost"] for r in all_results["rule"]]

        sac_vs_naive = (np.mean(naive_costs) - np.mean(sac_costs)) / np.mean(naive_costs) * 100
        sac_vs_rule = (np.mean(rule_costs) - np.mean(sac_costs)) / np.mean(rule_costs) * 100

        f.write("KEY FINDINGS (use these exact numbers in your report):\n\n")
        f.write(f"  1. SAC reduces daily energy cost by {sac_vs_naive:.1f}% "
                f"compared to a naive thermostat\n")
        f.write(f"     (from {np.mean(naive_costs):.1f}p to {np.mean(sac_costs):.1f}p per day)\n\n")
        f.write(f"  2. SAC reduces cost by {sac_vs_rule:.1f}% compared to "
                f"the rule-based controller\n")
        f.write(f"     (from {np.mean(rule_costs):.1f}p to {np.mean(sac_costs):.1f}p per day)\n\n")

        sac_comfort = [r["avg_comfort_penalty"] for r in all_results["sac"]]
        rule_comfort = [r["avg_comfort_penalty"] for r in all_results["rule"]]
        naive_comfort = [r["avg_comfort_penalty"] for r in all_results["naive"]]

        f.write(f"  3. Comfort penalty: SAC={np.mean(sac_comfort):.3f}, "
                f"Rule={np.mean(rule_comfort):.3f}, "
                f"Naive={np.mean(naive_comfort):.3f}\n")

        if np.mean(sac_comfort) > np.mean(rule_comfort):
            f.write(f"     NOTE: SAC has HIGHER comfort penalty than rule-based.\n")
            f.write(f"     This means SAC trades some comfort for cost savings.\n")
            f.write(f"     DISCUSS THIS TRADE-OFF in your Discussion section.\n")
        f.write("\n")

        from scipy import stats
        _, p_cost = stats.ttest_ind(sac_costs, rule_costs, equal_var=False)
        d_cost = cohens_d(np.array(sac_costs), np.array(rule_costs))

        f.write(f"  4. Statistical significance (SAC vs Rule cost):\n")
        f.write(f"     p-value = {p_cost:.6f} "
                f"({'significant' if p_cost < 0.05 else 'NOT significant'})\n")
        f.write(f"     Cohen's d = {d_cost:.3f} "
                f"({'large' if abs(d_cost) > 0.8 else 'medium' if abs(d_cost) > 0.5 else 'small'} effect)\n")
        f.write(f"     95% CI for SAC cost: {confidence_interval_95(np.array(sac_costs))}\n\n")

        f.write("PER-SEASON BREAKDOWN:\n\n")
        for s_id, s_name in SEASON_NAMES.items():
            s_sac = [r["total_energy_cost"] for r in all_results["sac"] if r["season"] == s_id]
            s_rule = [r["total_energy_cost"] for r in all_results["rule"] if r["season"] == s_id]
            s_naive = [r["total_energy_cost"] for r in all_results["naive"] if r["season"] == s_id]

            if s_sac:
                reduction = (np.mean(s_rule) - np.mean(s_sac)) / np.mean(s_rule) * 100
                f.write(f"  {s_name} (n={len(s_sac)}):\n")
                f.write(f"    SAC={np.mean(s_sac):.1f}p, "
                        f"Rule={np.mean(s_rule):.1f}p, "
                        f"Naive={np.mean(s_naive):.1f}p\n")
                f.write(f"    SAC vs Rule: {reduction:+.1f}%\n\n")

        f.write("\nLIMITATIONS TO DISCUSS IN YOUR REPORT:\n\n")
        f.write("  1. Simulation-only: no real-world deployment or validation\n")
        f.write("  2. Simplified thermal model (single-zone, no humidity)\n")
        f.write("  3. Cloud cover constant per day (real weather varies hourly)\n")
        f.write("  4. No battery degradation modelling\n")
        f.write("  5. Simplified tariff (not real half-hourly Agile prices)\n")
        f.write("  6. No occupancy prediction (agent reacts, doesn't anticipate)\n")
        f.write("  7. Training requires significant compute time\n")
        f.write("  8. No multi-zone temperature modelling\n")
        f.write("  9. Fixed house parameters (no adaptation to different homes)\n")

        f.write("\n\nFUTURE WORK SUGGESTIONS:\n\n")
        f.write("  1. Real-world data integration (Octopus Agile API, weather API)\n")
        f.write("  2. Multi-agent: coordinate multiple homes in a neighbourhood\n")
        f.write("  3. Transfer learning: adapt trained agent to different houses\n")
        f.write("  4. EV charging integration as additional controllable load\n")
        f.write("  5. Sim-to-real transfer with domain randomisation\n")
        f.write("  6. Battery degradation-aware control\n")
        f.write("  7. Multi-objective RL (Pareto-optimal policies)\n")
        f.write("  8. Explainability: interpret what the neural network learned\n")

    print(f"Saved to {summary_path}")
    print("This file contains every number and talking point for your report.")


# ================================================================
if __name__ == "__main__":
    ensure_dir()

    print("Loading trained agent...")
    try:
        agent = load_agent("best_model_actor.pth")
    except FileNotFoundError:
        print("ERROR: best_model_actor.pth not found.")
        print("Run train.py first, then come back to this script.")
        exit(1)

    find_failure_cases(agent, num_episodes=500)
    stress_tests(agent)
    generate_report_summary(agent, num_episodes=500)

    # Uncomment to run the full seed-sensitivity sweep (slow)
    # seed_sensitivity(num_seeds=3, steps_per_seed=300_000)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Results saved to '{OUTPUT_DIR}/' directory")
    print("=" * 60)
