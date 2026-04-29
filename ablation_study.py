"""
Ablation study — sweeps key hyperparameters to understand the agent's
sensitivity to the reward weighting and learning rate.

Two experiments:
  1) Lambda comfort sweep — varies lambda_comfort to map the
     cost-vs-comfort Pareto frontier.
  2) Learning rate sensitivity — compares convergence with different LRs.

These runs are shorter than the main training (200k steps) and their
numbers are noisier than the fully-trained agent's 1.5M step results.
They still produce the trend and frontier shape, which is what the
Discussion section needs.

Usage:
    python ablation_study.py --experiment lambda_sweep
    python ablation_study.py --experiment lr_sensitivity
    python ablation_study.py --experiment both
"""

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from smart_home_env import SmartHomeEnv
from sac_agent import SACAgent, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed evaluation seeds so each ablation config is compared on the same days
EVAL_SEEDS = list(range(9000, 9050))


def quick_train(lambda_comfort=1.0, lr=3e-4, max_steps=200_000,
                label="default"):
    """
    Train a single SAC agent with the given hyperparameters and
    evaluate it on the fixed eval seeds. Returns the aggregated metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"  lambda_comfort={lambda_comfort}, lr={lr}, steps={max_steps}")
    print(f"{'='*60}")

    env = SmartHomeEnv()
    env.lambda_comfort = lambda_comfort

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim=state_dim, action_dim=action_dim,
                     lr=lr, auto_alpha=True)
    buffer = ReplayBuffer(capacity=200_000, state_dim=state_dim,
                          action_dim=action_dim)

    batch_size = 128
    start_steps = 5_000

    state, _ = env.reset()
    episode = 0
    episode_reward = 0.0
    episode_steps = 0

    for step in range(max_steps):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.add(state, action, reward, next_state, 0.0)

        if step >= start_steps:
            agent.update(buffer, batch_size)

        episode_reward += reward
        episode_steps += 1
        state = next_state

        if done:
            if episode % 200 == 0:
                print(f"  Episode {episode}: reward={episode_reward:.1f}")
            episode += 1
            episode_reward = 0.0
            episode_steps = 0
            # Keep lambda override persistent across resets
            env.lambda_comfort = lambda_comfort
            state, _ = env.reset()

    # Evaluation on fixed seeds
    eval_env = SmartHomeEnv()
    costs = []
    comforts = []
    violations = []
    minutes_list = []

    for seed in EVAL_SEEDS:
        state, _ = eval_env.reset(seed=seed)
        done = False
        ep_cost = 0.0
        ep_comfort = 0.0
        ep_viols = 0
        ep_minutes = 0.0
        steps = 0

        while not done:
            action = agent.get_action(state, eval_mode=True)
            state, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_cost += info.get("energy_cost", 0.0)
            ep_comfort += info.get("comfort_penalty", 0.0)
            ep_viols += int(info.get("comfort_violation", 0))
            ep_minutes += info.get("minutes_outside_comfort", 0.0)
            steps += 1

        costs.append(ep_cost)
        comforts.append(ep_comfort / max(1, steps))
        violations.append(ep_viols)
        minutes_list.append(ep_minutes)

    result = {
        "label": label,
        "lambda_comfort": lambda_comfort,
        "lr": lr,
        "max_steps": max_steps,
        "mean_cost": np.mean(costs),
        "std_cost": np.std(costs),
        "mean_comfort": np.mean(comforts),
        "std_comfort": np.std(comforts),
        "mean_violations": np.mean(violations),
        "mean_minutes": np.mean(minutes_list),
    }

    print(f"  Result: cost={result['mean_cost']:.1f}p, "
          f"comfort={result['mean_comfort']:.3f}, "
          f"violations={result['mean_violations']:.1f}")

    return result


def lambda_sweep():
    """Train one agent per lambda_comfort value and collect results."""
    lambdas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    results = []

    for lc in lambdas:
        r = quick_train(
            lambda_comfort=lc,
            max_steps=200_000,
            label=f"lambda_comfort={lc}",
        )
        results.append(r)

    return results


def lr_sensitivity():
    """Compare 3 learning rates to assess SAC's LR sensitivity."""
    lrs = [1e-4, 3e-4, 1e-3]
    results = []

    for lr in lrs:
        r = quick_train(
            lr=lr,
            max_steps=200_000,
            label=f"lr={lr}",
        )
        results.append(r)

    return results


def save_results(results, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {filename}")


def plot_pareto(results, filename="plots/pareto_frontier.png"):
    """Plot the cost-comfort Pareto frontier from the lambda sweep."""
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    costs = [r["mean_cost"] for r in results]
    comforts = [r["mean_comfort"] for r in results]
    labels = [r["label"] for r in results]

    ax.scatter(costs, comforts, s=80, c="tab:blue", zorder=5)

    for i, label in enumerate(labels):
        lc = results[i]["lambda_comfort"]
        ax.annotate(f"λ={lc}", (costs[i], comforts[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Sort by cost so the frontier line connects points in order
    sorted_idx = np.argsort(costs)
    ax.plot([costs[i] for i in sorted_idx],
            [comforts[i] for i in sorted_idx],
            color="tab:blue", alpha=0.4, linestyle="--")

    ax.set_xlabel("Mean daily cost (pence)")
    ax.set_ylabel("Mean comfort penalty")
    ax.set_title("Cost–comfort Pareto frontier (λ_comfort sweep)")
    ax.grid(True, alpha=0.3)

    # "Ideal direction" annotation — points towards origin (low cost, low penalty)
    ax.annotate("", xy=(min(costs) * 0.9, min(comforts) * 0.5),
                xytext=(min(costs) * 0.9 + 50, min(comforts) * 0.5 + 1),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
    ax.text(min(costs) * 0.9 + 55, min(comforts) * 0.5 + 1.2,
            "Ideal\n(low cost,\nlow penalty)", fontsize=8, color="green")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved Pareto plot to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",
                        choices=["lambda_sweep", "lr_sensitivity", "both"],
                        default="both")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.random.seed(42)
    torch.manual_seed(42)

    if args.experiment in ["lambda_sweep", "both"]:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Lambda comfort sweep")
        print("=" * 60)
        lambda_results = lambda_sweep()
        save_results(lambda_results, f"ablation_lambda_{run_id}.csv")
        plot_pareto(lambda_results)

    if args.experiment in ["lr_sensitivity", "both"]:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Learning rate sensitivity")
        print("=" * 60)
        lr_results = lr_sensitivity()
        save_results(lr_results, f"ablation_lr_{run_id}.csv")

    print("\nAblation study complete.")
