"""
Training script for the SAC agent on SmartHomeEnv.

Features:
  - Periodic validation on fixed seeds for comparable tracking
  - Best-model saving by validation reward
  - Cosine-annealing LR schedule for all three networks
  - Auto alpha logging
  - Graceful Ctrl+C checkpoint
  - CSVs for training and validation metrics
"""

import csv
import logging
import signal
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch

from smart_home_env import SmartHomeEnv
from sac_agent import SACAgent, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed seeds for validation so metrics are comparable across training steps
VALIDATION_SEEDS = list(range(9000, 9040))


def setup_logger(log_file: str = "run.log") -> logging.Logger:
    logger = logging.getLogger("SmartflexTrain")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def validate(agent, seeds=None, num_episodes=40):
    """
    Run deterministic episodes on fixed seeds and aggregate metrics.
    Using the same seeds every validation call means improvement is real,
    not just luckier scenarios.
    """
    if seeds is None:
        seeds = VALIDATION_SEEDS[:num_episodes]

    val_env = SmartHomeEnv(carry_over_soc=False)

    total_rewards = []
    total_costs = []
    total_comfort = []
    total_violations = []
    total_minutes_outside = []
    season_costs = {0: [], 1: [], 2: [], 3: []}

    for seed in seeds:
        state, _ = val_env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        ep_comfort = 0.0
        ep_violations = 0
        ep_minutes_outside = 0.0
        steps = 0
        ep_season = val_env.season

        while not done:
            action = agent.get_action(state, eval_mode=True)
            state, reward, terminated, truncated, info = val_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_cost += info.get("energy_cost", 0.0)
            ep_comfort += info.get("comfort_penalty", 0.0)
            ep_violations += int(info.get("comfort_violation", 0))
            ep_minutes_outside += info.get("minutes_outside_comfort", 0.0)
            steps += 1

        total_rewards.append(ep_reward)
        total_costs.append(ep_cost)
        total_comfort.append(ep_comfort / max(1, steps))
        total_violations.append(ep_violations)
        total_minutes_outside.append(ep_minutes_outside)
        season_costs[ep_season].append(ep_cost)

    return {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_cost": np.mean(total_costs),
        "std_cost": np.std(total_costs),
        "mean_comfort": np.mean(total_comfort),
        "mean_violations": np.mean(total_violations),
        "mean_minutes_outside": np.mean(total_minutes_outside),
        "season_costs": {
            s: np.mean(c) if c else 0.0 for s, c in season_costs.items()
        },
    }


def save_checkpoint(agent, prefix, logger):
    """Save actor and both Q-networks."""
    torch.save(agent.actor.state_dict(), f"{prefix}_actor.pth")
    torch.save(agent.q1.state_dict(), f"{prefix}_q1.pth")
    torch.save(agent.q2.state_dict(), f"{prefix}_q2.pth")
    logger.info(f"Saved checkpoint: {prefix}")


def train():
    logger = setup_logger()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"===== New training run: {run_id} =====")

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Handle Ctrl+C so we get a checkpoint instead of losing everything
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        if interrupted:
            logger.warning("Force exit requested.")
            sys.exit(1)
        interrupted = True
        logger.warning("Interrupt received — will save checkpoint and exit after current episode.")

    signal.signal(signal.SIGINT, signal_handler)

    # Declare agent upfront so the except handler can reach it even if
    # construction fails part-way through
    agent = None

    try:
        env = SmartHomeEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        logger.info(f"Environment state_dim={state_dim}, action_dim={action_dim}")
        logger.info(f"Device: {device}")

        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            auto_alpha=True,
        )
        buffer = ReplayBuffer(
            capacity=500_000, state_dim=state_dim, action_dim=action_dim
        )

        # Training hyperparameters
        max_steps = 1_500_000
        batch_size = 128
        start_steps = 10_000
        checkpoint_every = 100
        validate_every = 50

        # Cosine LR schedule for all three optimisers
        total_updates = max_steps - start_steps
        scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.actor_opt, T_max=total_updates, eta_min=1e-5
        )
        scheduler_q1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.q1_opt, T_max=total_updates, eta_min=1e-5
        )
        scheduler_q2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            agent.q2_opt, T_max=total_updates, eta_min=1e-5
        )

        best_val_reward = -float("inf")

        state, info = env.reset()
        episode_reward = 0.0
        episode_energy_cost = 0.0
        episode_comfort_penalty = 0.0
        episode_violations = 0
        episode = 0
        episode_steps = 0

        # Smoothed logging windows
        window_size = 50
        recent_rewards = deque(maxlen=window_size)
        recent_costs = deque(maxlen=window_size)
        recent_comfort = deque(maxlen=window_size)

        logger.info(
            f"{'Step':>8} | {'Episode':>8} | {'AvgR(50ep)':>10} | "
            f"{'AvgCost(p)':>10} | {'AvgComfort':>11} | "
            f"{'Alpha':>7} | {'Q1Loss':>8} | {'ActorLoss':>10}"
        )
        logger.info("-" * 100)

        csv_filename = f"training_log_{run_id}.csv"
        val_csv_filename = f"validation_log_{run_id}.csv"

        with open(csv_filename, mode="w", newline="") as csv_file, \
             open(val_csv_filename, mode="w", newline="") as val_csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "episode",
                "total_reward",
                "total_energy_cost_p",
                "avg_comfort_penalty",
                "violations",
                "alpha",
            ])

            val_csv_writer = csv.writer(val_csv_file)
            val_csv_writer.writerow([
                "episode",
                "mean_reward",
                "std_reward",
                "mean_cost",
                "std_cost",
                "mean_comfort",
                "mean_violations",
                "mean_minutes_outside",
                "season_0_cost",
                "season_1_cost",
                "season_2_cost",
                "season_3_cost",
            ])

            for step in range(max_steps):
                if interrupted:
                    logger.info("Saving interrupt checkpoint...")
                    save_checkpoint(agent, f"interrupt_{run_id}_ep{episode}", logger)
                    break

                # Warm-up: random actions until the buffer has something useful
                if step < start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.get_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store done=0.0: this is a continuing task where day boundaries
                # are truncations, so Q-values should flow across them.
                buffer.add(state, action, reward, next_state, 0.0)

                update_info = {}
                if step >= start_steps:
                    update_info = agent.update(buffer, batch_size)

                    scheduler_actor.step()
                    scheduler_q1.step()
                    scheduler_q2.step()

                episode_reward += float(reward)
                episode_energy_cost += float(info.get("energy_cost", 0.0))
                episode_comfort_penalty += float(info.get("comfort_penalty", 0.0))
                episode_violations += int(info.get("comfort_violation", 0))
                episode_steps += 1

                state = next_state

                if done:
                    recent_rewards.append(episode_reward)
                    recent_costs.append(episode_energy_cost)
                    recent_comfort.append(
                        episode_comfort_penalty / max(1, episode_steps)
                    )

                    avg_r = float(np.mean(recent_rewards))
                    avg_cost = float(np.mean(recent_costs))
                    avg_comf = float(np.mean(recent_comfort))

                    log_msg = (
                        f"{step:8d} | {episode:8d} | {avg_r:10.2f} | "
                        f"{avg_cost:10.2f} | {avg_comf:11.3f}"
                    )
                    if update_info:
                        log_msg += (
                            f" | {update_info.get('alpha', 0):7.4f}"
                            f" | {update_info.get('q1_loss', 0):8.4f}"
                            f" | {update_info.get('actor_loss', 0):10.4f}"
                        )
                    logger.info(log_msg)

                    csv_writer.writerow([
                        episode,
                        episode_reward,
                        episode_energy_cost,
                        episode_comfort_penalty / max(1, episode_steps),
                        episode_violations,
                        update_info.get("alpha", agent.alpha),
                    ])
                    csv_file.flush()

                    # Validation
                    if episode > 0 and episode % validate_every == 0:
                        val_metrics = validate(agent)
                        sc = val_metrics["season_costs"]

                        logger.info(
                            f"[VALIDATION] Episode {episode}: "
                            f"Reward={val_metrics['mean_reward']:.2f}"
                            f"±{val_metrics['std_reward']:.2f}, "
                            f"Cost={val_metrics['mean_cost']:.2f}p"
                            f"±{val_metrics['std_cost']:.2f}, "
                            f"Comfort={val_metrics['mean_comfort']:.3f}, "
                            f"Violations={val_metrics['mean_violations']:.1f}, "
                            f"MinOutside={val_metrics['mean_minutes_outside']:.1f} | "
                            f"Season costs: W={sc[0]:.1f} Sp={sc[1]:.1f} "
                            f"Su={sc[2]:.1f} Au={sc[3]:.1f}"
                        )

                        val_csv_writer.writerow([
                            episode,
                            val_metrics["mean_reward"],
                            val_metrics["std_reward"],
                            val_metrics["mean_cost"],
                            val_metrics["std_cost"],
                            val_metrics["mean_comfort"],
                            val_metrics["mean_violations"],
                            val_metrics["mean_minutes_outside"],
                            sc[0], sc[1], sc[2], sc[3],
                        ])
                        val_csv_file.flush()

                        # Keep the best model we've seen so far
                        if val_metrics["mean_reward"] > best_val_reward:
                            best_val_reward = val_metrics["mean_reward"]
                            save_checkpoint(agent, "best_model", logger)
                            logger.info(
                                f"  ** New best validation reward: "
                                f"{best_val_reward:.2f} **"
                            )

                    # Regular rolling checkpoint
                    if episode > 0 and episode % checkpoint_every == 0:
                        save_checkpoint(
                            agent,
                            f"checkpoint_{run_id}_ep{episode}",
                            logger,
                        )

                    episode += 1
                    episode_reward = 0.0
                    episode_energy_cost = 0.0
                    episode_comfort_penalty = 0.0
                    episode_violations = 0
                    episode_steps = 0
                    state, info = env.reset()

        save_checkpoint(agent, "sac_final", logger)
        logger.info(f"Training finished. Training log: {csv_filename}")

        # Larger final validation on a separate held-out set
        logger.info("Running final validation with 100 episodes...")
        final_seeds = list(range(8000, 8100))
        final_val = validate(agent, seeds=final_seeds, num_episodes=100)
        sc = final_val["season_costs"]
        logger.info(
            f"[FINAL VALIDATION]: "
            f"Reward={final_val['mean_reward']:.2f}"
            f"±{final_val['std_reward']:.2f}, "
            f"Cost={final_val['mean_cost']:.2f}p"
            f"±{final_val['std_cost']:.2f}, "
            f"Comfort={final_val['mean_comfort']:.3f}, "
            f"Violations={final_val['mean_violations']:.1f}, "
            f"MinOutside={final_val['mean_minutes_outside']:.1f} | "
            f"Season costs: W={sc[0]:.1f} Sp={sc[1]:.1f} "
            f"Su={sc[2]:.1f} Au={sc[3]:.1f}"
        )

    except Exception as e:
        logger.exception(f"Training failed with exception: {e}")
        # Only try to save if the agent actually got constructed
        if agent is not None:
            try:
                save_checkpoint(agent, f"emergency_{run_id}", logger)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    train()
