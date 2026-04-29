"""
Evaluation script: compares the trained SAC agent against baselines
(naive thermostat, rule-based, random) on identical scenarios.

Reports:
  - Aggregate stats with 95% CIs
  - Per-season breakdown
  - Welch's t-test + Cohen's d
  - Full per-step trajectory CSV for plotting

Usage:
    python test_and_evaluate.py --model best_model_actor.pth --episodes 500
"""

import argparse
import csv
import logging
from datetime import datetime

import numpy as np
import torch
from scipy import stats

from smart_home_env import SmartHomeEnv
from sac_agent import SACAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State vector indices (14-dim layout):
#   0: SOC           1: P_solar       2: P_price       3: T_in
#   4: T_out         5: sin_h         6: cos_h         7: occupancy
#   8-11: season one-hot (W, Sp, Su, Au)
#  12: cloud_factor  13: base_load_multiplier
IDX_SOC = 0
IDX_SOLAR = 1
IDX_PRICE = 2
IDX_T_IN = 3
IDX_T_OUT = 4
IDX_OCC = 7
IDX_SEASON_W = 8
IDX_SEASON_SP = 9
IDX_SEASON_SU = 10
IDX_SEASON_AU = 11
IDX_CLOUD = 12
IDX_BASE_LOAD = 13

STATE_DIM = 14
SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

# Thresholds used by the rule-based baseline
RULE_CONFIG = {
    "low_price": 22.0,
    "high_price": 35.0,
    "T_setpoint": 21.0,
    "comfort_band": 0.5,
    "spike_threshold": 60.0,
}


def setup_logger(log_file: str = "eval_run.log") -> logging.Logger:
    logger = logging.getLogger("SmartflexEval")
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


def get_season_from_state(state: np.ndarray) -> int:
    """Decode one-hot season back to integer 0-3."""
    season_vec = state[IDX_SEASON_W:IDX_SEASON_AU + 1]
    return int(np.argmax(season_vec))


def rule_based_policy(state: np.ndarray) -> np.ndarray:
    """
    Price-aware rule-based baseline.

    HVAC: thermostat with both heating and cooling. Tight 0.5°C band
    when occupied, wider tolerance when unoccupied.

    Battery: charge from solar or during off-peak, discharge during peak
    or spikes. Thresholds are fixed — doesn't adapt to the actual day's
    price profile (this is part of why it loses to SAC).
    """
    assert len(state) == STATE_DIM, f"Expected {STATE_DIM}-dim state, got {len(state)}"

    soc = float(state[IDX_SOC])
    pv = float(state[IDX_SOLAR])
    price = float(state[IDX_PRICE])
    T_in = float(state[IDX_T_IN])
    occupancy = float(state[IDX_OCC])

    T_set = RULE_CONFIG["T_setpoint"]
    band = RULE_CONFIG["comfort_band"]

    # HVAC logic
    if occupancy > 0.5:
        if T_in < T_set - band:
            a_hvac = 1.0
        elif T_in > T_set + band:
            a_hvac = -1.0
        else:
            # Gentle maintenance inside the band
            if T_in < T_set:
                a_hvac = 0.3
            elif T_in > T_set:
                a_hvac = -0.3
            else:
                a_hvac = 0.0
    else:
        # Unoccupied: only act to stop extreme drift
        if T_in < T_set - 4.0:
            a_hvac = 0.5
        elif T_in > T_set + 4.0:
            a_hvac = -0.5
        elif T_in < T_set - 2.0:
            a_hvac = 0.2
        elif T_in > T_set + 2.0:
            a_hvac = -0.2
        else:
            a_hvac = 0.0

    # Battery logic
    if price >= RULE_CONFIG["spike_threshold"] and soc > 0.2:
        a_bat = -1.0
    elif pv > 1.0 and soc < 0.95:
        a_bat = 1.0
    elif price >= RULE_CONFIG["high_price"] and soc > 0.3:
        a_bat = -0.8
    elif price <= RULE_CONFIG["low_price"] and soc < 0.8:
        a_bat = 0.7
    elif pv > 0.3 and soc < 0.9:
        a_bat = 0.5
    else:
        a_bat = 0.0

    return np.array([a_bat, a_hvac], dtype=np.float32)


def naive_policy(state: np.ndarray) -> np.ndarray:
    """
    Dumb thermostat: heat if cold, nothing otherwise.
    No cooling, no battery management, no price awareness.
    Represents a homeowner who just sets a setpoint and forgets about it.
    """
    assert len(state) == STATE_DIM, f"Expected {STATE_DIM}-dim state, got {len(state)}"

    T_in = float(state[IDX_T_IN])
    T_set = 21.0

    a_hvac = 1.0 if T_in < T_set else 0.0
    a_bat = 0.0

    return np.array([a_bat, a_hvac], dtype=np.float32)


def run_episode(env, policy_fn: str, sac_agent=None,
                deterministic: bool = True, seed: int = None) -> dict:
    """Run one 24-hour episode and return aggregate metrics."""
    state, _ = env.reset(seed=seed)
    done = False

    total_reward = 0.0
    total_energy_cost = 0.0
    total_comfort_pen = 0.0
    total_violations = 0
    total_minutes_outside = 0.0
    steps = 0
    temp_devs = []
    season = env.season

    while not done:
        if policy_fn == "sac":
            action = sac_agent.get_action(state, eval_mode=deterministic)
        elif policy_fn == "rule":
            action = rule_based_policy(state)
        elif policy_fn == "naive":
            action = naive_policy(state)
        elif policy_fn == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown policy: {policy_fn}")

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += float(reward)
        total_energy_cost += float(info.get("energy_cost", 0.0))
        total_comfort_pen += float(info.get("comfort_penalty", 0.0))
        total_violations += int(info.get("comfort_violation", 0))
        total_minutes_outside += float(info.get("minutes_outside_comfort", 0.0))
        temp_devs.append(float(info.get("temp_dev", 0.0)))
        steps += 1

    return {
        "total_reward": total_reward,
        "total_energy_cost": total_energy_cost,
        "avg_comfort_penalty": total_comfort_pen / max(1, steps),
        "comfort_violations": total_violations,
        "minutes_outside_comfort": total_minutes_outside,
        "max_temp_dev": max(temp_devs) if temp_devs else 0.0,
        "mean_temp_dev": float(np.mean(temp_devs)) if temp_devs else 0.0,
        "season": season,
    }


def run_episode_with_trajectory(env, policy_fn: str, sac_agent=None,
                                 seed: int = None) -> dict:
    """Run one episode and return full per-step trajectory for plotting."""
    state, _ = env.reset(seed=seed)
    done = False

    hours = []
    temps_in = []
    temps_out = []
    prices = []
    socs = []
    p_grid = []
    p_solar = []
    comfort_dev = []
    occupancy = []
    actions_bat = []
    actions_hvac = []
    hvac_modes = []

    total_cost = 0.0
    total_minutes = 0.0
    step_idx = 0
    season = env.season

    while not done:
        if policy_fn == "sac":
            action = sac_agent.get_action(state, eval_mode=True)
        elif policy_fn == "rule":
            action = rule_based_policy(state)
        elif policy_fn == "naive":
            action = naive_policy(state)
        elif policy_fn == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown policy: {policy_fn}")

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        hour = (step_idx % env.steps_per_day) * env.dt_hours

        hours.append(hour)
        # Reading env state directly for correct time alignment
        temps_in.append(float(env.t_in))
        temps_out.append(float(env.t_out))
        prices.append(float(info.get("price", 0.0)))
        socs.append(float(env.soc))
        p_grid.append(float(info.get("p_grid_kw", 0.0)))
        p_solar.append(float(info.get("p_solar_kw", 0.0)))
        comfort_dev.append(float(info.get("temp_dev", 0.0)))
        occupancy.append(float(info.get("occupancy", 0.0)))
        actions_bat.append(float(info.get("a_bat", action[0])))
        actions_hvac.append(float(info.get("a_hvac", action[1])))
        hvac_modes.append(info.get("hvac_mode", "off"))

        total_cost += float(info.get("energy_cost", 0.0))
        total_minutes += float(info.get("minutes_outside_comfort", 0.0))
        step_idx += 1

    return {
        "hours": np.array(hours),
        "temps_in": np.array(temps_in),
        "temps_out": np.array(temps_out),
        "prices": np.array(prices),
        "socs": np.array(socs),
        "p_grid": np.array(p_grid),
        "p_solar": np.array(p_solar),
        "comfort_dev": np.array(comfort_dev),
        "occupancy": np.array(occupancy),
        "actions_bat": np.array(actions_bat),
        "actions_hvac": np.array(actions_hvac),
        "hvac_modes": hvac_modes,
        "total_cost_pence": total_cost,
        "total_minutes_outside": total_minutes,
        "season": season,
    }


def cohens_d(group1, group2):
    """
    Cohen's d effect size (pooled SD).
    Rough guide: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def confidence_interval_95(data):
    """95% CI for the mean. Returns (mean, mean) for n < 2."""
    n = len(data)
    if n < 2:
        m = np.mean(data)
        return (m, m)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return ci


def format_ci(data):
    mean = np.mean(data)
    ci = confidence_interval_95(data)
    return f"{mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SAC agent against baselines"
    )
    parser.add_argument("--model", default="best_model_actor.pth",
                        help="Path to trained SAC actor weights")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of evaluation episodes per policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for reproducible evaluation")
    parser.add_argument("--trajectory-seeds", type=int, nargs="+",
                        default=[42, 100, 200, 300],
                        help="Seeds for detailed trajectory plots")
    args = parser.parse_args()

    logger = setup_logger()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"===== Evaluation run: {run_id} =====")
    logger.info(f"Model: {args.model} | Episodes: {args.episodes} | Seed: {args.seed}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        env = SmartHomeEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        assert state_dim == STATE_DIM, (
            f"Environment state_dim={state_dim} but expected {STATE_DIM}. "
            f"Check smart_home_env.py matches this evaluation script."
        )

        sac_agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
        sac_agent.actor.load_state_dict(
            torch.load(args.model, map_location=device, weights_only=True)
        )
        sac_agent.actor.eval()

        # Run every policy on the same seed list so comparisons are fair.
        # Fresh env per policy (carry_over_soc=False) means starting SOC
        # is seeded identically instead of depending on the previous policy.
        results = {"naive": [], "sac": [], "rule": [], "random": []}
        test_seeds = [args.seed + i for i in range(args.episodes)]

        for policy in ["naive", "sac", "rule", "random"]:
            policy_env = SmartHomeEnv(carry_over_soc=False)
            print(f"\nEvaluating {policy.upper()} policy ({args.episodes} episodes)...")
            for ep, seed in enumerate(test_seeds):
                r = run_episode(
                    policy_env, policy,
                    sac_agent=sac_agent if policy == "sac" else None,
                    seed=seed,
                )
                results[policy].append(r)

                if (ep + 1) % 100 == 0:
                    print(
                        f"  [{policy.upper()}] ep {ep+1:>3}: "
                        f"reward={r['total_reward']:>8.2f}  "
                        f"cost={r['total_energy_cost']:>7.2f}p  "
                        f"comfort={r['avg_comfort_penalty']:.3f}  "
                        f"violations={r['comfort_violations']}  "
                        f"max_dev={r['max_temp_dev']:.2f}°C"
                    )

        # Convert to arrays for analysis
        def extract(policy, key):
            return np.array([r[key] for r in results[policy]])

        def extract_by_season(policy, key, season):
            return np.array([
                r[key] for r in results[policy] if r["season"] == season
            ])

        naive_costs = extract("naive", "total_energy_cost")
        sac_costs = extract("sac", "total_energy_cost")
        rule_costs = extract("rule", "total_energy_cost")
        rand_costs = extract("random", "total_energy_cost")

        naive_rewards = extract("naive", "total_reward")
        sac_rewards = extract("sac", "total_reward")
        rule_rewards = extract("rule", "total_reward")
        rand_rewards = extract("random", "total_reward")

        naive_comfort = extract("naive", "avg_comfort_penalty")
        sac_comfort = extract("sac", "avg_comfort_penalty")
        rule_comfort = extract("rule", "avg_comfort_penalty")
        rand_comfort = extract("random", "avg_comfort_penalty")

        naive_viols = extract("naive", "comfort_violations")
        sac_viols = extract("sac", "comfort_violations")
        rule_viols = extract("rule", "comfort_violations")

        naive_minutes = extract("naive", "minutes_outside_comfort")
        sac_minutes = extract("sac", "minutes_outside_comfort")
        rule_minutes = extract("rule", "minutes_outside_comfort")

        naive_max_dev = extract("naive", "max_temp_dev")
        sac_max_dev = extract("sac", "max_temp_dev")
        rule_max_dev = extract("rule", "max_temp_dev")

        # Statistical tests (Welch's t-test handles unequal variance)
        _, p_cost_sr = stats.ttest_ind(sac_costs, rule_costs, equal_var=False)
        _, p_reward_sr = stats.ttest_ind(sac_rewards, rule_rewards, equal_var=False)
        _, p_comfort_sr = stats.ttest_ind(sac_comfort, rule_comfort, equal_var=False)
        _, p_cost_sn = stats.ttest_ind(sac_costs, naive_costs, equal_var=False)
        _, p_cost_rnd = stats.ttest_ind(sac_costs, rand_costs, equal_var=False)

        d_cost_sr = cohens_d(sac_costs, rule_costs)
        d_cost_sn = cohens_d(sac_costs, naive_costs)
        d_reward_sr = cohens_d(sac_rewards, rule_rewards)
        d_comfort_sr = cohens_d(sac_comfort, rule_comfort)

        sac_vs_naive_pct = (np.mean(naive_costs) - np.mean(sac_costs)) / np.mean(naive_costs) * 100
        sac_vs_rule_pct = (np.mean(rule_costs) - np.mean(sac_costs)) / np.mean(rule_costs) * 100
        rule_vs_naive_pct = (np.mean(naive_costs) - np.mean(rule_costs)) / np.mean(naive_costs) * 100

        season_names = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}

        summary = f"""
============================================================
  Results over {args.episodes} episodes (identical scenarios)
============================================================

AGGREGATE METRICS (mean [95% CI]):

  Policy     | Cost (pence/day)           | Comfort penalty          | Violations   | Max temp dev
  -----------|----------------------------|--------------------------|--------------|-------------
  Naive      | {format_ci(naive_costs):>26} | {format_ci(naive_comfort):>24} | {np.mean(naive_viols):>6.1f}       | {np.mean(naive_max_dev):.2f}°C
  SAC (AI)   | {format_ci(sac_costs):>26} | {format_ci(sac_comfort):>24} | {np.mean(sac_viols):>6.1f}       | {np.mean(sac_max_dev):.2f}°C
  Rule-based | {format_ci(rule_costs):>26} | {format_ci(rule_comfort):>24} | {np.mean(rule_viols):>6.1f}       | {np.mean(rule_max_dev):.2f}°C
  Random     | {format_ci(rand_costs):>26} | {format_ci(rand_comfort):>24} |              |

COST REDUCTIONS:
  SAC vs Naive:      {sac_vs_naive_pct:>+.1f}%
  SAC vs Rule-based: {sac_vs_rule_pct:>+.1f}%
  Rule vs Naive:     {rule_vs_naive_pct:>+.1f}%

STATISTICAL SIGNIFICANCE (Welch's t-test, α=0.05):
  SAC vs Rule — cost:    p={p_cost_sr:.4f} {'***' if p_cost_sr < 0.001 else '**' if p_cost_sr < 0.01 else '*' if p_cost_sr < 0.05 else 'ns'}  Cohen's d={d_cost_sr:.3f}
  SAC vs Rule — reward:  p={p_reward_sr:.4f} {'***' if p_reward_sr < 0.001 else '**' if p_reward_sr < 0.01 else '*' if p_reward_sr < 0.05 else 'ns'}  Cohen's d={d_reward_sr:.3f}
  SAC vs Rule — comfort: p={p_comfort_sr:.4f} {'***' if p_comfort_sr < 0.001 else '**' if p_comfort_sr < 0.01 else '*' if p_comfort_sr < 0.05 else 'ns'}  Cohen's d={d_comfort_sr:.3f}
  SAC vs Naive — cost:   p={p_cost_sn:.4f} {'***' if p_cost_sn < 0.001 else '**' if p_cost_sn < 0.01 else '*' if p_cost_sn < 0.05 else 'ns'}  Cohen's d={d_cost_sn:.3f}
  SAC vs Random — cost:  p={p_cost_rnd:.4f} {'***' if p_cost_rnd < 0.001 else '**' if p_cost_rnd < 0.01 else '*' if p_cost_rnd < 0.05 else 'ns'}

PER-SEASON COST BREAKDOWN (mean pence/day):
"""

        for s_id, s_name in season_names.items():
            s_naive = extract_by_season("naive", "total_energy_cost", s_id)
            s_sac = extract_by_season("sac", "total_energy_cost", s_id)
            s_rule = extract_by_season("rule", "total_energy_cost", s_id)
            n_eps = len(s_sac)
            if n_eps > 0:
                summary += (
                    f"  {s_name:>8} (n={n_eps:>3}): "
                    f"Naive={np.mean(s_naive):>7.1f}  "
                    f"SAC={np.mean(s_sac):>7.1f}  "
                    f"Rule={np.mean(s_rule):>7.1f}  "
                    f"SAC vs Rule: {(np.mean(s_rule) - np.mean(s_sac)) / np.mean(s_rule) * 100:>+.1f}%\n"
                )

        print(summary)
        logger.info(summary)

        # Save per-episode results
        eval_csv = f"eval_results_{run_id}.csv"
        with open(eval_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "policy", "episode", "seed", "season",
                "total_reward", "total_energy_cost_p",
                "avg_comfort_penalty", "comfort_violations",
                "minutes_outside", "max_temp_dev", "mean_temp_dev",
            ])
            for policy in ["naive", "sac", "rule", "random"]:
                for i, r in enumerate(results[policy]):
                    writer.writerow([
                        policy, i, test_seeds[i], r["season"],
                        r["total_reward"], r["total_energy_cost"],
                        r["avg_comfort_penalty"], r["comfort_violations"],
                        r["minutes_outside_comfort"], r["max_temp_dev"],
                        r["mean_temp_dev"],
                    ])

        print(f"\nSaved aggregate results to {eval_csv}")
        logger.info(f"Saved evaluation results to {eval_csv}")

        # Per-step trajectories for the requested seeds
        print("\nGenerating per-step trajectory data...")
        traj_csv = f"trajectories_{run_id}.csv"

        with open(traj_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "seed", "policy", "season", "hour",
                "T_in", "T_out", "price", "SOC",
                "p_grid_kw", "p_solar_kw",
                "comfort_dev", "occupancy",
                "a_bat", "a_hvac", "hvac_mode",
            ])

            for seed in args.trajectory_seeds:
                for policy in ["naive", "sac", "rule"]:
                    traj_env = SmartHomeEnv(carry_over_soc=False)
                    traj = run_episode_with_trajectory(
                        traj_env, policy,
                        sac_agent=sac_agent if policy == "sac" else None,
                        seed=seed,
                    )
                    for j in range(len(traj["hours"])):
                        writer.writerow([
                            seed, policy, traj["season"],
                            f"{traj['hours'][j]:.2f}",
                            f"{traj['temps_in'][j]:.2f}",
                            f"{traj['temps_out'][j]:.2f}",
                            f"{traj['prices'][j]:.2f}",
                            f"{traj['socs'][j]:.4f}",
                            f"{traj['p_grid'][j]:.4f}",
                            f"{traj['p_solar'][j]:.4f}",
                            f"{traj['comfort_dev'][j]:.4f}",
                            f"{traj['occupancy'][j]:.0f}",
                            f"{traj['actions_bat'][j]:.4f}",
                            f"{traj['actions_hvac'][j]:.4f}",
                            traj["hvac_modes"][j],
                        ])

                    print(
                        f"  Seed {seed}, {policy.upper()}: "
                        f"cost={traj['total_cost_pence']:.1f}p, "
                        f"season={season_names.get(traj['season'], '?')}"
                    )

        print(f"Saved trajectory data to {traj_csv}")
        logger.info(f"Saved trajectory data to {traj_csv}")

    except FileNotFoundError:
        logger.error(
            f"Model file '{args.model}' not found. "
            f"Train the agent first, or specify --model path."
        )
        raise
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise
