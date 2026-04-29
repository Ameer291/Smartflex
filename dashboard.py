"""
Streamlit dashboard for live demonstration of the SAC agent.

Two modes:
  - Single policy: full detail panels for one policy on one scenario
  - Compare all: overlay all policies on the same scenario

Usage:
    streamlit run dashboard.py
"""

from typing import Optional

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from smart_home_env import SmartHomeEnv
from sac_agent import SACAgent
from test_and_evaluate import rule_based_policy, naive_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEASON_NAMES = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
POLICY_COLOURS = {
    "SAC (AI)": "tab:blue",
    "Rule-based": "tab:orange",
    "Naive thermostat": "tab:gray",
    "Random baseline": "tab:red",
}


@st.cache_resource
def load_sac_agent(model_path: str) -> Optional[SACAgent]:
    """Load trained actor weights. Cached so we only load once per session."""
    try:
        env_tmp = SmartHomeEnv()
        state_dim = env_tmp.observation_space.shape[0]
        action_dim = env_tmp.action_space.shape[0]

        agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        agent.actor.load_state_dict(state_dict)
        agent.actor.eval()
        return agent
    except Exception as e:
        st.warning(f"Could not load SAC model from '{model_path}': {e}")
        return None


def run_one_day(policy_label: str, sac_agent: Optional[SACAgent],
                seed: int = 0) -> dict:
    """Simulate a single day under the chosen policy, returning per-step data."""
    env = SmartHomeEnv()
    state, _ = env.reset(seed=seed)

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

    total_cost_pence = 0.0
    total_minutes_outside = 0.0

    done = False
    step_idx = 0
    season = env.season

    while not done:
        if policy_label == "SAC (AI)":
            if sac_agent is None:
                action = env.action_space.sample()
            else:
                action = sac_agent.get_action(state, eval_mode=True)
        elif policy_label == "Rule-based":
            action = rule_based_policy(state)
        elif policy_label == "Naive thermostat":
            action = naive_policy(state)
        else:
            action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        hour = (step_idx % env.steps_per_day) * env.dt_hours

        hours.append(hour)
        temps_in.append(env.t_in)
        temps_out.append(env.t_out)
        prices.append(info.get("price", 0.0))
        socs.append(env.soc)
        p_grid.append(info.get("p_grid_kw", 0.0))
        p_solar.append(info.get("p_solar_kw", env.p_solar_t))
        comfort_dev.append(info.get("temp_dev", 0.0))
        occupancy.append(info.get("occupancy", 0.0))
        actions_bat.append(info.get("a_bat", action[0]))
        actions_hvac.append(info.get("a_hvac", action[1]))
        hvac_modes.append(info.get("hvac_mode", "off"))

        total_cost_pence += float(info.get("energy_cost", 0.0))
        total_minutes_outside += float(info.get("minutes_outside_comfort", 0.0))

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
        "total_cost_pence": total_cost_pence,
        "total_minutes_outside": total_minutes_outside,
        "season": season,
    }


def plot_single_policy(results: dict, policy_label: str):
    """Detail plots for a single policy run."""
    hours = results["hours"]
    colour = POLICY_COLOURS.get(policy_label, "tab:blue")

    # Temperature and comfort band
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(hours, results["temps_in"], label="Indoor", color=colour, linewidth=1.5)
    ax1.plot(hours, results["temps_out"], label="Outdoor", color="black",
             alpha=0.4, linestyle="--")
    ax1.axhline(21.0, color="green", linestyle="--", alpha=0.6, label="Setpoint (21°C)")
    ax1.fill_between(hours, 20.0, 22.0, color="green", alpha=0.08, label="±1°C band")

    # Highlight occupied periods
    occ = results["occupancy"]
    for j in range(len(occ) - 1):
        if occ[j] > 0.5:
            ax1.axvspan(hours[j], hours[j + 1], color="yellow", alpha=0.06)

    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Indoor temperature and comfort band")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # Grid and solar power
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(hours, results["p_grid"], label="Grid import (kW)", color=colour)
    ax2.plot(hours, results["p_solar"], label="Solar PV (kW)", color="gold", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Grid import and solar generation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # Battery SOC
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(hours, results["socs"], label="SOC", color=colour)
    ax3.axhline(0.2, color="red", linestyle=":", alpha=0.4, label="Healthy range")
    ax3.axhline(0.8, color="red", linestyle=":", alpha=0.4)
    ax3.set_xlabel("Hour of day")
    ax3.set_ylabel("State of charge")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("Battery state of charge")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    # Price profile
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(hours, results["prices"], color="tab:orange", linewidth=1.2)
    ax4.set_xlabel("Hour of day")
    ax4.set_ylabel("Price (p/kWh)")
    ax4.set_title("Electricity price over the day")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

    # Agent actions
    fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    ax5a.plot(hours, results["actions_bat"], color=colour, alpha=0.8)
    ax5a.axhline(0, color="black", linewidth=0.5)
    ax5a.set_ylabel("Battery action")
    ax5a.set_ylim(-1.1, 1.1)
    ax5a.set_title("Agent actions over the day")
    ax5a.text(0.01, 0.9, "Charge ↑", transform=ax5a.transAxes, fontsize=7, color="green")
    ax5a.text(0.01, 0.05, "Discharge ↓", transform=ax5a.transAxes, fontsize=7, color="red")
    ax5a.grid(True, alpha=0.3)

    ax5b.plot(hours, results["actions_hvac"], color=colour, alpha=0.8)
    ax5b.axhline(0, color="black", linewidth=0.5)
    ax5b.set_ylabel("HVAC action")
    ax5b.set_xlabel("Hour of day")
    ax5b.set_ylim(-1.1, 1.1)
    ax5b.text(0.01, 0.9, "Heating ↑", transform=ax5b.transAxes, fontsize=7, color="red")
    ax5b.text(0.01, 0.05, "Cooling ↓", transform=ax5b.transAxes, fontsize=7, color="tab:cyan")
    ax5b.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig5)


def plot_comparison(all_results: dict):
    """Overlay plots for all policies on the same scenario."""
    policies = list(all_results.keys())
    ref = all_results[policies[0]]
    hours = ref["hours"]

    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(hours, ref["temps_out"], color="black", alpha=0.3, linestyle="--",
             label="Outdoor")
    for policy, res in all_results.items():
        ax1.plot(hours, res["temps_in"], color=POLICY_COLOURS[policy],
                 label=policy, linewidth=1.2)
    ax1.axhline(21.0, color="green", linestyle="--", alpha=0.5)
    ax1.fill_between(hours, 20.0, 22.0, color="green", alpha=0.06)
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Indoor temperature — all policies")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    for policy, res in all_results.items():
        ax2.plot(hours, res["socs"], color=POLICY_COLOURS[policy],
                 label=policy, linewidth=1.2)
    ax2.set_ylabel("Battery SOC")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Battery state of charge — all policies")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    for policy, res in all_results.items():
        ax3.plot(hours, res["p_grid"], color=POLICY_COLOURS[policy],
                 label=policy, linewidth=1.0, alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("Grid power (kW)")
    ax3.set_xlabel("Hour of day")
    ax3.set_title("Grid import — all policies")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

    # Summary metrics side-by-side
    st.markdown("### Cost and comfort comparison")
    cols = st.columns(len(policies))
    for i, (policy, res) in enumerate(all_results.items()):
        cost_pounds = res["total_cost_pence"] / 100.0
        occ_mask = res["occupancy"] > 0.5
        if occ_mask.any():
            within = (res["comfort_dev"][occ_mask] <= 1.0).sum() / occ_mask.sum() * 100
        else:
            within = 100.0

        with cols[i]:
            st.metric(policy, f"£{cost_pounds:.2f}")
            st.caption(f"{res['total_minutes_outside']:.0f} min outside ±1°C")
            st.caption(f"{within:.0f}% time in comfort")


def main():
    st.set_page_config(page_title="SmartFlex Dashboard", layout="centered")
    st.title("SmartFlex: Smart Home Energy Control Demo")

    st.markdown("""
This dashboard simulates a single 24-hour period for a UK home with
rooftop solar PV, a home battery, and electric heating/cooling.
Compare an **AI controller (SAC)** with baselines on identical scenarios.
""")

    mode = st.radio("View mode", ["Single policy", "Compare all policies"],
                    horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        if mode == "Single policy":
            policy_choice = st.selectbox(
                "Control policy",
                ["SAC (AI)", "Rule-based", "Naive thermostat", "Random baseline"],
            )
        else:
            policy_choice = None
    with col2:
        scenario_seed = st.slider(
            "Scenario seed (weather / prices / occupancy)",
            min_value=0, max_value=100, value=0, step=1,
        )

    st.markdown("---")

    sac_agent = load_sac_agent("best_model_actor.pth")
    if sac_agent is None:
        st.warning(
            "SAC model not found at 'best_model_actor.pth'. "
            "Train the agent first. SAC will use random actions as fallback."
        )

    if st.button("Run 24-hour simulation", type="primary"):
        if mode == "Single policy":
            with st.spinner(f"Simulating {policy_choice}..."):
                results = run_one_day(policy_choice, sac_agent, seed=scenario_seed)

            season_name = SEASON_NAMES.get(results["season"], "Unknown")
            st.info(f"Scenario: {season_name} | Seed: {scenario_seed}")

            plot_single_policy(results, policy_choice)

            st.markdown("### Daily summary")
            col_a, col_b, col_c = st.columns(3)
            cost_pounds = results["total_cost_pence"] / 100.0
            with col_a:
                st.metric("Total energy cost", f"£{cost_pounds:.2f}")
            with col_b:
                st.metric("Minutes outside ±1°C", f"{results['total_minutes_outside']:.0f} min")
            with col_c:
                occ_mask = results["occupancy"] > 0.5
                if occ_mask.any():
                    within = (results["comfort_dev"][occ_mask] <= 1.0).sum() / occ_mask.sum() * 100
                else:
                    within = 100.0
                st.metric("Occupied time in comfort", f"{within:.0f}%")

        else:
            # Run every policy on the same seed and overlay them
            all_results = {}
            policies_to_run = ["SAC (AI)", "Rule-based", "Naive thermostat"]

            with st.spinner("Simulating all policies on the same scenario..."):
                for p in policies_to_run:
                    all_results[p] = run_one_day(p, sac_agent, seed=scenario_seed)

            season_name = SEASON_NAMES.get(all_results["SAC (AI)"]["season"], "Unknown")
            st.info(f"Scenario: {season_name} | Seed: {scenario_seed}")

            plot_comparison(all_results)

        st.success("Simulation complete. Change seed or policy to compare different scenarios.")


if __name__ == "__main__":
    main()
