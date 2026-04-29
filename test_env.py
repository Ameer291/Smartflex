"""
Unit tests for SmartHomeEnv.

Verifies physical correctness, API compliance, and reward behaviour.
Run with: python -m pytest test_env.py -v
"""

import numpy as np
import pytest

from smart_home_env import SmartHomeEnv


class TestEnvironmentBasics:
    """Gymnasium API compliance and basic dimensions."""

    def test_state_dimension(self):
        env = SmartHomeEnv()
        state, _ = env.reset(seed=42)
        assert state.shape == (14,), f"Expected 14-dim state, got {state.shape}"

    def test_action_dimension(self):
        env = SmartHomeEnv()
        assert env.action_space.shape == (2,)

    def test_reset_returns_valid_state(self):
        env = SmartHomeEnv()
        state, info = env.reset(seed=42)
        assert env.observation_space.contains(state), "Reset state outside observation bounds"

    def test_episode_length(self):
        """96 steps per day at 15-minute resolution."""
        env = SmartHomeEnv()
        state, _ = env.reset(seed=42)
        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == 96, f"Episode lasted {steps} steps, expected 96"

    def test_reproducibility(self):
        """Same seed -> identical episode."""
        env1 = SmartHomeEnv()
        env2 = SmartHomeEnv()

        s1, _ = env1.reset(seed=123)
        s2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(s1, s2)

        action = np.array([0.5, 0.3], dtype=np.float32)
        s1, r1, _, _, _ = env1.step(action)
        s2, r2, _, _, _ = env2.step(action)
        np.testing.assert_array_almost_equal(s1, s2, decimal=5)
        assert abs(r1 - r2) < 1e-5


class TestSolarPhysics:
    """Solar PV should produce sensible output patterns."""

    def test_solar_peaks_at_noon(self):
        """Peak generation should be at or near hour 12."""
        env = SmartHomeEnv()
        env.reset(seed=10)
        env.cloud_factor = 1.0
        env.solar_max_kw = 4.0

        solar_by_hour = {}
        for step in range(96):
            hour = step * 0.25
            angle = 2 * np.pi * hour / 24.0
            # Average many samples to smooth out the flicker term
            vals = [env._solar_power(angle) for _ in range(100)]
            solar_by_hour[hour] = np.mean(vals)

        peak_hour = max(solar_by_hour, key=solar_by_hour.get)
        assert 11.0 <= peak_hour <= 13.0, (
            f"Solar peaks at hour {peak_hour}, expected ~12.0 (noon)"
        )

    def test_solar_zero_at_night(self):
        """No generation overnight."""
        env = SmartHomeEnv()
        env.reset(seed=10)
        env.cloud_factor = 1.0
        env.solar_max_kw = 4.0

        for hour in [0, 1, 2, 3, 4, 22, 23]:
            angle = 2 * np.pi * hour / 24.0
            solar = env._solar_power(angle)
            assert solar == 0.0, f"Solar should be 0 at hour {hour}, got {solar}"

    def test_cloud_factor_reduces_output(self):
        """Heavy overcast should be well below clear sky."""
        env = SmartHomeEnv()
        env.reset(seed=10)
        env.solar_max_kw = 4.0
        noon_angle = 2 * np.pi * 12.0 / 24.0

        env.cloud_factor = 1.0
        clear = np.mean([env._solar_power(noon_angle) for _ in range(100)])

        env.cloud_factor = 0.1
        cloudy = np.mean([env._solar_power(noon_angle) for _ in range(100)])

        assert cloudy < clear * 0.2, "Cloudy output should be <20% of clear sky"

    def test_minimum_cloud_factor(self):
        """Cloud factor minimum is 0.1, never zero."""
        env = SmartHomeEnv()
        for seed in range(200):
            env.reset(seed=seed)
            assert env.cloud_factor >= 0.1, (
                f"Cloud factor {env.cloud_factor} below minimum 0.1 at seed {seed}"
            )


class TestBattery:
    """Battery physics: SOC bounds and efficiency losses."""

    def test_soc_stays_in_bounds(self):
        """SOC must stay in [0, 1] no matter what actions we take."""
        env = SmartHomeEnv()
        env.reset(seed=42)

        for _ in range(96):
            action = np.array([1.0, 0.0])
            state, _, _, _, _ = env.step(action)
            assert 0.0 <= state[0] <= 1.0, f"SOC out of bounds: {state[0]}"

        env.reset(seed=42)
        for _ in range(96):
            action = np.array([-1.0, 0.0])
            state, _, _, _, _ = env.step(action)
            assert 0.0 <= state[0] <= 1.0, f"SOC out of bounds: {state[0]}"

    def test_charging_efficiency_loss(self):
        """Energy stored should be less than energy drawn (round-trip loss)."""
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.soc = 0.5
        env.p_solar_t = 0.0

        initial_soc = env.soc
        action = np.array([1.0, 0.0])
        env.step(action)

        energy_commanded = env.batt_max_power_kw * env.dt_hours
        energy_stored = (env.soc - initial_soc) * env.batt_capacity_kwh

        assert energy_stored < energy_commanded, (
            f"Stored {energy_stored:.4f} kWh but commanded {energy_commanded:.4f} kWh — "
            f"efficiency losses should reduce stored amount"
        )

    def test_discharging_efficiency_loss(self):
        """SOC should drop when discharging."""
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.soc = 0.8

        action = np.array([-1.0, 0.0])
        _, _, _, _, info = env.step(action)

        p_grid = info.get("p_grid_kw", 0.0)
        assert env.soc < 0.8, "SOC should decrease on discharge"


class TestHVAC:
    """HVAC: heating warms, cooling cools, off does nothing active."""

    def test_heating_increases_temperature(self):
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.t_in = 15.0
        env.t_out = 5.0

        initial_temp = env.t_in
        action = np.array([0.0, 1.0])
        env.step(action)

        assert env.t_in > initial_temp, (
            f"Heating should increase temp: {initial_temp} → {env.t_in}"
        )

    def test_cooling_decreases_temperature(self):
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.t_in = 28.0
        env.t_out = 30.0

        initial_temp = env.t_in
        action = np.array([0.0, -1.0])
        env.step(action)

        assert env.t_in < initial_temp, (
            f"Cooling should decrease temp: {initial_temp} → {env.t_in}"
        )

    def test_hvac_off_no_active_heating(self):
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.t_in = 21.0
        env.t_out = 21.0

        action = np.array([0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["hvac_power_kw"] == 0.0, "HVAC power should be 0 when action is 0"

    def test_temperature_stays_in_bounds(self):
        """Temperature should clamp to [-5, 45]°C even under constant heating."""
        env = SmartHomeEnv()
        env.reset(seed=42)

        for _ in range(96):
            action = np.array([0.0, 1.0])
            state, _, _, _, _ = env.step(action)
            assert -5.0 <= state[3] <= 45.0, f"Temp out of bounds: {state[3]}"


class TestSeasonEncoding:
    """Season is one-hot encoded, not ordinal."""

    def test_season_one_hot(self):
        env = SmartHomeEnv()
        for seed in range(50):
            state, _ = env.reset(seed=seed)
            season_vec = state[8:12]
            assert sum(season_vec) == 1.0, f"Season not one-hot: {season_vec}"
            assert max(season_vec) == 1.0
            assert min(season_vec) == 0.0

    def test_all_seasons_reachable(self):
        env = SmartHomeEnv()
        seasons_seen = set()
        for seed in range(200):
            env.reset(seed=seed)
            seasons_seen.add(env.season)
        assert seasons_seen == {0, 1, 2, 3}, f"Missing seasons: {seasons_seen}"


class TestPricing:
    """Electricity price sanity checks."""

    def test_peak_price_higher_than_offpeak(self):
        env = SmartHomeEnv()
        offpeak_price = env.base_price_profile[8]   # hour 2
        peak_price = env.base_price_profile[72]     # hour 18
        assert peak_price > offpeak_price, (
            f"Peak ({peak_price}) should exceed off-peak ({offpeak_price})"
        )

    def test_prices_non_negative(self):
        env = SmartHomeEnv()
        for seed in range(50):
            env.reset(seed=seed)
            for step in range(96):
                hour = step * env.dt_hours
                price = env._compute_price(hour)
                assert price >= 0.0, f"Negative price {price} at hour {hour}"


class TestRewardSignal:
    """Reward function behaviour."""

    def test_comfort_violation_when_occupied(self):
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.t_in = 25.0
        env.occupancy_scenario = 1

        action = np.array([0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["comfort_violation"] == 1.0, "Should flag comfort violation"
        assert info["comfort_penalty"] > 0, "Should have non-zero comfort penalty"

    def test_no_violation_when_unoccupied(self):
        env = SmartHomeEnv()
        env.reset(seed=42)
        env.t_in = 25.0
        env.occupancy_scenario = 0

        # Force step to a working-hours slot (scenario 0 away 09-17)
        env.step_idx = 40

        action = np.array([0.0, 0.0])
        _, _, _, _, info = env.step(action)

        assert info["comfort_violation"] == 0.0, "No violation when unoccupied"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
