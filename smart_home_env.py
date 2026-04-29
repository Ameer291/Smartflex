"""
SmartHomeEnv: Gymnasium environment for a UK home with solar PV,
battery storage and HVAC (heating + cooling) over a single 24-hour day.

Observation (14-dim):
    [SOC, P_solar, P_price, T_in, T_out, sin_h, cos_h,
     occupancy, season_w, season_sp, season_su, season_au,
     cloud_factor, base_load_multiplier]

Action (2-dim, both in [-1, 1]):
    a_bat  : +1 = max charge, -1 = max discharge
    a_hvac : +1 = full heating, -1 = full cooling, 0 = off

Key simplifications (discussed in report limitations):
  - Cloud cover is constant for the whole day.
  - Tariff is a simplified time-of-use profile, not real Agile data.
  - Single-zone thermal model, no humidity, no multi-room dynamics.
  - Observations have no sensor noise.
  - 93% round-trip battery efficiency is split sqrt(0.93) each way.
"""

import gymnasium as gym
import numpy as np


class SmartHomeEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, carry_over_soc: bool = True, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.carry_over_soc = carry_over_soc
        self._last_soc = 0.5

        # Observation bounds. Season uses one-hot encoding (indices 8-11)
        # rather than ordinal so the network doesn't see a false ordering.
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, -5.0, -20.0, -1.0, -1.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 10.0, 150.0, 45.0, 45.0, 1.0, 1.0,
                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
                dtype=np.float32,
            ),
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # 15-minute time steps -> 96 steps per day
        self.dt_hours = 0.25
        self.steps_per_day = int(24 / self.dt_hours)

        # Battery parameters (10 kWh, 3 kW charge/discharge, 93% round-trip)
        self.batt_max_capacity_kwh = 10.0
        self.batt_max_power_kw = 3.0
        self.battery_rt_efficiency = 0.93
        self.battery_one_way_efficiency = np.sqrt(self.battery_rt_efficiency)

        # Standby / always-on household load
        self.base_load_kw = 0.5

        # HVAC parameters. Typical air-source heat pump COPs.
        self.hvac_max_power_kw = 4.0
        # 0.15 kW/°C ~ 150 W/K: well-insulated UK semi-detached.
        # SAP methodology gives 100 W/K (best) to 300 W/K (worst) as typical range.
        self.heat_loss_coeff = 0.15
        self.hvac_cop_heating = 2.5
        self.hvac_cop_cooling = 2.0
        self.comfort_setpoint = 21.0

        # Reward weights. Ablation study in report sweeps lambda_comfort.
        self.lambda_cost = 1.0
        self.lambda_comfort = 1.0

        # Simplified time-of-use tariff (p/kWh) inspired by Octopus Agile:
        #   00-05 off-peak (18), 05-16 standard (24),
        #   16-20 peak (40),     20-00 evening (26)
        prices = []
        for step in range(self.steps_per_day):
            hour = step * self.dt_hours
            if 0.0 <= hour < 5.0:
                prices.append(18.0)
            elif 5.0 <= hour < 16.0:
                prices.append(24.0)
            elif 16.0 <= hour < 20.0:
                prices.append(40.0)
            else:
                prices.append(26.0)
        self.base_price_profile = np.array(prices, dtype=np.float32)

        # Season-dependent weather and solar profiles for UK climate
        self.season_config = {
            0: {"name": "Winter", "t_range": (-5.0, 8.0), "solar": (0.2, 1.5), "amp": 3.0},
            1: {"name": "Spring", "t_range": (4.0, 15.0), "solar": (1.5, 3.0), "amp": 5.0},
            2: {"name": "Summer", "t_range": (12.0, 32.0), "solar": (2.5, 4.5), "amp": 7.0},
            3: {"name": "Autumn", "t_range": (2.0, 14.0), "solar": (0.8, 2.5), "amp": 4.0},
        }

        # Smart Export Guarantee rate. Capped at 3.6 kW single-phase inverter limit.
        self.export_rate_p_kwh = 15.0
        self.max_export_kw = 3.6

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Using self.np_random (seeded by super().reset) for full reproducibility.
        rng = self.np_random

        # Sample the day's scenario
        self.season = int(rng.integers(0, 4))
        cfg = self.season_config[self.season]

        self.t_out_base = float(rng.uniform(*cfg["t_range"]))
        self.temp_amplitude = cfg["amp"]

        # Extreme weather event (~3% of days)
        self.extreme_weather = bool(rng.random() < 0.03)
        if self.extreme_weather:
            if self.season in [2, 3]:
                self.t_out_base += float(rng.uniform(5.0, 10.0))
            else:
                self.t_out_base -= float(rng.uniform(5.0, 10.0))
            # Clamp to UK historical extremes (-27.2°C to 40.3°C, relaxed here)
            self.t_out_base = float(np.clip(self.t_out_base, -15.0, 40.0))

        self.solar_max_kw = float(rng.uniform(*cfg["solar"]))

        # Cloud factor sampled once per day. Minimum 0.1 because even
        # heavy overcast still produces ~10% of clear-sky diffuse radiation.
        cloud_options = [0.1, 0.2, 0.4, 0.7, 1.0]
        cloud_probs = [0.12, 0.18, 0.25, 0.25, 0.20]
        self.cloud_factor = float(rng.choice(cloud_options, p=cloud_probs))

        self.batt_capacity_kwh = self.batt_max_capacity_kwh

        # Starting SOC. If carrying over, apply small overnight self-discharge.
        if self.carry_over_soc:
            drift = float(rng.uniform(-0.02, 0.0))
            self.soc = float(np.clip(self._last_soc + drift, 0.05, 0.95))
        else:
            self.soc = float(np.clip(0.4 + 0.2 * rng.standard_normal(), 0.05, 0.95))

        # Starting indoor temperature, close to setpoint
        self.t_in = float(
            np.clip(20.5 + 1.2 * rng.standard_normal(), self.t_out_base - 2.0, 28.0)
        )

        # Occupancy scenarios:
        #   0: working adult (away 09-17)   55%
        #   1: weekend / WFH (home all day) 20%
        #   2: retired (home all day)       15%
        #   3: night shift (home 09-17)     10%
        self.occupancy_scenario = int(
            rng.choice([0, 1, 2, 3], p=[0.55, 0.20, 0.15, 0.10])
        )

        # Base load events: normal / high (EV, guests) / low (away)
        self.base_load_event = int(rng.choice([0, 1, 2], p=[0.88, 0.07, 0.05]))
        load_multipliers = {0: 1.0, 1: 2.2, 2: 0.3}
        self.base_load_multiplier = load_multipliers[self.base_load_event]
        self.effective_base_load = self.base_load_kw * self.base_load_multiplier

        # Daily price variation (~12%) plus rare spike / cheap-night events.
        # Both can occur together (spike during day, cheap rate overnight).
        self.price_scale = float(rng.uniform(0.88, 1.12))

        self.price_spike = bool(rng.random() < 0.05)
        if self.price_spike:
            self.price_scale *= float(rng.uniform(1.3, 2.0))

        self.cheap_night = bool(rng.random() < 0.05)

        self.step_idx = 0
        self.t_out = self._outdoor_temp(0.0)
        self.p_solar_t = self._solar_power(0.0)

        # Cache price so observation and step() use the same value
        self._current_price = self._compute_price(0.0)

        return self._get_state(), {
            "energy_cost": 0.0,
            "comfort_penalty": 0.0,
            "comfort_violation": 0.0,
        }

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        a_bat = float(action[0])
        a_hvac = float(action[1])

        hour = (self.step_idx % self.steps_per_day) * self.dt_hours
        angle = 2 * np.pi * hour / 24.0

        # Update external conditions
        self.t_out = self._outdoor_temp(angle)
        self.p_solar_t = self._solar_power(angle)
        p_solar = self.p_solar_t

        self._current_price = self._compute_price(hour)
        price = self._current_price

        occupancy = self._get_occupancy(hour)

        # --- Battery ---
        p_batt_cmd = self.batt_max_power_kw * a_bat
        e_cmd = p_batt_cmd * self.dt_hours

        max_charge_kwh = (1.0 - self.soc) * self.batt_capacity_kwh
        max_discharge_kwh = self.soc * self.batt_capacity_kwh

        if e_cmd > 0:
            e_cmd_clipped = min(e_cmd, max_charge_kwh)
        else:
            e_cmd_clipped = max(e_cmd, -max_discharge_kwh)

        # Symmetric efficiency split: grid pays full amount on charge,
        # house receives reduced amount on discharge. Losses are heat.
        if e_cmd_clipped > 0:
            e_stored = e_cmd_clipped * self.battery_one_way_efficiency
            self.soc = float(np.clip(
                self.soc + e_stored / self.batt_capacity_kwh, 0.0, 1.0
            ))
            p_batt_grid = e_cmd_clipped / self.dt_hours
    
        else:
            self.soc = float(np.clip(
                self.soc + e_cmd_clipped / self.batt_capacity_kwh, 0.0, 1.0
            ))
            e_delivered = e_cmd_clipped * self.battery_one_way_efficiency
            p_batt_grid = e_delivered / self.dt_hours

        # --- HVAC ---
        if a_hvac >= 0:
            hvac_power = self.hvac_max_power_kw * a_hvac
            heat_transfer = self.hvac_cop_heating * hvac_power
        else:
            hvac_power = self.hvac_max_power_kw * abs(a_hvac)
            heat_transfer = -self.hvac_cop_cooling * hvac_power

        # Single-zone lumped thermal model.
        # Implicit thermal mass is ~1 kWh/°C (light/medium UK home).
        heat_loss = self.heat_loss_coeff * (self.t_in - self.t_out)
        self.t_in = float(np.clip(
            self.t_in + (heat_transfer - heat_loss) * self.dt_hours,
            -5.0, 45.0
        ))

        # --- Grid balance ---
        # Positive = importing, negative = exporting surplus
        p_grid = self.effective_base_load + hvac_power - p_solar + p_batt_grid

        if p_grid < 0.0:
            p_export = min(abs(p_grid), self.max_export_kw)
            e_export = p_export * self.dt_hours
            energy_cost = -(e_export * self.export_rate_p_kwh)
            p_grid = 0.0
        else:
            energy_cost = p_grid * self.dt_hours * price

        # --- Reward ---

        # Scale cost so it's on similar magnitude to comfort penalty
        normalized_cost = energy_cost / 50.0

        temp_dev = abs(self.t_in - self.comfort_setpoint)

        # Comfort penalty. Stepped thresholds when occupied so the agent
        # strongly avoids big deviations; softer quadratic when unoccupied
        # to allow pre-heating / pre-cooling flexibility.
        if occupancy > 0.5:
            base_comfort = temp_dev * 10.0

            if temp_dev > 2.0:
                comfort_penalty = 40.0
            elif temp_dev > 1.5:
                comfort_penalty = 20.0
            elif temp_dev > 1.0:
                comfort_penalty = 10.0 + base_comfort
            else:
                comfort_penalty = base_comfort
        else:
            if temp_dev > 5.0:
                comfort_penalty = 4.0 * temp_dev
            elif temp_dev > 3.0:
                comfort_penalty = 2.0 * temp_dev
            else:
                comfort_penalty = temp_dev ** 2

        # Flag a "violation" if temp drifted >1°C while occupied
        comfort_violation = 1.0 if temp_dev > 1.0 and occupancy > 0.5 else 0.0
        minutes_outside_comfort = comfort_violation * (self.dt_hours * 60.0)

        # Extra penalty per-minute of discomfort (sustained > brief)
        time_penalty = minutes_outside_comfort * 0.5 if occupancy > 0.5 else 0.0

        total_comfort_penalty = comfort_penalty + time_penalty

        reward = -(
            self.lambda_cost * normalized_cost
            + self.lambda_comfort * total_comfort_penalty
        )

        # Small bonus for healthy battery SOC window (discourages deep cycling)
        if 0.2 <= self.soc <= 0.8:
            reward += 0.05

        self.step_idx += 1

        # Day boundary is a truncation (the home keeps existing), not a terminal
        terminated = False
        truncated = self.step_idx >= self.steps_per_day

        if truncated:
            self._last_soc = self.soc

        info = {
            "energy_cost": energy_cost,
            "comfort_penalty": total_comfort_penalty,
            "comfort_violation": comfort_violation,
            "price": price,
            "p_grid_kw": p_grid,
            "p_solar_kw": p_solar,
            "hvac_power_kw": hvac_power,
            "hvac_mode": "off" if a_hvac == 0 else ("heating" if a_hvac > 0 else "cooling"),
            "occupancy": occupancy,
            "season": self.season,
            "cloud_factor": self.cloud_factor,
            "price_spike": self.price_spike,
            "cheap_night": self.cheap_night,
            "is_weekend": self.occupancy_scenario == 1,
            "extreme_weather": self.extreme_weather,
            "base_load_event": self.base_load_event,
            "batt_capacity_kwh": self.batt_capacity_kwh,
            "temp_dev": temp_dev,
            "normalized_cost": normalized_cost,
            "minutes_outside_comfort": minutes_outside_comfort,
            "a_bat": a_bat,
            "a_hvac": a_hvac,
        }

        return self._get_state(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_price(self, hour: float) -> float:
        """Return import price (p/kWh) for a given hour."""
        step_in_day = int(hour / self.dt_hours) % self.steps_per_day
        base_price = self.base_price_profile[step_in_day]
        if self.cheap_night and (hour < 5.0 or hour >= 23.0):
            price = float(base_price * 0.1)
        else:
            price = float(base_price * self.price_scale)
        return max(0.0, price)

    def _get_occupancy(self, hour: float) -> float:
        """Return 1.0 if home is occupied at this hour, else 0.0."""
        if self.occupancy_scenario == 0:
            return 0.0 if 9.0 <= hour < 17.0 else 1.0
        elif self.occupancy_scenario == 3:
            return 1.0 if 9.0 <= hour < 17.0 else 0.0
        else:
            return 1.0

    def _outdoor_temp(self, angle: float) -> float:
        """
        Diurnal temperature cycle. sin(angle - pi/2) gives coldest at
        midnight, warmest at noon. Real UK peak is closer to 15:00 but
        this is a reasonable simplification.
        """
        return self.t_out_base + self.temp_amplitude * np.sin(angle - np.pi / 2)

    def _solar_power(self, angle: float) -> float:
        """
        Solar PV output. Peaks at noon (hour 12), zero outside daylight.
        Flicker term (0.96-1.0) models brief cloud variation on top of
        the day's constant cloud factor.
        """
        raw = max(0.0, self.solar_max_kw * np.sin(angle - np.pi / 2))
        flicker = float(self.np_random.uniform(0.96, 1.0))
        return raw * self.cloud_factor * flicker

    def _get_state(self) -> np.ndarray:
        """Build the 14-dim observation vector from current state."""
        hour = (self.step_idx % self.steps_per_day) * self.dt_hours
        angle = 2 * np.pi * hour / 24.0

        sin_h = float(np.sin(angle))
        cos_h = float(np.cos(angle))

        p_solar = self.p_solar_t
        price = self._current_price
        occupancy = self._get_occupancy(hour)

        season_onehot = [0.0, 0.0, 0.0, 0.0]
        season_onehot[self.season] = 1.0

        return np.array(
            [
                self.soc,
                p_solar,
                price,
                self.t_in,
                self.t_out,
                sin_h,
                cos_h,
                occupancy,
                season_onehot[0],
                season_onehot[1],
                season_onehot[2],
                season_onehot[3],
                self.cloud_factor,
                self.base_load_multiplier,
            ],
            dtype=np.float32,
        )

    def render(self):
        """Simple text rendering for debugging."""
        if self.render_mode == "human":
            hour = (self.step_idx % self.steps_per_day) * self.dt_hours
            season_name = self.season_config[self.season]["name"]
            print(
                f"[{season_name}] Hour {hour:05.2f} | "
                f"T_in={self.t_in:.1f}°C  T_out={self.t_out:.1f}°C | "
                f"SOC={self.soc:.2f}  Solar={self.p_solar_t:.2f}kW | "
                f"Price={self._current_price:.1f}p/kWh"
            )
