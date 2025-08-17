"""
CustomRewardWrapper

This module provides the CustomRewardWrapper class which reconstructs the original
reward shaping used by the autonomous driving environment outside of the base
environment implementation. The base environment intentionally returns zero
(reward-less) steps; this wrapper composes a shaped reward from environment
observables so reward logic can be configured, swapped, or tested independently
from the environment dynamics.

Key behavior
- Reconstructs reward as a weighted sum of: progress, deviation-from-center-line,
  speed shaping, and termination-related bonuses/penalties.
- Uses defaults intended to mirror the previous environment: k_progress=0.0,
  k_deviation=100.0, k_speed=1.0. All coefficients are configurable via the
  `cfg` argument.
- Adds an optional finished bonus and an optional timeout penalty (for
  TimeLimit truncations).
- Tracks previous progress internally to compute incremental progress reward.
- When the base env returns info dictionaries, the wrapper looks for two
  optional structures in `info`: `termination` (dict of boolean flags like
  'off_track' / 'finished' / 'speed_high' / 'speed_low') and
  `termination_rewards` (numeric values used for termination-related rewards).
  If present, those values are applied on termination.
- On truncation, if `info.get("TimeLimit.truncated")` is True, the
  `timeout_penalty` is applied.

Expected environment interface
- The wrapped environment should expose `current_ground_truth` (a dict)
  containing at least:
    - 'road_progress_at_closest_point' : progress along the road at the
      closest reference point (float-like)
    - 'distance_to_closest_point' : distance magnitude to the center line
      (float-like)
- The wrapper reads the vehicle longitudinal velocity from
  `env.car.vx` (converted to km/h internally). If these attributes are
  missing the wrapper falls back to sensible zeros.

Configuration (cfg keys)
- k_progress (float): multiplier for progress increment (default 0.0)
- k_deviation (float): multiplier for deviation-from-line reward (default 100.0)
- k_speed (float): multiplier for speed-shaped reward (default 1.0)
- finished_bonus (float): additional reward added when `termination['finished']`
  is True (default 0.0)
- timeout_penalty (float): penalty applied when the episode is truncated by a
  TimeLimit (default 0.0)

Usage examples
- Wrap an environment with default shaping:
    env = AutonomousDrivingEnv(...)  # environment that exposes expected attrs
    env = CustomRewardWrapper(env)

- Customize coefficients:
    cfg = { 'k_deviation': 50.0, 'k_speed': 0.5, 'finished_bonus': 100.0 }
    env = CustomRewardWrapper(env, cfg=cfg)

- Typical step/return semantics remain: step(action) -> (obs, reward, terminated,
  truncated, info). The wrapper returns the reconstructed scalar reward as the
  second element.

Notes and tips
- If you want to disable any shaped term, set its coefficient to 0 in `cfg`.
- The wrapper assumes the base environment's own reward is not needed. If your
  environment does provide a meaningful reward, wrapping will add the shaped
  reward on top of it (the current implementation assumes base env reward is
  zero but does not explicitly subtract it).
- The wrapper tracks `_prev_progress` across steps and resets it on `reset()`.

"""

import gymnasium as gym
from typing import Optional


class CustomRewardWrapper(gym.Wrapper):
    """
    A custom wrapper that reconstructs the environment's previous reward logic
    (progress, deviation from line, speed shaping, and termination rewards)
    while the base environment returns zero reward. This keeps reward shaping
    out of the env and makes it configurable/replaceable.
    """

    def __init__(self, env: gym.Env, cfg: Optional[dict] = None):
        super().__init__(env)
        self.cfg = cfg or {}
        # Coefficients for previous env reward composition
        # Previous env used: 0.0*progress + 100.0*deviation + 1.0*speed + termination
        # Maintain same defaults here; allow override via cfg
        self.k_progress = self.cfg.get("k_progress", 0.0)
        self.k_deviation = self.cfg.get("k_deviation", 100.0)
        self.k_speed = self.cfg.get("k_speed", 1.0)
        # Bonus when finished (not present previously, default 0)
        self.finished_bonus = self.cfg.get("finished_bonus", 0.0)
        # Timeout penalty if a TimeLimit truncation occurs
        self.timeout_penalty = self.cfg.get("timeout_penalty", 0.0)

    # Standalone re-implementation of previous reward shaping terms
    @staticmethod
    def reward_for_distance_to_line(d: float) -> float:
        """
        Mirror AutonomousDrivingEnv.compute_default_reward_for_distance_to_line
        d >= 0: distance magnitude to center line.
        """
        a = 3.0
        b = 1.0
        c = 2.0
        if d < 0.5:
            return a * (1 - d**2)
        elif 0.5 <= d < 2:
            return b * (2 - d) ** 2
        else:
            return -c * (d - 2) ** 3

    @staticmethod
    def reward_for_speed(speed_in_kmph: float) -> float:
        """
        Mirror AutonomousDrivingEnv.compute_default_reward_for_speed.
        """
        a = 1.0 / 12.0
        b = 300.0
        e = 10.0
        if 0 <= speed_in_kmph < 120:
            return -a * (speed_in_kmph - 60) ** 2 + b
        else:
            return -e * (speed_in_kmph - 120)

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        # Base env reward is intentionally zero; we reconstruct externally
        r = 0.0

        # We need values used by the original env reward terms.
        # These are available via env.current_ground_truth and the car state.
        # Access safely; if missing, default to zeros.
        gt = getattr(self.env, "current_ground_truth", {}) or {}
        progress_at_closest_point = float(gt.get("road_progress_at_closest_point", 0.0))
        distance_to_closest_point = float(abs(gt.get("distance_to_closest_point", 0.0)))

        # The env used previous_progress_at_closest_p internally; in the wrapper
        # we track our own previous progress to compute progress_reward.
        if not hasattr(self, "_prev_progress"):
            self._prev_progress = progress_at_closest_point
        progress_reward = progress_at_closest_point - self._prev_progress
        self._prev_progress = progress_at_closest_point

        # Speed shaping used body vx converted to km/h
        vx = float(getattr(getattr(self.env, "car", None), "vx", 0.0))
        speed_kmph = vx * 3.6

        deviation_reward = self.reward_for_distance_to_line(distance_to_closest_point)
        speed_reward = self.reward_for_speed(speed_kmph)

        r += self.k_progress * progress_reward
        r += self.k_deviation * deviation_reward
        r += self.k_speed * speed_reward

        # Termination-based rewards/penalties
        term = info.get("termination", {}) if isinstance(info, dict) else {}
        term_cfg = info.get("termination_rewards", {}) if isinstance(info, dict) else {}
        if terminated:
            if term.get("speed_high"):
                r += float(term_cfg.get("speed_upper_bound", 0.0))
            if term.get("speed_low"):
                r += float(term_cfg.get("speed_lower_bound", 0.0))
            if term.get("off_track"):
                r += float(term_cfg.get("off_track", 0.0))
            if term.get("finished"):
                r += float(self.finished_bonus)

        # Time-limit truncation (Gymnasium's TimeLimit sets this flag in info)
        if truncated and isinstance(info, dict) and info.get("TimeLimit.truncated", False):
            r += float(self.timeout_penalty)

        return obs, r, terminated, truncated, info

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        # Reset wrapper progress tracker
        self._prev_progress = 0.0
        # After reset, try to align with current env ground-truth if available
        gt = getattr(self.env, "current_ground_truth", {}) or {}
        self._prev_progress = float(gt.get("road_progress_at_closest_point", 0.0))
        return res
