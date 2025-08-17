import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):
    """
    A custom wrapper that reconstructs the environment's previous reward logic
    (progress, deviation from line, speed shaping, and termination rewards)
    while the base environment returns zero reward. This keeps reward shaping
    out of the env and makes it configurable/replaceable.
    """

    def __init__(self, env: gym.Env, cfg: dict | None = None):
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