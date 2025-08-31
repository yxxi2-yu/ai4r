# CustomRewardWrapper Documentation

> Note: This is a previous version of the docs and may not reflect some of the most recent changes.

The `CustomRewardWrapper` class reconstructs the original reward shaping used by the autonomous driving environment outside of the base environment implementation. This allows reward logic to be configured, swapped, or tested independently from the environment dynamics.

## Key Behavior

- Reconstructs reward as a weighted sum of:
  - **Progress**
  - **Deviation-from-center-line**
  - **Speed shaping** (quadratic around the road’s recommended speed, in m/s)
  - **Termination-related bonuses/penalties**
- Uses defaults intended to mirror the previous environment:
  - `k_progress=0.0`
  - `k_deviation=100.0`
  - `k_speed=1.0`
- Adds optional:
  - **Finished bonus**
  - **Timeout penalty** (for TimeLimit truncations)
- Tracks previous progress internally to compute incremental progress reward.
- Reads optional structures from `info` dictionaries:
  - `termination` (dict of boolean flags like `off_track`, `finished`, `speed_high`, `speed_low`)
  - `termination_rewards` (numeric values for termination-related rewards)
- Applies `timeout_penalty` if `info.get("TimeLimit.truncated")` is `True`.

## Expected Environment Interface

The wrapped environment should expose `current_ground_truth` (a dictionary) containing at least:

- `road_progress_at_closest_point`: Progress along the road at the closest reference point (float-like).
- `distance_to_closest_point`: Distance magnitude to the center line (float-like).

The wrapper reads the vehicle's longitudinal velocity from `env.car.vx` (m/s) and the road’s recommended speed from `env.current_ground_truth['recommended_speed_at_closest_point']` (m/s). If these attributes are missing, the wrapper defaults to zeros.

## Configuration (`cfg` Keys)

- `k_progress` (float): Multiplier for progress increment (default: `0.0`).
- `k_deviation` (float): Multiplier for deviation-from-line reward (default: `100.0`).
- `k_speed` (float): Multiplier for speed-shaped reward (default: `1.0`). The speed term is a quadratic centered at the road’s recommended speed (m/s).
- `finished_bonus` (float): Additional reward added when `termination['finished']` is `True` (default: `0.0`).
- `timeout_penalty` (float): Penalty applied when the episode is truncated by a TimeLimit (default: `0.0`).

## Usage Examples

### Wrap an Environment with Default Shaping
```python
env = AutonomousDrivingEnv(...)  # Environment that exposes expected attributes
env = CustomRewardWrapper(env)
```

### Customize Coefficients
```python
cfg = {
    'k_deviation': 50.0,
    'k_speed': 0.5,
    'finished_bonus': 100.0
}
env = CustomRewardWrapper(env, cfg=cfg)
```

### Step/Return Semantics
Typical step/return semantics remain:
```python
step(action) -> (obs, reward, terminated, truncated, info)
```
The wrapper returns the reconstructed scalar reward as the second element.

## Speed Shaping Details

- The speed component uses m/s throughout and rewards speeds close to the road’s recommended speed at the current position.
- The shape is quadratic around the recommended speed: `-a * (v - v_rec)^2 + b` with `a ≈ 0.006` and `b = 300` by default.
- Exceeding the posted speed limit is not penalized here by default; you can add an additional penalty term in your own logic if desired.

## Notes and Tips

- To disable any shaped term, set its coefficient to `0` in `cfg`.
- The wrapper assumes the base environment's own reward is not needed. If your environment provides a meaningful reward, wrapping will add the shaped reward on top of it (the current implementation assumes the base environment reward is zero but does not explicitly subtract it).
- The wrapper tracks `_prev_progress` across steps and resets it on `reset()`.
