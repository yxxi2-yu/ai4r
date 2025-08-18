# Internal Policies

This environment optionally provides simple “internal policies” to help you bootstrap training and iterate faster. These policies can be toggled on/off and can be replaced by your own custom controllers.

The internal policies affect only the action sent to the dynamics model; rewards and observations are unchanged. Defaults are off (backward compatible).

## Enabling Policies

Pass `internal_policy_config` to `AutonomousDrivingEnv`:

```python
internal_policy_config = {
    "enable_lane_keep": True,          # steering helper
    "enable_cruise_control": True,     # longitudinal helper
}
env = AutonomousDrivingEnv(
    render_mode=None,
    bicycle_model_parameters=..., road_elements_list=...,
    numerical_integration_parameters=..., termination_parameters=...,
    initial_state_bounds=..., observation_parameters=...,
    internal_policy_config=internal_policy_config,
)
```

You can also toggle at runtime:

```python
env.set_lane_keep_enabled(True)
env.set_cruise_control_enabled(True)
```

## Custom Controllers (Function Handles)

You may provide your own controllers. These functions are called inside `env.step()` before integration and receive the current observation dictionary (what an agent sees), not the ground truth.

- Lane keep function: `lane_keep_fn(env, obs_dict) -> float` returns steering angle (radians).
- Cruise function: `cruise_control_fn(env, obs_dict) -> float` returns drive command percentage in `[-100, 100]`.

Example:

```python
def my_lane_keep(env, obs):
    # Use whatever fields you exposed in observation_parameters
    psi = float(obs["heading_angle_relative_to_line"][0])
    kappa = float(obs["road_curvature_at_closest_point"][0])
    # If you included closest point coords in body frame:
    y = float(obs["closest_point_coords_in_body_frame"][1])  # lateral error
    L = env.car.Lf + env.car.Lr
    delta_ff = np.arctan(L * kappa)
    delta = 0.3*y + 1.0*psi + 1.0*delta_ff
    return float(np.clip(delta, -env.car.delta_request_max, env.car.delta_request_max))

def my_cruise(env, obs):
    # Use ground-truth vx if you exposed it, or your own sensor
    vx = float(obs.get("ground_truth_vx", obs.get("vx_sensor"))[0])
    target = 50.0/3.6
    e = target - vx
    return float(np.clip(25.0*e, -100.0, 100.0))

env.set_lane_keep_fn(my_lane_keep)
env.set_cruise_control_fn(my_cruise)
```

Note: If you did not include a needed signal in the observation space, your custom function won’t find it; either expose it via `observation_parameters` or adapt your controller.

## Built-in Controllers (Defaults)

If you enable a policy without providing a function, a simple default is used:

- Lane keeping: `delta = k_y*y + k_psi*psi + k_ff*atan((Lf+Lr)*kappa)`
  - Gains (override via `internal_policy_config`):
    - `lane_keep_k_y` (default 0.40)
    - `lane_keep_k_psi` (default 1.20)
    - `lane_keep_k_ff` (default 0.80)

- Cruise control: PI(D) on longitudinal speed
  - Target: either a fixed `cruise_target_speed_mps` (default 60 km/h) or the
    road’s `recommended_speed_at_closest_point` if you set
    `cruise_use_recommended_speed=True`.
  - Gains: `cruise_kp=20.0`, `cruise_ki=5.0`, `cruise_kd=0.0`
  - Integral clamp: `cruise_integral_limit=50.0`

### Cruise Target Options

You can switch the default cruise target from a fixed value to the road-recommended target derived from curvature and the segment’s speed limit:

```python
internal_policy_config = {
    "enable_cruise_control": True,
    # Use the road’s recommended speed at the closest point (m/s)
    "cruise_use_recommended_speed": True,
    # Fallback fixed target if recommended is unavailable
    "cruise_target_speed_mps": 60.0/3.6,
}
```

When enabled, the built-in controller uses the latest
`recommended_speed_at_closest_point` produced by the road interface. If that value
is missing or non-finite, it falls back to `cruise_target_speed_mps`.

## Tuning Guidance

- Lane keeping
  - Increase `k_psi` if under-steering on curves; reduce if oscillatory.
  - Increase `k_y` to reduce lateral offset; lower if oscillations appear.
  - `k_ff` near 1.0 helps steady curves; reduce if it over-steers.
  - At higher speeds, lower `k_psi` and `k_y` by 10–30%.

- Cruise
  - Start with `kp` 10–25, `ki` 2–8. Reduce `ki` if overshoot.
  - Add small `kd` (1–5) if acceleration jitter is noticeable.
  - If clamped at ±100 often, reduce gains or revisit `Cm`/`Cd` scale.

## Notes

- Custom controllers consume the current observation snapshot created at the start of `step()`; the observation returned by `step()` is at the next state and may include new measurement noise.
- Internal defaults can use ground truth internally (for simplicity), but your custom functions are fed the agent’s observation dictionary.
