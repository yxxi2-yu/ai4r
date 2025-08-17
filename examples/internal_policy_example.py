#!/usr/bin/env python

import os
import numpy as np
import gymnasium
import ai4rgym
import matplotlib.pyplot as plt

from evaluation.evaluation_for_autonomous_driving import (
    simulate_policy,
    plot_road_from_list_of_road_elements,
    plot_results_from_time_series_dict,
)


class NoOpPolicy:
    """A placeholder policy that sends neutral actions.

    Internal policies in the env will override steering/drive when enabled.
    """

    def compute_action(self, observation, info_dict, terminated, truncated):
        # [drive_command, delta_request]
        return np.array([0.0, 0.0], dtype=np.float32)


def save_time_series_csv(path, filename, results):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, filename)

    # Build a simple CSV with key signals for tuning
    headers = [
        "time_index",
        "time_in_seconds",
        "road_progress_at_closest_point",
        "distance_to_closest_point",
        "vx",
        "drive_command",
        "delta_request",
    ]

    # The evaluation helper exposes arrays with length N+1
    ti = np.asarray(results["time_index"]).flatten()
    ts = np.asarray(results["time_in_seconds"]).flatten()
    prog = np.asarray(results["road_progress_at_closest_point"]).flatten()
    dist = np.asarray(results["distance_to_closest_point"]).flatten()
    vx = np.asarray(results["vx"]).flatten()
    dc = np.asarray(results["drive_command"]).flatten()
    dr = np.asarray(results["delta_request"]).flatten()

    data = np.stack([ti, ts, prog, dist, vx, dc, dr], axis=1)

    # Write CSV
    with open(fpath, "w") as f:
        f.write(",".join(headers) + "\n")
        np.savetxt(f, data, delimiter=",", fmt="%.6f")
    print(f"Saved results CSV: {fpath}")


def main():
    # Print working dir (for parity with other examples)
    print("This script is running from:")
    print(os.getcwd())

    # Where to save plots/logs
    save_dir = "examples/saved_figures"
    os.makedirs(save_dir, exist_ok=True)

    # Road: straight + gentle curves
    road_elements_list = [
        {"type": "straight", "length": 120.0},
        {"type": "curved", "curvature": 1 / 800.0, "angle_in_degrees": 20.0},
        {"type": "straight", "length": 200.0},
        {"type": "curved", "curvature": -1 / 600.0, "angle_in_degrees": 30.0},
        {"type": "straight", "length": 120.0},
    ]

    # Plot the road for reference
    plot_road_from_list_of_road_elements(road_elements_list, save_dir, file_name_suffix="internal_policies")

    # Vehicle parameters (inspired by existing examples)
    bicycle_model_parameters = {
        "Lf": 0.55 * 2.875,
        "Lr": 0.45 * 2.875,
        "m": 2000.0,
        "Iz": (1.0 / 12.0) * 2000.0 * (4.692**2 + 1.850**2),
        "Cm": (1.0 / 100.0) * (1.0 * 400.0 * 9.0) / 0.2286,
        "Cd": 0.5 * 0.24 * 2.2204 * 1.202,
        "delta_offset": 0.0,
        "delta_request_max": 45 * np.pi / 180,
        "Ddelta_lower_limit": -90 * np.pi / 180,
        "Ddelta_upper_limit": 90 * np.pi / 180,
        # Use mostly kinematic for simplicity in this demo
        "v_transition_min": 500.0 / 3.6,
        "v_transition_max": 600.0 / 3.6,
        "body_len_f": (0.55 * 2.875) * 1.5,
        "body_len_r": (0.45 * 2.875) * 1.5,
        "body_width": 2.50,
    }

    numerical_integration_parameters = {"method": "rk4", "Ts": 0.05, "num_steps_per_Ts": 1}

    termination_parameters = {
        "speed_lower_bound": (10.0 / 3.6),
        "speed_upper_bound": (200.0 / 3.6),
        "distance_to_closest_point_upper_bound": 20.0,
    }

    # Start near center, near target speed
    initial_state_bounds = {
        "px_init_min": 0.0,
        "px_init_max": 0.0,
        "py_init_min": -0.5,
        "py_init_max": 0.5,
        "theta_init_min": 0.0,
        "theta_init_max": 0.0,
        "vx_init_min": 58.0 / 3.6,
        "vx_init_max": 62.0 / 3.6,
        "vy_init_min": 0.0,
        "vy_init_max": 0.0,
        "omega_init_min": 0.0,
        "omega_init_max": 0.0,
        "delta_init_min": 0.0,
        "delta_init_max": 0.0,
    }

    # Observations – include some signals for potential custom callbacks
    observation_parameters = {
        "should_include_ground_truth_px": "info",
        "should_include_ground_truth_py": "info",
        "should_include_ground_truth_theta": "info",
        "should_include_ground_truth_vx": "obs",
        "should_include_ground_truth_vy": "info",
        "should_include_ground_truth_omega": "info",
        "should_include_ground_truth_delta": "info",
        "should_include_road_progress_at_closest_point": "info",
        "should_include_vx_sensor": "info",
        "should_include_distance_to_closest_point": "obs",
        "should_include_heading_angle_relative_to_line": "obs",
        "should_include_closest_point_coords_in_body_frame": "obs",
        "should_include_road_curvature_at_closest_point": "obs",
        "should_include_look_ahead_road_curvatures": "info",

        # Keep scalings to 1.0 for readability in saved CSV
        "scaling_for_ground_truth_vx": 1.0,
        "scaling_for_distance_to_closest_point": 1.0,
        "scaling_for_heading_angle_relative_to_line": 1.0,
        "scaling_for_road_curvature_at_closest_point": 1.0,
        "scaling_for_closest_point_coords_in_body_frame": 1.0,
    }

    # Internal policy config: enable both, with defaults
    internal_policy_config = {
        "enable_lane_keep": True,
        "enable_cruise_control": True,
        # Adjust target speed and gains as needed
        "cruise_target_speed_mps": 60.0 / 3.6,
        # "lane_keep_k_y": 0.40, "lane_keep_k_psi": 1.20, "lane_keep_k_ff": 0.80,
    }

    # Build environment
    env = gymnasium.make(
        "ai4rgym/autonomous_driving_env",
        render_mode=None,
        bicycle_model_parameters=bicycle_model_parameters,
        road_elements_list=road_elements_list,
        numerical_integration_parameters=numerical_integration_parameters,
        termination_parameters=termination_parameters,
        initial_state_bounds=initial_state_bounds,
        observation_parameters=observation_parameters,
        internal_policy_config=internal_policy_config,
    )

    # Optional: set different surface condition
    env.unwrapped.set_road_condition(road_condition="dry")

    # Run simulation with neutral action policy (let internals drive)
    N_sim = 1500
    results = simulate_policy(env, N_sim, NoOpPolicy(), seed=42, should_save_look_ahead_results=False, should_save_observations=True, verbose=1)

    # Save timeseries to CSV for tuning
    save_time_series_csv(save_dir, "internal_policies_results.csv", results)

    # Quick trajectory/time-series plots
    plot_results_from_time_series_dict(env, results, save_dir, file_name_suffix="internal_policies", should_plot_reward=False)

    # Optional: animation (commented to keep example lightweight)
    # px, py = results["px"], results["py"]
    # theta, delta = results["theta"], results["delta"]
    # ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px, py, theta, delta, numerical_integration_parameters["Ts"], traj_increment=3)
    # ani.save(os.path.join(save_dir, "internal_policies_animation.gif"))

    # Print simple summary metrics
    total_progress = np.nanmax(results["road_progress_at_closest_point"]) - np.nanmin(results["road_progress_at_closest_point"])
    mean_abs_lat = np.nanmean(np.abs(results["distance_to_closest_point"]))
    mean_speed = np.nanmean(results["vx"]) * 3.6
    print(f"Total progress along road: {total_progress:.2f} m")
    print(f"Mean |distance to line|: {mean_abs_lat:.3f} m")
    print(f"Mean speed: {mean_speed:.2f} km/h")


if __name__ == "__main__":
    main()