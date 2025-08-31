"""
This is just a demo python script to demonstrate how you can run evaluation
in a script without relying on a notebook. 

This script needs to be run from the root directory of the project.

This script has not been verified. If you run into an unexpected error, 
refer to the notebooks for all details.
"""

import ai4rgym
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG

from utils import ensure_dir
from policies.rl_policy import RLPolicy
from evaluation.evaluation_for_autonomous_driving import (
    simulate_policy,
    plot_results_from_time_series_dict,
)
import matplotlib.pyplot as plt

# -------------------------- ENVIRONMENT SETTINGS ---------------------------- #

# NOTE: The environment settings should ideally match the original training script

# SPECIFY THE VEHCILE PARAMETERS
bicycle_model_parameters = {
    "Lf" : 0.55*2.875,
    "Lr" : 0.45*2.875,
    "m"  : 2000.0,
    "Iz" : (1.0/12.0) * 2000.0 * (4.692**2+1.850**2),
    "Cm" : (1.0/100.0) * (1.0 * 400.0 * 9.0) / 0.2286,
    "Cd" : 0.5 * 0.24 * 2.2204 * 1.202,
    "delta_offset" : 0 * np.pi/180,
    "delta_request_max" : 45 * np.pi/180,
    "Ddelta_lower_limit" : -45 * np.pi/180,
    "Ddelta_upper_limit" :  45 * np.pi/180,
    "v_transition_min" : 500.0 / 3.6,
    "v_transition_max" : 600.0 / 3.6,
    "body_len_f" : (0.55*2.875) * 1.5,
    "body_len_r" : (0.45*2.875) * 1.5,
    "body_width" : 2.50,
}

# SPECIFY THE ROAD
road_elements_list = [
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":1/800.0, "angle_in_degrees":15.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/400.0, "angle_in_degrees":30.0},
    {"type":"straight", "length":100.0},
]

# SPECIFY THE NUMERICAL INTEGRATION DETAILS
numerical_integration_parameters = {
    "method" : "rk4",
    "Ts" : 0.05,
    "num_steps_per_Ts" : 1,
}

# SPECIFY THE INITIAL STATE DISTRIBUTION

py_init_min = -1.0
py_init_max =  1.0

v_init_min_in_kmh = 55.0
v_init_max_in_kmh = 65.0

py_init_min = -1.0
py_init_max =  1.0

v_init_min_in_kmh = 55.0
v_init_max_in_kmh = 65.0

initial_state_bounds = {
    "px_init_min" : 0.0,
    "px_init_max" : 0.0,
    "py_init_min" : py_init_min,
    "py_init_max" : py_init_max,
    "theta_init_min" : 0.0,
    "theta_init_max" : 0.0,
    "vx_init_min" : v_init_min_in_kmh * (1.0/3.6),
    "vx_init_max" : v_init_max_in_kmh * (1.0/3.6),
    "vy_init_min" : 0.0,
    "vy_init_max" : 0.0,
    "omega_init_min" : 0.0,
    "omega_init_max" : 0.0,
    "delta_init_min" : 0.0,
    "delta_init_max" : 0.0,
}

# SPECIFY THE OBSERVATION PARAMETERS
observation_parameters = {
    "should_include_ground_truth_px"                       :  "info",
    "should_include_ground_truth_py"                       :  "info",
    "should_include_ground_truth_theta"                    :  "info",
    "should_include_ground_truth_vx"                       :  "info",
    "should_include_ground_truth_vy"                       :  "info",
    "should_include_ground_truth_omega"                    :  "info",
    "should_include_ground_truth_delta"                    :  "info",
    "should_include_road_progress_at_closest_point"        :  "info",
    "should_include_vx_sensor"                             :  "info",
    "should_include_distance_to_closest_point"             :  "obs",
    "should_include_heading_angle_relative_to_line"        :  "obs",
    "should_include_heading_angular_rate_gyro"             :  "info",
    "should_include_closest_point_coords_in_body_frame"    :  "info",
    "should_include_look_ahead_line_coords_in_body_frame"  :  "info",
    "should_include_road_curvature_at_closest_point"       :  "obs",
    "should_include_look_ahead_road_curvatures"            :  "info",

    "scaling_for_ground_truth_px"                       :  1.0,
    "scaling_for_ground_truth_py"                       :  1.0,
    "scaling_for_ground_truth_theta"                    :  1.0,
    "scaling_for_ground_truth_vx"                       :  1.0,
    "scaling_for_ground_truth_vy"                       :  1.0,
    "scaling_for_ground_truth_omega"                    :  1.0,
    "scaling_for_ground_truth_delta"                    :  1.0,
    "scaling_for_road_progress_at_closest_point"        :  1.0,
    "scaling_for_vx_sensor"                             :  1.0,
    "scaling_for_distance_to_closest_point"             :  1.0,
    "scaling_for_heading_angle_relative_to_line"        :  1.0,
    "scaling_for_heading_angular_rate_gyro"             :  1.0,
    "scaling_for_closest_point_coords_in_body_frame"    :  1.0,
    "scaling_for_look_ahead_line_coords_in_body_frame"  :  1.0,
    "scaling_for_road_curvature_at_closest_point"       :  1.0,
    "scaling_for_look_ahead_road_curvatures"            :  1.0,

    "vx_sensor_bias"    : 0.0,
    "vx_sensor_stddev"  : 0.1,

    "distance_to_closest_point_bias"    :  0.0,
    "distance_to_closest_point_stddev"  :  0.01,

    "heading_angle_relative_to_line_bias"    :  0.0,
    "heading_angle_relative_to_line_stddev"  :  0.01,

    "heading_angular_rate_gyro_bias"    :  0.0,
    "heading_angular_rate_gyro_stddev"  :  0.01,

    "closest_point_coords_in_body_frame_bias"    :  0.0,
    "closest_point_coords_in_body_frame_stddev"  :  0.0,

    "look_ahead_line_coords_in_body_frame_bias"    :  0.0,
    "look_ahead_line_coords_in_body_frame_stddev"  :  0.0,

    "road_curvature_at_closest_point_bias"    :  0.0,
    "road_curvature_at_closest_point_stddev"  :  0.0,

    "look_ahead_road_curvatures_bias"    :  0.0,
    "look_ahead_road_curvatures_stddev"  :  0.0,

    "look_ahead_line_coords_in_body_frame_distance"    :  100.0,
    "look_ahead_line_coords_in_body_frame_num_points"  :  10,
}

# SPECIFY THE TERMINATION PARAMETERS (updated interface names)
termination_parameters = {
    "speed_lower_bound"  :  0.0,
    "speed_upper_bound"  :  (200.0/3.6),
    "distance_to_closest_point_upper_bound"  :  20.0,
    # Updated keys with "reward_for_*" to match env API
    "reward_for_speed_lower_bound"  :  -1000.0,
    "reward_for_speed_upper_bound"  :  -1000.0,
    "reward_for_distance_to_closest_point_upper_bound"  :  -1000.0,
}

# INTERNAL POLICY CONFIG (defaults from notebook)
internal_policy_config = {
    "enable_lane_keep": False,
    "enable_cruise_control": False,
    "action_interface": "full",
}


# -------------------------- ENVIRONMENT FACTORY ----------------------------- #

class CombinedRewardWrapper(gym.Wrapper):
    def __init__(self, env,
                 k_distance: float = 1.0,
                 k_speed: float = 1.0,
                 target_speed_mps: float = 60.0/3.6,
                 use_recommended: bool = False):
        super().__init__(env)
        self.k_distance = float(k_distance)
        self.k_speed = float(k_speed)
        self.target_speed_mps = float(target_speed_mps)
        self.use_recommended = bool(use_recommended)

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        gt = self.env.unwrapped.get_current_ground_truth()
        v = float(self.env.unwrapped.car.vx)
        d = float(abs(gt["distance_to_closest_point"]))
        v_rec = float(gt.get("recommended_speed_at_closest_point", 0.0))
        theta = float(gt.get("heading_angle_relative_to_line", 0.0))
        r_lane = self.k_distance * (1.0 / (1.0 + d))
        v_tgt = v_rec if (self.use_recommended and v_rec > 0.0) else self.target_speed_mps
        gate = max(0.0, np.cos(theta))
        r_speed = self.k_speed * (1.0 / (1.0 + abs(v - v_tgt)))
        r = r_lane + gate * r_speed
        info = dict(info,
                    lane_distance=d,
                    r_lane=r_lane,
                    heading_rel_line=theta,
                    heading_gate=gate,
                    r_speed=r_speed,
                    v=v,
                    v_tgt=v_tgt)
        return obs, r, terminated, truncated, info


def create_env(road_elements):
    env = gym.make(
        "ai4rgym/autonomous_driving_env",
        render_mode=None,
        bicycle_model_parameters=bicycle_model_parameters,
        road_elements_list=road_elements,
        numerical_integration_parameters=numerical_integration_parameters,
        termination_parameters=termination_parameters,
        initial_state_bounds=initial_state_bounds,
        observation_parameters=observation_parameters,
        internal_policy_config=internal_policy_config,
    )
    env.unwrapped.set_integration_method("rk4")
    env.unwrapped.set_integration_Ts(0.05)
    env.unwrapped.set_road_condition(road_condition="wet")
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    return env

# --------------------- MODEL DEFINITION & SETUP ----------------------------- #

from stable_baselines3 import PPO, SAC, DDPG

# --------------------- MODEL DEFINITION & SETUP ----------------------------- #

model_name = "PPO"
model_idx = 950000

eval_env = create_env(road_elements_list)
env = CombinedRewardWrapper(eval_env, use_recommended=True)

path_for_saving_figures = f"models/{model_name}/eval"
ensure_dir(path_for_saving_figures)

# Load Model
print("Loading saved model ..")
model_path = f"models/{model_name}/{str(model_idx)}.zip"
model = PPO.load(model_path, env=eval_env)

# Wrap into standardized policy interface
rl_policy = RLPolicy(model)

# Simulate policy and collect robust time-series
print("Simulating policy for evaluation ..")
N_sim = 5000
sim_results = simulate_policy(eval_env, N_sim, rl_policy, seed=1, should_save_look_ahead_results=False, should_save_observations=True, verbose=1)

# Plot trajectory and time-series
plot_results_from_time_series_dict(eval_env, sim_results, path_for_saving_figures, file_name_suffix=str(model_idx), should_plot_reward=True)

# Compute performance metrics
def compute_performance_metrics_from_time_series(sim_time_series_dict):
    abs_dist = np.abs(sim_time_series_dict["distance_to_closest_point"])  # signed → abs
    avg_dist = float(np.nanmean(abs_dist))
    std_dist = float(np.nanstd(abs_dist))
    max_dist = float(np.nanmax(abs_dist))

    speed = np.abs(sim_time_series_dict["vx"])  # m/s
    avg_speed = float(np.nanmean(speed) * 3.6)
    std_speed = float(np.nanstd(speed) * 3.6)
    max_speed = float(np.nanmax(speed) * 3.6)
    min_speed = float(np.nanmin(speed) * 3.6)

    return {
        "avg_dist": avg_dist,
        "std_dist": std_dist,
        "max_dist": max_dist,
        "avg_speed": avg_speed,
        "std_speed": std_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,
    }

pm_dict = compute_performance_metrics_from_time_series(sim_results)
print("Performance Metric dictionary:")
print(pm_dict)

# Optional: save animation of trajectory
def animate_from_sim_time_series_dict(env, sim_time_series_dict, Ts, path_for_saving_figures):
    px_traj = sim_time_series_dict["px"]
    py_traj = sim_time_series_dict["py"]
    theta_traj = sim_time_series_dict["theta"]
    delta_traj = sim_time_series_dict["delta"]
    ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=3)
    ani.save(f"{path_for_saving_figures}/trajectory_animation.gif")
    print(f"Saved animation at {path_for_saving_figures}/trajectory_animation.gif")
    return ani

Ts = numerical_integration_parameters["Ts"]
_ = animate_from_sim_time_series_dict(eval_env, sim_results, Ts, path_for_saving_figures)
