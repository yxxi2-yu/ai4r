"""
This is just a demo python script to demonstrate how you can run training
in a script without relying on a notebook. 

This script needs to be run from the root directory of the project.

This script has not been verified. If you run into an unexpected error, 
refer to the notebooks for all details.
"""

import ai4rgym
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO, SAC, DDPG

from utils import ensure_dirs, eval_model
from ai4rgym.envs.road import Road

# -------------------------- ENVIRONMENT SETTINGS ---------------------------- #

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
    "reward_for_speed_lower_bound"  :  0.0,
    "reward_for_speed_upper_bound"  :  0.0,
    "reward_for_distance_to_closest_point_upper_bound"  :  0.0,
}

# INTERNAL POLICY CONFIG
internal_policy_config = {
    "enable_lane_keep": False,
    "enable_cruise_control": False,
    "action_interface": "full",
}


# --------------------------- COMBINED REWARD -------------------------------- #

class CombinedRewardWrapper(gym.Wrapper):
    """
    Reward combining lane keeping (distance to center) and cruise control
    (target speed with heading gating). Ported from notebook defaults.
    """
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


# ------------------- DOMAIN RANDOMIZATION WRAPPER --------------------------- #

class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, road_randomization_params=None):
        super(DomainRandomizationWrapper, self).__init__(env)
        self.road_randomization_params = road_randomization_params or {}

    def generate_random_road_elements_list(self):
        params = self.road_randomization_params
        num_elements_range = params.get('num_elements_range', (2, 5))
        straight_length_range = params.get('straight_length_range', (50.0, 200.0))
        curvature_range = params.get('curvature_range', (-1/500.0, 1/500.0))
        angle_range = params.get('angle_range', (10.0, 60.0))

        import random
        road_elements = []
        num_elements = random.randint(*num_elements_range)
        for _ in range(num_elements):
            element_type = random.choice(['straight', 'curved'])
            if element_type == 'straight':
                length = random.uniform(*straight_length_range)
                road_elements.append({"type": "straight", "length": length})
            else:
                curvature = random.uniform(*curvature_range)
                angle = random.uniform(*angle_range)
                road_elements.append({
                    "type": "curved",
                    "curvature": curvature,
                    "angle_in_degrees": angle
                })
        return road_elements

    def reset(self, **kwargs):
        # Generate a new random road each reset
        self.unwrapped.road_elements_list = self.generate_random_road_elements_list()
        self.unwrapped.road = Road(epsilon_c=(1/10000), road_elements_list=self.unwrapped.road_elements_list)
        self.unwrapped.total_road_length = self.unwrapped.road.get_total_length()
        self.unwrapped.total_road_length_for_termination = max(
            self.unwrapped.total_road_length - 0.1,
            0.9999 * self.unwrapped.total_road_length,
        )
        return self.env.reset(**kwargs)


# --------------------------- ENV FACTORY ----------------------------------- #

def create_env(road_elements_list):
    env = gym.make(
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
    # Integration and surface defaults
    env.unwrapped.set_integration_method("rk4")
    env.unwrapped.set_integration_Ts(0.05)
    env.unwrapped.set_road_condition(road_condition="wet")
    # Wrap with action scaling and reward shaping
    env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
    return env

# Create training env with domain randomization
road_randomization_params = None  # Use None for default params
env = create_env(road_elements_list)
env = DomainRandomizationWrapper(env, road_randomization_params=road_randomization_params)
env = CombinedRewardWrapper(env, use_recommended=True)

# --------------------- MODEL DEFINITION & SETUP ----------------------------- #

from stable_baselines3 import PPO, SAC, DDPG

model_name = "PPO"

TIMESTEPS_PER_EPOCH = 50000
EPOCHS = 20

logdir = "logs"
models_dir = f"models/{model_name}"
figs_dir = f"models/{model_name}/figs"

ensure_dirs([logdir, models_dir, figs_dir])

model = PPO("MultiInputPolicy", env, verbose = 1, tensorboard_log=logdir)

for i in range(1, EPOCHS+1):
    model.learn(
        total_timesteps=TIMESTEPS_PER_EPOCH,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}"
    )
    model.save(f"{models_dir}/{TIMESTEPS_PER_EPOCH * i}")
    eval_model(env, model, figs_dir, TIMESTEPS_PER_EPOCH * i)
