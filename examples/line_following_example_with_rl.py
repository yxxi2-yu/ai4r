#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium
import ai4rgym
from policies.pid_policy_for_autonomous_driving import PIDPolicyForAutonomousDriving

from stable_baselines3 import PPO, SAC, DDPG


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
    {"type":"straight", "length":1000.0},
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

# SPECIFY THE TRUNCATION PARAMETERS
truncation_parameters = {
    "speed_lower_bound"  :  (10.0/3.6),
    "speed_upper_bound"  :  (200.0/3.6),
    "distance_to_closest_point_upper_bound"  :  20.0,
}

# SPECIFY THE INITIAL STATE DISTRIBUTION
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
    "should_include_road_progress_at_closest_point"        :  "obs",
    "should_include_vx_sensor"                             :  "obs",
    "should_include_distance_to_closest_point"             :  "obs",
    "should_include_heading_angle_relative_to_line"        :  "obs",
    "should_include_heading_angular_rate_gyro"             :  "obs",
    "should_include_accel_in_body_frame_x"                 :  "neither",
    "should_include_accel_in_body_frame_y"                 :  "neither",
    "should_include_closest_point_coords_in_body_frame"    :  "obs",
    "should_include_look_ahead_line_coords_in_body_frame"  :  "obs",
    "should_include_road_curvature_at_closest_point"       :  "obs",
    "should_include_look_ahead_road_curvatures"            :  "obs",
    "should_include_gps_line_coords_in_world_frame"        :  "neither",

    "scaling_for_ground_truth_px"                       :  1/1000.0,
    "scaling_for_ground_truth_py"                       :  1/1000.0,
    "scaling_for_ground_truth_theta"                    :  1.0,
    "scaling_for_ground_truth_vx"                       :  1/100.0,
    "scaling_for_ground_truth_vy"                       :  1/100.0,
    "scaling_for_ground_truth_omega"                    :  1.0,
    "scaling_for_ground_truth_delta"                    :  1.0,
    "scaling_for_road_progress_at_closest_point"        :  1/1000.0,
    "scaling_for_vx_sensor"                             :  1/100.0,
    "scaling_for_distance_to_closest_point"             :  1/100.0,
    "scaling_for_heading_angle_relative_to_line"        :  1.0,
    "scaling_for_heading_angular_rate_gyro"             :  1.0,
    "scaling_for_accel_in_body_frame_x"                 :  1/10.0,
    "scaling_for_accel_in_body_frame_y"                 :  1/10.0,
    "scaling_for_closest_point_coords_in_body_frame"    :  1/100.0,
    "scaling_for_look_ahead_line_coords_in_body_frame"  :  1/100.0,
    "scaling_for_road_curvature_at_closest_point"       :  1.0,
    "scaling_for_look_ahead_road_curvatures"            :  1.0,
    "scaling_for_gps_line_coords_in_world_frame"        :  1/1000.0,

    "vx_sensor_bias"   : 0.0,
    "vx_sensor_stddev" : 0.1,

    "distance_to_closest_point_bias"    :  0.0,
    "distance_to_closest_point_stddev"  :  0.05,

    "heading_angle_relative_to_line_bias"    :  0.0,
    "heading_angle_relative_to_line_stddev"  :  0.01,

    "heading_angular_rate_gyro_bias"    :  0.0,
    "heading_angular_rate_gyro_stddev"  :  0.0,

    "closest_point_coords_in_body_frame_bias"    :  0.0,
    "closest_point_coords_in_body_frame_stddev"  :  0.0,

    "look_ahead_line_coords_in_body_frame_bias"    :  0.0,
    "look_ahead_line_coords_in_body_frame_stddev"  :  0.0,

    "road_curvature_at_closest_point_bias"    :  0.0,
    "road_curvature_at_closest_point_stddev"  :  0.0,

    "look_ahead_road_curvatures_bias"    :  0.0,
    "look_ahead_road_curvatures_stddev"  :  0.0,

    "look_ahead_line_coords_in_body_frame_distance"             :  100.0,
    "look_ahead_line_coords_in_body_frame_num_points"           :  10,
    "look_ahead_line_coords_in_body_frame_stddev_lateral"       :  0.0,
    "look_ahead_line_coords_in_body_frame_stddev_longitudinal"  :  0.0,
}

# INITIALIZE THE AUTONOMUS DRIVING ENVIRONMENT
env = gymnasium.make(
    "ai4rgym/autonomous_driving_env",
    render_mode=None,
    bicycle_model_parameters=bicycle_model_parameters,
    road_elements_list=road_elements_list,
    numerical_integration_parameters=numerical_integration_parameters,
    truncation_parameters=truncation_parameters,
    initial_state_bounds=initial_state_bounds,
    observation_parameters=observation_parameters,
)

# SET THE ROAD CONDITION
env.unwrapped.set_road_condition(road_condition="wet")

# RESET THE GYMNASIUM
# > And display things as a sanity check
# > which also returns the first observation
observation, info_dict = env.reset()

print("\nobservation = ")
print(observation)

print("\nobservation unscaled = ")
print(env.unwrapped.undo_scaling(observation))

print("\ninfo_dict = ")
print(info_dict)

print("\ninfo_dict unscaled = ")
print(env.unwrapped.undo_scaling(info_dict))


print(env.observation_space)
print(env.action_space)


class RewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = reward * 1000.0

        # Implement additional rewards/penalties here

        return observation, reward, terminated, truncated, info
    
env = gymnasium.make(
    "ai4rgym/autonomous_driving_env",
    render_mode=None,
    bicycle_model_parameters=bicycle_model_parameters,
    road_elements_list=road_elements_list,
    numerical_integration_parameters=numerical_integration_parameters,
    truncation_parameters=truncation_parameters,
    initial_state_bounds=initial_state_bounds,
    observation_parameters=observation_parameters,
)

# > Time increment per simulation step (units: seconds)
Ts_sim = numerical_integration_parameters["Ts"]
# Set the road condition
env.unwrapped.set_road_condition(road_condition="wet")

env = RewardWrapper(env)

observation, info = env.reset()
random_action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(random_action)
print(reward)

model = PPO("MultiInputPolicy", env, verbose = 1)
#model = SAC("MultiInputPolicy", env, verbose = 1)
#model = DDPG("MultiInputPolicy", env, verbose = 1)
#model = TD3("MultiInputPolicy", env, verbose = 1)
#model = A2C("MultiInputPolicy", env, verbose = 1)

model.learn(total_timesteps = 100000)

# EVALUATING THE POLICY
def evaluate_policy(env, model, N_sim):
    # Initialize arrays for storing trajectory data
    px_traj = np.empty(N_sim + 1, dtype=np.float32)
    py_traj = np.empty(N_sim + 1, dtype=np.float32)
    theta_traj = np.empty(N_sim + 1, dtype=np.float32)
    delta_traj = np.empty(N_sim + 1, dtype=np.float32)

    # Reset the environment and get initial observation
    observation, info = env.reset()
    px_traj[0] = info["ground_truth_px"][0]
    py_traj[0] = info["ground_truth_py"][0]
    theta_traj[0] = info["ground_truth_theta"][0]
    delta_traj[0] = info["ground_truth_delta"][0]

    run_terminated = False

    # Iterate over simulation steps
    for i_step in range(N_sim):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Update trajectory arrays
        px_traj[i_step + 1] = info["ground_truth_px"][0]
        py_traj[i_step + 1] = info["ground_truth_py"][0]
        theta_traj[i_step + 1] = info["ground_truth_theta"][0]
        delta_traj[i_step + 1] = info["ground_truth_delta"][0]

        if terminated:
            if not run_terminated:
                run_terminated = True
                break

    return px_traj, py_traj, theta_traj, delta_traj

def plot_and_animate_trajectory(env, px_traj, py_traj, theta_traj, delta_traj, Ts, path_for_saving_figures):
    fig, axs = plt.subplots()
    env.unwrapped.road.render_road(axs)
    line, = axs.plot(px_traj, py_traj, label="Trajectory")
    axs.set_xlabel('px [meters]')
    axs.set_ylabel('py [meters]')
    axs.grid(True)
    axs.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.suptitle("Road and Trajectory")
    #plt.show()

    # Saving the plot
    fig.savefig(f"{path_for_saving_figures}/trajectory_plot.pdf")
    print(f'Saved plot at {path_for_saving_figures}/trajectory_plot.pdf')

    # Creating and saving an animation
    ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=3)
    ani.save(f"{path_for_saving_figures}/trajectory_animation.gif")
    print(f'Saved animation at {path_for_saving_figures}/trajectory_animation.gif')
    return ani


path_for_saving_figures = 'examples/saved_figures'

# Evaluating the Policy
N_sim = 1000
px_traj, py_traj, theta_traj, delta_traj = evaluate_policy(env, model, N_sim)

# Plot and animate Trajectory
ani = plot_and_animate_trajectory(
    env, px_traj, py_traj, theta_traj, delta_traj, numerical_integration_parameters["Ts"], path_for_saving_figures)

ani.save(path_for_saving_figures + '/ad_animation.gif')
print('Saved animation: ' + path_for_saving_figures + '/ad_animation.gif')

#from IPython.display import HTML
#HTML(ani.to_jshtml())