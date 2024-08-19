import ai4rgym
import gymnasium as gym
from stable_baselines3 import PPO
from configs import *
from utils import *
import os

# model path
model_path = 'models/PPO_nr/100000.zip'

print("Setting up environment ..")
env = gym.make(
    "ai4rgym/autonomous_driving_env",
    render_mode=None,
    bicycle_model_parameters=bicycle_model_parameters,
    road_elements_list=road_elements_list,
    numerical_integration_parameters=numerical_integration_parameters,
    truncation_parameters=truncation_parameters, # Added truncation params
    initial_state_bounds=initial_state_bounds,
    observation_parameters=observation_parameters, # Added observation params
)

print("Applying environment wrappers ..")
env = gym.wrappers.RescaleAction(env, min_action = -1, max_action = 1)
env = CustomRewardWrapper(env)

print("Loading saved model ..")
model_name = "PPO_nr"

model = PPO.load(model_path, env = env)

N_sim = 1000
print("Evaluating model .. ")
px_traj, py_traj, theta_traj, delta_traj = evaluate_policy(
    env, model, N_sim)

Ts = numerical_integration_parameters["Ts"]
path_for_saving_figures = f'models/{model_name}/figs'
ensure_dir(path_for_saving_figures)

print("Plotting and animating trajectory .. ")
ani = plot_and_animate_trajectory(
    env, px_traj, py_traj, theta_traj, delta_traj, Ts, path_for_saving_figures
)