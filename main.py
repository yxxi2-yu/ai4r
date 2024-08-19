from configs import *

import ai4rgym
import gymnasium as gym
#import numpy as np

from stable_baselines3 import PPO, SAC, DDPG

from utils import ensure_dirs, eval_model, CustomRewardWrapper

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


#env = FlattenActionWrapper(env) # Unnecessary since the action is already flat
env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)  # This might help
#env = CustomObservationWrapper(env) # Unnecessary since we have already filtered the observations
env = CustomRewardWrapper(env)
#env = TerminateTruncateWrapper(env) # Unnecessary since termination and truncation is already built into the environment

from stable_baselines3 import PPO, SAC, DDPG

model_name = "PPO"

TIMESTEPS_PER_EPOCH = 10000
EPOCHS = 10

logdir = "logs"
models_dir = f"models/{model_name}"
figs_dir = f"models/{model_name}/figs"

ensure_dirs([logdir, models_dir, figs_dir])

model = PPO("MultiInputPolicy", env, verbose = 1, tensorboard_log=logdir)

for i in range(1, EPOCHS):
    model.learn(
        total_timesteps=TIMESTEPS_PER_EPOCH,
        reset_num_timesteps=False,
        tb_log_name=f"{model_name}"
    )
    model.save(f"{models_dir}/{TIMESTEPS_PER_EPOCH * i}")
    eval_model(env, model, figs_dir, TIMESTEPS_PER_EPOCH * i)