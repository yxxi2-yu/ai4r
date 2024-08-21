import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gymnasium as gym

def evaluate_policy(env, model, keys = ['px', 'py', 'theta', 'delta'], max_steps = 5000, return_rewards = False):
    # Reset the environment and get initial observation
    observation, info = env.reset()
    
    # Initialize trajectories with the initial conditions using list comprehension
    trajectories = {key: [info[f'ground_truth_{key}'][0]] for key in keys}
    rewards = []

    # Iterate over simulation steps
    for _ in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Update trajectory lists using list comprehension
        [trajectories[key].append(info[f'ground_truth_{key}'][0]) for key in keys]
        rewards.append(reward)

        if terminated:
            break
    
    # Convert lists to NumPy arrays before returning using list comprehension
    trajectories = {key: np.array(trajectories[key], dtype=np.float32) for key in keys}

    # Return
    if return_rewards:
        return trajectories, rewards
    else:
        return trajectories

def plot_trajectory(env, px, py):
    fig, axs = plt.subplots()
    env.unwrapped.road.render_road(axs)
    line, = axs.plot(px, py, color="#FF0000", label="Trajectory")  # Bright red color
    # Add an invisible line for the road legend
    line_road, = axs.plot([], [], color="black", label="Road")
    axs.set_xlabel('px [meters]')
    axs.set_ylabel('py [meters]')
    axs.grid(True)
    axs.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.suptitle("Road and Trajectory")
    #plt.show()
    return fig

def plot_and_animate_trajectory(env, trajectory, Ts, path_for_saving_figures):
    px_traj, py_traj, theta_traj, delta_traj = \
        trajectory['px'], trajectory['py'], trajectory['theta'], trajectory['delta']

    fig = plot_trajectory(env, px_traj, py_traj)

    # Saving the plot
    fig.savefig(f"{path_for_saving_figures}/trajectory_plot.pdf")
    print(f'Saved plot at {path_for_saving_figures}/trajectory_plot.pdf')

    # Creating and saving an animation
    ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=3)
    ani.save(f"{path_for_saving_figures}/trajectory_animation.gif")
    print(f'Saved animation at {path_for_saving_figures}/trajectory_animation.gif')
    return ani

def eval_model(env, model, figs_dir, timestep, max_steps = 5000):
    #px_traj, py_raj, rewards = simulate_episode(env, model, max_steps)
    trajectory, rewards = evaluate_policy(env, model, keys = ['px', 'py'], max_steps = max_steps, return_rewards = True)
    fig_trajectory = plot_trajectory(env, trajectory['px'], trajectory['py'])
    fig_rewards = plot_rewards(rewards)
    fig_trajectory.savefig(f"{figs_dir}/{timestep}_traj.png")
    fig_rewards.savefig(f"{figs_dir}/{timestep}_rwd.png")

def eval_episode(env, model, max_steps = 5000):
    print("Evaluating 1 episode for a maximum of {} steps".format(max_steps))
    terminated = False
    truncated = False
    rewards = []
    env.reset()
    for step in tqdm(range(max_steps)):
        if terminated or truncated:
            break
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic = True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
    avg_rewards = sum(rewards)/len(rewards)
    return step, avg_rewards

def plot_rewards(rewards):
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Plot the rewards
    axs.plot(rewards, color='black', linewidth=2)
    # Set the labels:
    axs.set_xlabel('Time step', fontsize=10)
    axs.set_ylabel('Reward', fontsize=10)
    # Add grid lines
    axs.grid(visible=True, which="both", axis="both", linestyle='--')
    # Add an overall figure title
    fig.suptitle("Rewards per time step", fontsize=12)
    return fig

def ensure_dirs(paths):
    for path in paths:
        ensure_dir(path)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def distance_reward(d: float) -> float:
    """
    Calculate the reward based on the distance to the center of the road.

    Parameters:
    d (float): Distance to the center of the road in meters.

    Returns:
    float: Reward value.
    Maximum value of reward: 3, goes to negative for distances > 2
    """
    # Define the coefficients
    a = 3.0
    b = 1.0
    c = 2.0

    # Calculate reward based on distance
    if d < 0.5:     # If distance < 0.5; reward with higher curvature
        return a * (1 - d**2)
    elif 0.5 <= d < 2:  # If 0.5 <= distance < 2; reward with lower curvature
        return b * (2 - d)**2
    else:       # If
        return -c * (d - 2)**3

def speed_reward(speed):
    """
    Custom reward function for an autonomous car to encourage maintaining
    a good speed and discourage overspeeding.

    Parameters:
    speed (float): The speed of the car in kmph.

    Returns:
    float: The reward corresponding to the input speed.
    Maximum reward: 300
    """
    # Coefficients derived from solving the equations
    a = 1/12  # Coefficient for the quadratic function in the first and second segments
    b = 300  # Offset for the quadratic function in the first segment
    e = 10  # Coefficient for the linear function in the third segment

    if 0 <= speed < 120:
        return -a * (speed - 60)**2 + b  # Quadratic function for 0 to 120 kmph
    else:
        return -e * (speed - 120)  # Linear decreasing function for speed above 120 kmph

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        # self.roadloss = RoadLoss(a1 = 0.1, a2 = 1, a3 = 10)

    def reward(self, reward):
        """
        Modify the reward value given by the environment.

        Parameters
        ----------
        reward : float
            The reward returned by the environment.

        Returns
        -------
        modified_reward : float
            The modified reward.
        """
        # info_dict = self.env.road.road_info_at_given_pose_and_progress_queries(
        #     px=self.env.car.px, py=self.env.car.py, theta=self.env.car.theta, progress_queries=self.progress_queries)

        observation, info_dict = self.env.unwrapped._get_observation_and_info_and_update_ground_truth()

        progress_delta = reward                                                 # This is the default reward (current progress - previous progress)

        #modified_reward = progress_delta*1000                                   # Multiply by 1000 so that it's in a range ~300-400 depending on speed
        modified_reward = 0

        # Distance reward
        #modified_reward += distance_reward(info_dict["closest_distance"])*100    # Multiply by 100 so that the maximum reward is in a range ~300
        modified_reward += distance_reward(observation["distance_to_closest_point"])*100

        # Speed limit reward
        modified_reward += speed_reward(info_dict['ground_truth_vx']*3.6)                    # Multiply the vx by 3.6 to convert to kph; no need to multiply the reward because reward max is already ~300

        return modified_reward