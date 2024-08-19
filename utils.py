import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gymnasium as gym

def eval_episode(env, model, max_steps = 3500):
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

def simulate_episode(env, model, max_steps = 5000):
    obs, info = env.reset()

    px_traj = []
    py_traj = []
    rewards = []

    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        px_traj.append(info['ground_truth_px'])
        py_traj.append(info['ground_truth_py'])
        rewards.append(reward)

        if terminated or truncated:
            break

    return px_traj, py_traj, rewards

def plot_trajectory(env, px_traj, py_traj):
    # Open the figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Render the road onto the axis
    env.unwrapped.road.render_road(axs)

    # Initialize a list for the legend
    legend_lines = []

    # trajectory = axs.plot(px_traj, py_traj, color='red', linewidth=2, label='Trajectory')
    trajectory, = axs.plot(px_traj, py_traj, color = 'black')
    trajectory.set_label('Trajectory')
    legend_lines.append(trajectory)

    # Set the labels:
    axs.set_xlabel('px [meters]', fontsize=10)
    axs.set_ylabel('py [meters]', fontsize=10)

    # Add grid lines
    axs.grid(visible=True, which="both", axis="both", linestyle='--')

    # Set the aspect ratio for equally scaled axes
    axs.set_aspect('equal', adjustable='box')

    # Show a legend
    fig.legend(handles=legend_lines, loc="lower center", ncol=4, labelspacing=0.1)

    # Add an overall figure title
    fig.suptitle("Showing the road and the (px,py) trajectory", fontsize=12)

    return fig

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

    #axs.set_ylim(0, 1)  # Set the y-axis limits

    return fig

def ensure_dirs(paths):
    for path in paths:
        ensure_dir(path)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval_model(env, model, figs_dir, timestep, max_steps = 5000):
    px_traj, py_raj, rewards = simulate_episode(env, model, max_steps)
    fig_trajectory = plot_trajectory(env, px_traj, py_raj)
    fig_rewards = plot_rewards(rewards)
    fig_trajectory.savefig(f"{figs_dir}/{timestep}_traj.png")
    fig_rewards.savefig(f"{figs_dir}/{timestep}_rwd.png")

class CustomObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Initialization function,
        initializes a new observation space as a dictionary of boxes

        New Observations
        ----------------

            - distance_to_center: Closest distance of the car from the center
                of the road. This is also multiplied by the side_of_the_road
                to identify if the car is to the left/right side of the center
                of the road
            - angle_diff: The difference in angle between the car's heading
                and the tangent of the road at the closest point
            - curvature_at_closest_p: The curvature of the road at the closest
                point.

        Adding observations:
        --------------------
        If you chose to add more observations, create a `gym.spaces.Box`
            corresponding to your observation with the `low` and `high` values

        """

        super(CustomObservationWrapper, self).__init__(env)

        self.DISTANCE_SCALE = 1
        self.ANGLE_SCALE = 1
        self.CURVATURE_SCALE = 1

        # Update the observation space to include new observations
        # You need to remove the spaces that are not returned in the new observation and add the new ones.
        self.observation_space = gym.spaces.Dict({
            'distance_to_road': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'angle_diff': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'curvature_at_closest_p': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        })

        # Other observations that you might consider including but not limited to:
        # - The progress queries from self.progress_queries
        #   Which will allow the model to look ahead
        # - Current progress (find it from the info_dict)
        # - Velocity/Angular velocity of the car (self.env.car.vx/vy/omega)
        # - Position/Orientation in real-world co-ordinates (self.env.car.px/py/theta)
        # - Current steering anlge (self.env.car.delta)
        # - Other observations from the info_dict and/or custom observations

        # Note that more observations will add to the complexity of the model

    def observation(self, observation):
        """
        This is the method that receives the original observation from the
        default environment and returns a modified dictionary of observations

        Returns:
        --------
            - Dictionary of new observation

        Adding observations:
        --------------------
            - Compute the observation from the car/info_dict
            - Add a corresponding np.array() to the `new_observation` dict

        """

        # This method receives the original observation, and your task is to return the modified observation
        #info_dict = self.env.road.road_info_at_given_pose_and_progress_queries(
        #    px=self.env.car.px, py=self.env.car.py, theta=self.env.car.theta, progress_queries=self.progress_queries)

        #distance_to_road = info_dict['closest_distance'] * info_dict['side_of_the_road_line'] / self.DISTANCE_SCALE
        #angle_diff = (info_dict['road_angle_at_closest_p'] - self.env.car.theta) / self.ANGLE_SCALE
        #curvature_at_closest_p = self.env.road.get_c()[info_dict['closest_element_idx']] / self.CURVATURE_SCALE

        distance_to_road = observation['distance_to_closest_point'] / self.DISTANCE_SCALE
        angle_diff = observation['heading_angle_relative_to_line'] / self.ANGLE_SCALE
        curvature_at_closest_p = observation['road_curvature_at_closest_point'] / self.CURVATURE_SCALE

        new_observation = {
            'distance_to_road': np.array([distance_to_road], dtype=np.float32),
            'angle_diff': np.array([angle_diff], dtype=np.float32),
            'curvature_at_closest_p': np.array([curvature_at_closest_p], dtype=np.float32),
        }

        return new_observation

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

    # Reward function definition
    if 0 <= speed < 60:
        return -a * (speed - 60)**2 + b  # Upward facing quadratic function for 0 to 60 kmph
    elif 60 <= speed < 120:
        return -a * (speed - 60)**2 + b  # Downward facing quadratic function for 60 to 120 kmph
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