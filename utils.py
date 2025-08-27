import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gymnasium as gym

def evaluate_policy(env, model, keys = ['px', 'py', 'theta', 'delta'], max_steps = 5000, return_rewards = False, return_extras: bool = False):
    # Helper: robustly fetch a scalar for a given key; raise if missing
    def _extract_value(key: str, obs: dict, info: dict, step_idx: int):
        # 1) Try ground truth channels for canonical state keys
        gt_key = f"ground_truth_{key}"
        if gt_key in info:
            val = info[gt_key]
            try:
                return float(val[0])
            except Exception:
                return float(val)
        if gt_key in obs:
            val = obs[gt_key]
            try:
                return float(val[0])
            except Exception:
                return float(val)

        # 2) Try direct key in info/obs (for non-GT items like distance_to_closest_point)
        if key in info:
            val = info[key]
            try:
                return float(val[0])
            except Exception:
                return float(val)
        if key in obs:
            val = obs[key]
            try:
                return float(val[0])
            except Exception:
                return float(val)

        # 3) Fallback: environment's ground truth dictionary
        gt = env.unwrapped.get_current_ground_truth()
        if key in gt:
            return float(gt[key])

        # 4) If not found, raise with a helpful message
        info_keys = list(info.keys())
        obs_keys = list(obs.keys())
        gt_keys = list(gt.keys()) if isinstance(gt, dict) else []
        raise KeyError(
            f"Key '{key}' not found at step {step_idx}. Looked for 'ground_truth_{key}' in info/obs, then '{key}' in info/obs, then env.unwrapped.get_current_ground_truth().\n"
            f"Available info keys: {info_keys}\nAvailable obs keys: {obs_keys}\nAvailable ground truth keys: {gt_keys}"
        )

    # Reset the environment and get initial observation
    observation, info = env.reset()

    # Initialize trajectories with the initial conditions
    trajectories = {key: [_extract_value(key, observation, info, step_idx=0)] for key in keys}
    rewards = []

    # Optional extras we may want to track (kept separate to avoid breaking callers)
    extras = {}
    if return_extras:
        # Seed extras with initial recommended speed from ground truth (m/s)
        gt0 = env.unwrapped.get_current_ground_truth()
        extras["recommended_speed"] = [float(gt0.get('recommended_speed_at_closest_point', 0.0))]

    # Iterate over simulation steps
    for t in range(max_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)

        # Update trajectory lists for each requested key
        for key in keys:
            trajectories[key].append(_extract_value(key, observation, info, step_idx=t+1))

        rewards.append(reward)
        if return_extras:
            gt = env.unwrapped.get_current_ground_truth()
            extras.setdefault("recommended_speed", []).append(float(gt.get('recommended_speed_at_closest_point', 0.0)))

        if terminated:
            break

    # Convert lists to NumPy arrays before returning
    trajectories = {key: np.array(trajectories[key], dtype=np.float32) for key in keys}

    # Convert extras lists to numpy arrays (if any)
    if return_extras:
        for k, v in list(extras.items()):
            extras[k] = np.array(v, dtype=np.float32)

    # Return (preserve old behavior unless extras explicitly requested)
    if return_rewards and return_extras:
        return trajectories, rewards, extras
    if return_rewards:
        return trajectories, rewards
    if return_extras:
        return trajectories, extras
    return trajectories

def plot_and_save_trajectory(env, px, py, figs_dir: str, timestep):
    fig, axs = plt.subplots()
    env.unwrapped.road.render_road(axs)
    axs.plot(px, py, color="#FF0000", label="Trajectory")
    # Add an invisible line for the road legend
    axs.plot([], [], color="black", label="Road")
    axs.set_xlabel('px [meters]')
    axs.set_ylabel('py [meters]')
    axs.grid(True)
    axs.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.suptitle(f"Road and Trajectory ({timestep})")
    outfile = f"{figs_dir}/{timestep}_traj.png"
    fig.savefig(outfile)
    plt.close(fig)
    return outfile

def plot_and_animate_trajectory(env, trajectory, Ts, path_for_saving_figures):
    px_traj, py_traj, theta_traj, delta_traj = \
        trajectory['px'], trajectory['py'], trajectory['theta'], trajectory['delta']

    # Create a figure and plot road + trajectory
    fig, axs = plt.subplots()
    env.unwrapped.road.render_road(axs)
    axs.plot(px_traj, py_traj, color="#FF0000", label="Trajectory")
    axs.plot([], [], color="black", label="Road")
    axs.set_xlabel('px [meters]')
    axs.set_ylabel('py [meters]')
    axs.grid(True)
    axs.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.suptitle("Road and Trajectory")
    pdf_path = f"{path_for_saving_figures}/trajectory_plot.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f'Saved plot at {pdf_path}')

    # Creating and saving an animation
    ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=3)
    ani.save(f"{path_for_saving_figures}/trajectory_animation.gif")
    print(f'Saved animation at {path_for_saving_figures}/trajectory_animation.gif')
    return ani

def eval_model(
    env,
    model,
    figs_dir,
    timestep,
    max_steps = 5000,
    plot_rewards = True,
    plot_trajectory = False,
    plot_distance_to_line: bool = False,
    plot_velocity: bool = False,  
    plot_speed_error: bool = False,
    speed_units: str = "kmh",
):
    # Build keys to request from evaluate_policy in a DRY way
    keys = ['px', 'py']
    if plot_velocity or plot_speed_error:
        keys.append('vx')
    if plot_distance_to_line:
        # Pull distance to line from ground truth
        keys.append('distance_to_closest_point')

    # Need extras whenever we need recommended speed (velocity or speed error plots)
    return_extras = bool(plot_velocity or plot_speed_error)

    # Call evaluate_policy once with the composed keys
    if return_extras:
        trajectory, rewards, extras = evaluate_policy(
            env,
            model,
            keys=keys,
            max_steps=max_steps,
            return_rewards=True,
            return_extras=True,
        )
    else:
        trajectory, rewards = evaluate_policy(
            env,
            model,
            keys=keys,
            max_steps=max_steps,
            return_rewards=True,
            return_extras=False,
        )

    # Reward plot
    if plot_rewards:
        plot_and_save_rewards(rewards, figs_dir, timestep)

    # Trajectory plot
    if plot_trajectory:
        plot_and_save_trajectory(env, trajectory['px'], trajectory['py'], figs_dir, timestep)

    # Optional speed comparison plot
    if plot_velocity:
        vx_series = trajectory.get('vx', None)
        rec_series = extras.get('recommended_speed', None)
        if vx_series is not None and rec_series is not None and len(vx_series) == len(rec_series):
            plot_and_save_speed_vs_recommended(vx_series, rec_series, figs_dir, timestep, units=speed_units)
        else:
            print("[WARN] Could not plot speeds: mismatch or missing data.")

    # Optional speed error plot: (car.vx - recommended_speed)
    if plot_speed_error:
        vx_series = trajectory.get('vx', None)
        rec_series = extras.get('recommended_speed', None)
        if vx_series is not None and rec_series is not None and len(vx_series) == len(rec_series):
            plot_and_save_speed_error(vx_series, rec_series, figs_dir, timestep, units=speed_units)
        else:
            print("[WARN] Could not plot speed error: mismatch or missing data.")

    # Optional distance-to-line plot
    if plot_distance_to_line:
        dist_series = trajectory.get('distance_to_closest_point', None)
        if dist_series is not None:
            plot_and_save_distance_to_line(dist_series, figs_dir, timestep)
        else:
            print("[WARN] Could not plot distance: series not available.")

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
    """Return a matplotlib Figure that plots rewards per timestep.
    This maintains backward compatibility for notebooks importing plot_rewards.
    """
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})
    axs.plot(rewards, color='black', linewidth=2)
    axs.set_xlabel('Time step', fontsize=10)
    axs.set_ylabel('Reward', fontsize=10)
    axs.grid(visible=True, which="both", axis="both", linestyle='--')
    fig.suptitle("Rewards per time step", fontsize=12)
    return fig

def plot_trajectory(env, px, py):
    """Return a matplotlib Figure that plots the road and a trajectory.
    Backward-compatible shim for notebooks that import plot_trajectory.
    """
    fig, axs = plt.subplots()
    env.unwrapped.road.render_road(axs)
    axs.plot(px, py, color="#FF0000", label="Trajectory")
    axs.plot([], [], color="black", label="Road")  # legend handle
    axs.set_xlabel('px [meters]')
    axs.set_ylabel('py [meters]')
    axs.grid(True)
    axs.set_aspect('equal', adjustable='box')
    fig.legend()
    fig.suptitle("Road and Trajectory")
    return fig

def plot_and_save_rewards(rewards, figs_dir: str, timestep):
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})
    axs.plot(rewards, color='black', linewidth=2)
    axs.set_xlabel('Time step', fontsize=10)
    axs.set_ylabel('Reward', fontsize=10)
    axs.grid(visible=True, which="both", axis="both", linestyle='--')
    fig.suptitle(f"Rewards per time step ({timestep})", fontsize=12)
    outfile = f"{figs_dir}/{timestep}_rwd.png"
    fig.savefig(outfile)
    plt.close(fig)
    return outfile

def plot_and_save_speed_vs_recommended(vx_series_mps: np.ndarray, rec_series_mps: np.ndarray, figs_dir: str, timestep, units: str = "kmh"):
    # Convert units if requested
    if units.lower() in ["kmh", "km/h", "kph"]:
        scale = 3.6
        ylabel = "Speed [km/h]"
    else:
        scale = 1.0
        ylabel = "Speed [m/s]"

    vx_plot = np.asarray(vx_series_mps, dtype=np.float32) * scale
    rec_plot = np.asarray(rec_series_mps, dtype=np.float32) * scale

    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})
    ax.plot(vx_plot, color='tab:blue', linewidth=2, label='Car speed')
    ax.plot(rec_plot, color='tab:orange', linewidth=2, linestyle='--', label='Recommended speed')
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(visible=True, which="both", axis="both", linestyle='--')
    ax.legend()
    fig.suptitle(f"Speed vs Recommended ({timestep})", fontsize=12)
    outfile = f"{figs_dir}/{timestep}_spd.png"
    fig.savefig(outfile)
    plt.close(fig)
    return outfile

def plot_and_save_distance_to_line(distance_series_m: np.ndarray, figs_dir: str, timestep):
    d = np.asarray(distance_series_m, dtype=np.float32)
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})
    ax.plot(d, color='tab:green', linewidth=2, label='Distance to line')
    ax.axhline(0.0, color='gray', linewidth=1, linestyle=':', label='Centerline')
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('Distance to line [m]', fontsize=10)
    ax.grid(visible=True, which="both", axis="both", linestyle='--')
    ax.legend()
    fig.suptitle(f"Distance to Line ({timestep})", fontsize=12)
    outfile = f"{figs_dir}/{timestep}_dist.png"
    fig.savefig(outfile)
    plt.close(fig)
    return outfile

def plot_and_save_speed_error(vx_series_mps: np.ndarray, rec_series_mps: np.ndarray, figs_dir: str, timestep, units: str = "kmh"):
    """
    Plot and save the speed error: (car.vx - recommended_speed) over time.
    Positive means car is faster than recommended; negative means slower.
    """
    vx = np.asarray(vx_series_mps, dtype=np.float32)
    rec = np.asarray(rec_series_mps, dtype=np.float32)
    err = vx - rec  # m/s

    if units.lower() in ["kmh", "km/h", "kph"]:
        scale = 3.6
        ylabel = "Speed error [km/h]"
    else:
        scale = 1.0
        ylabel = "Speed error [m/s]"

    err_plot = err * scale

    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})
    ax.plot(err_plot, color='tab:purple', linewidth=2, label='Speed error (car - recommended)')
    ax.axhline(0.0, color='gray', linewidth=1, linestyle=':', label='Zero error')
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(visible=True, which="both", axis="both", linestyle='--')
    ax.legend()
    fig.suptitle(f"Speed Error (vx - recommended) ({timestep})", fontsize=12)
    outfile = f"{figs_dir}/{timestep}_spd_err.png"
    fig.savefig(outfile)
    plt.close(fig)
    return outfile

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
