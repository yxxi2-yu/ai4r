import os
import matplotlib.pyplot as plt
from tqdm import tqdm

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

        px_traj.append(info['px'])
        py_traj.append(info['py'])
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
