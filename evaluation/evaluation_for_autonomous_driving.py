#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def simulate_policy(env, N_sim, policy, should_save_look_ahead_results=False):

    # Initialise arrays for storing time series:
    reward_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)

    px_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    py_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    theta_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)
    vx_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    vy_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    omega_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)
    delta_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)

    road_progress_at_closest_point_sim   =  np.full([N_sim+1], np.nan, dtype=np.float32)
    distance_to_closest_point_sim        =  np.full([N_sim+1], np.nan, dtype=np.float32)
    heading_angle_relative_to_line_sim   =  np.full([N_sim+1], np.nan, dtype=np.float32)
    road_curvature_at_closest_point_sim  =  np.full([N_sim+1], np.nan, dtype=np.float32)
    px_closest_sim                       =  np.full([N_sim+1], np.nan, dtype=np.float32)
    py_closest_sim                       =  np.full([N_sim+1], np.nan, dtype=np.float32)
    px_closest_in_body_frame_sim         =  np.full([N_sim+1], np.nan, dtype=np.float32)
    py_closest_in_body_frame_sim         =  np.full([N_sim+1], np.nan, dtype=np.float32)

    if (should_save_look_ahead_results):
        look_ahead_length = env.unwrapped.look_ahead_progress_queries.size
        look_ahead_x_coords_in_body_frame  =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
        look_ahead_y_coords_in_body_frame  =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
        look_ahead_road_curvatures         =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
    else:
        look_ahead_x_coords_in_body_frame  =  np.nan
        look_ahead_y_coords_in_body_frame  =  np.nan
        look_ahead_road_curvatures         =  np.nan
    
    drive_command_sim  =  np.full([N_sim+1], np.nan, dtype=np.float32)
    delta_request_sim  =  np.full([N_sim+1], np.nan, dtype=np.float32)

    # Reset the gymnasium, which also:
    # > Provides the first observation
    # > Updates the "current_ground_truth" dictionary within the environment
    observation, info_dict = env.reset()

    # Get the current ground truth dictionary
    current_ground_truth = env.unwrapped.get_current_ground_truth()

    # Put the initial condition into the first entry of the state trajectory results
    this_time_index = 0
    px_sim[this_time_index]     =  current_ground_truth["px"]
    py_sim[this_time_index]     =  current_ground_truth["py"]
    theta_sim[this_time_index]  =  current_ground_truth["theta"]
    vx_sim[this_time_index]     =  current_ground_truth["vx"]
    vy_sim[this_time_index]     =  current_ground_truth["vy"]
    omega_sim[this_time_index]  =  current_ground_truth["omega"]
    delta_sim[this_time_index]  =  current_ground_truth["delta"]

    road_progress_at_closest_point_sim[this_time_index]   =  current_ground_truth["road_progress_at_closest_point"]
    distance_to_closest_point_sim[this_time_index]        =  current_ground_truth["distance_to_closest_point"]
    heading_angle_relative_to_line_sim[this_time_index]   =  current_ground_truth["heading_angle_relative_to_line"]
    road_curvature_at_closest_point_sim[this_time_index]  =  current_ground_truth["road_curvature_at_closest_point"]
    px_closest_sim[this_time_index]                       =  current_ground_truth["px_closest"]
    py_closest_sim[this_time_index]                       =  current_ground_truth["py_closest"]
    px_closest_in_body_frame_sim[this_time_index]         =  current_ground_truth["px_closest_in_body_frame"]
    py_closest_in_body_frame_sim[this_time_index]         =  current_ground_truth["py_closest_in_body_frame"]

    if (should_save_look_ahead_results):
        look_ahead_x_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,0]
        look_ahead_y_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,1]
        look_ahead_road_curvatures[:,this_time_index]         =  current_ground_truth["look_ahead_road_curvatures"]

    # Display that we are starting this simulation run
    print("\n")
    print("Now starting simulation.")

    # Initialize the flag to when termination occurs
    # > Which corresponds to the car reaching the end of the road
    sim_terminated = False

    # Initialize the flag to when truncation occurs
    # > Which corresponds to any of:
    #   - The car going too fast or too slow
    #   - The car deviating too far from the line
    sim_truncated = False

    # ITERATE OVER THE TIME STEPS OF THE SIMULATION
    for i_step in range(N_sim):
        # Compute the action to apply at this time step
        action = policy.compute_action(observation, info_dict, sim_terminated, sim_truncated)
        # Step forward the gymnasium
        # > This also updates the "current_ground_truth" dictionary within the environment
        observation, reward, terminated, truncated, info_dict = env.step(action)

        # Get the current ground truth dictionary
        current_ground_truth = env.unwrapped.get_current_ground_truth()

        # Store the reward for this time step
        reward_sim[this_time_index] = reward

        # Store the action applied at this time step
        drive_command_sim[this_time_index]  =  action[0]
        delta_request_sim[this_time_index]  =  action[1]

        # Store the states results from the step
        this_time_index = this_time_index+1
        px_sim[this_time_index]     =  current_ground_truth["px"]
        py_sim[this_time_index]     =  current_ground_truth["py"]
        theta_sim[this_time_index]  =  current_ground_truth["theta"]
        vx_sim[this_time_index]     =  current_ground_truth["vx"]
        vy_sim[this_time_index]     =  current_ground_truth["vy"]
        omega_sim[this_time_index]  =  current_ground_truth["omega"]
        delta_sim[this_time_index]  =  current_ground_truth["delta"]

        road_progress_at_closest_point_sim[this_time_index]   =  current_ground_truth["road_progress_at_closest_point"]
        distance_to_closest_point_sim[this_time_index]        =  current_ground_truth["distance_to_closest_point"]
        heading_angle_relative_to_line_sim[this_time_index]   =  current_ground_truth["heading_angle_relative_to_line"]
        road_curvature_at_closest_point_sim[this_time_index]  =  current_ground_truth["road_curvature_at_closest_point"]
        px_closest_sim[this_time_index]                       =  current_ground_truth["px_closest"]
        py_closest_sim[this_time_index]                       =  current_ground_truth["py_closest"]
        px_closest_in_body_frame_sim[this_time_index]         =  current_ground_truth["px_closest_in_body_frame"]
        py_closest_in_body_frame_sim[this_time_index]         =  current_ground_truth["py_closest_in_body_frame"]

        if (should_save_look_ahead_results):
            look_ahead_x_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,0]
            look_ahead_y_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,1]
            look_ahead_road_curvatures[:,this_time_index]         =  current_ground_truth["look_ahead_road_curvatures"]

        # Check whether termination occurred
        if terminated:
            sim_terminated = True
        # Check whether truncation occurred
        if truncated:
            sim_truncated = True
        # End the simulation is either terminated or truncated
        if (terminated or truncated):
            break
    # FINISHED ITERATING OVER THE SIMULATION TIME

    # Display that the simulation is finished
    print("Simulation finished")
    print("\n")

    # Compute the time and time indicies for returning
    time_index_sim = np.arange(start=0, stop=N_sim+1, step=1)
    time_in_seconds_sim = time_index_sim * env.unwrapped.get_integration_Ts()

    # Put all the time series into a dictionary
    sim_time_series_dict = {
        "terminated"  :  sim_terminated,
        "truncated"   :  sim_truncated,

        "time_in_seconds"  :  time_index_sim,
        "time_index"       :  time_in_seconds_sim,

        "reward"  : reward_sim,

        "px"     :  px_sim,
        "py"     :  py_sim,
        "theta"  :  theta_sim,
        "vx"     :  vx_sim,
        "vy"     :  vy_sim,
        "omega"  :  omega_sim,
        "delta"  :  delta_sim,

        "road_progress_at_closest_point"   :  road_progress_at_closest_point_sim,
        "distance_to_closest_point"        :  distance_to_closest_point_sim,
        "heading_angle_relative_to_line"   :  heading_angle_relative_to_line_sim,
        "road_curvature_at_closest_point"  :  road_curvature_at_closest_point_sim,
        "px_closest"                       :  px_closest_sim,
        "py_closest"                       :  py_closest_sim,
        "px_closest_in_body_frame"         :  px_closest_in_body_frame_sim,
        "py_closest_in_body_frame"         :  py_closest_in_body_frame_sim,

        "look_ahead_x_coords_in_body_frame"  :  look_ahead_x_coords_in_body_frame,
        "look_ahead_y_coords_in_body_frame"  :  look_ahead_y_coords_in_body_frame,
        "look_ahead_road_curvatures"         :  look_ahead_road_curvatures,

        "drive_command"  :  drive_command_sim,
        "delta_request"  :  delta_request_sim,
    }

    # Return the results dictionary
    return sim_time_series_dict


def plot_results_from_time_series_dict(env, sim_time_series_results, path_for_saving_figures, file_name_suffix):

    plot_details_list = []

    ## -------------------------------------------------
    #  PLOT THE RESULTS - TRAJECTORY IN CARTESIAN COORDS
    #  -------------------------------------------------

    # Open the figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Render the road onto the axis
    env.unwrapped.road.render_road(axs)

    # Initialize a list for the legend
    legend_lines = []

    # Plot the (px,py) trajectory
    this_line, = axs.plot(sim_time_series_results["px"],sim_time_series_results["py"])
    # > Add the legend entry
    this_line.set_label("trajectory")
    legend_lines.append(this_line)

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

    # Save the plot
    fig.savefig(path_for_saving_figures + '/ad_cartesian_coords.pdf')
    print('Saved figure: ' + path_for_saving_figures + '/ad_cartesian_coords.pdf')

    plot_details_list.append({
        "name"  :  "trajectory in cartesian coordinates",
        "fig"   :  fig,
        "axs"   :  axs,
    })


    ## ------------------------------------------------
    #  PLOT THE RESULTS - TIME SERIES
    #  ------------------------------------------------

    # Open the figure
    fig, axs = plt.subplots(4, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    this_ax_idx = -1

    # Plot the reward time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["reward"])
    # > Add the legend entry
    this_line.set_label("reward")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('reward', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="lower center", ncol=1, labelspacing=0.1)


    # Plot the "distance to line" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["distance_to_closest_point"])
    # > Add the legend entry
    this_line.set_label("dist to line")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('dist [meters]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="lower center", ncol=1, labelspacing=0.1)


    # Plot the "drive command" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["drive_command"])
    # > Add the legend entry
    this_line.set_label("drive command")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('drive [%]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="lower center", ncol=1, labelspacing=0.1)


    # Plot the "drive command" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["delta_request"]*180.0/3.141)
    # > Add the legend entry
    this_line.set_label("delta request")
    legend_lines = []
    legend_lines.append(this_line)
    # Plot the actual delta
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["delta"]*180.0/3.141)
    # > Add the legend entry
    this_line.set_label("actual delta")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('delta [degrees]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="lower center", ncol=1, labelspacing=0.1)


    # Add an overall figure title
    fig.suptitle("Time series results", fontsize=12)

    # Save the plot
    fig.savefig(path_for_saving_figures + '/ad_time_series.pdf')
    print('Saved figure: ' + path_for_saving_figures + '/ad_time_series.pdf')

    plot_details_list.append({
        "name"  :  "time series data",
        "fig"   :  fig,
        "axs"   :  axs,
    })

    return plot_details_list