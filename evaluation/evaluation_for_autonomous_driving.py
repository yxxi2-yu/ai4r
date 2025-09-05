#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from ai4rgym.envs.road import Road
from policies.rl_policy import RLPolicy



## -----------------
#  SIMULATE A POLICY
#  -----------------
def simulate_policy(env, N_sim, policy, seed=None, should_save_look_ahead_results=False, should_save_observations=False, verbose=0):

    # Initialise arrays for storing time series:
    reward_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)

    px_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    py_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    theta_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)
    vx_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    vy_sim    = np.full([N_sim+1,], np.nan, dtype=np.float32)
    omega_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)
    delta_sim = np.full([N_sim+1,], np.nan, dtype=np.float32)

    road_progress_at_closest_point_sim     =  np.full([N_sim+1], np.nan, dtype=np.float32)
    distance_to_closest_point_sim          =  np.full([N_sim+1], np.nan, dtype=np.float32)
    heading_angle_relative_to_line_sim     =  np.full([N_sim+1], np.nan, dtype=np.float32)
    road_curvature_at_closest_point_sim    =  np.full([N_sim+1], np.nan, dtype=np.float32)
    speed_limit_at_closest_point_sim       =  np.full([N_sim+1], np.nan, dtype=np.float32)
    recommended_speed_at_closest_point_sim =  np.full([N_sim+1], np.nan, dtype=np.float32)

    px_closest_sim                         =  np.full([N_sim+1], np.nan, dtype=np.float32)
    py_closest_sim                         =  np.full([N_sim+1], np.nan, dtype=np.float32)
    px_closest_in_body_frame_sim           =  np.full([N_sim+1], np.nan, dtype=np.float32)
    py_closest_in_body_frame_sim           =  np.full([N_sim+1], np.nan, dtype=np.float32)

    if (should_save_look_ahead_results):
        look_ahead_length = env.unwrapped.look_ahead_progress_queries.size
        look_ahead_x_coords_in_body_frame  =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
        look_ahead_y_coords_in_body_frame  =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
        look_ahead_road_curvatures         =  np.full([look_ahead_length,N_sim+1], np.nan, dtype=np.float32)
    else:
        look_ahead_x_coords_in_body_frame  =  np.nan
        look_ahead_y_coords_in_body_frame  =  np.nan
        look_ahead_road_curvatures         =  np.nan

    if (should_save_observations):
        obs_distance_to_closest_point       =  np.full([N_sim+1], np.nan, dtype=np.float32)
        obs_heading_angle_relative_to_line  =  np.full([N_sim+1], np.nan, dtype=np.float32)
        obs_road_curvature_at_closest_point =  np.full([N_sim+1], np.nan, dtype=np.float32)
    else:
        obs_distance_to_closest_point       =  np.nan
        obs_heading_angle_relative_to_line  =  np.nan
        obs_road_curvature_at_closest_point =  np.nan
    
    drive_command_sim  =  np.full([N_sim+1], np.nan, dtype=np.float32)
    delta_request_sim  =  np.full([N_sim+1], np.nan, dtype=np.float32)

    # Reset the gymnasium, which also:
    # > Provides the first observation
    # > Updates the "current_ground_truth" dictionary within the environment
    observation, info_dict = env.reset(seed=seed)

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

    road_progress_at_closest_point_sim[this_time_index]      =  current_ground_truth["road_progress_at_closest_point"]
    distance_to_closest_point_sim[this_time_index]           =  current_ground_truth["distance_to_closest_point"]
    heading_angle_relative_to_line_sim[this_time_index]      =  current_ground_truth["heading_angle_relative_to_line"]
    road_curvature_at_closest_point_sim[this_time_index]     =  current_ground_truth["road_curvature_at_closest_point"]
    speed_limit_at_closest_point_sim[this_time_index]        =  current_ground_truth["speed_limit_at_closest_point"]
    recommended_speed_at_closest_point_sim[this_time_index]  =  current_ground_truth["recommended_speed_at_closest_point"]

    px_closest_sim[this_time_index]                          =  current_ground_truth["px_closest"]
    py_closest_sim[this_time_index]                          =  current_ground_truth["py_closest"]
    px_closest_in_body_frame_sim[this_time_index]            =  current_ground_truth["px_closest_in_body_frame"]
    py_closest_in_body_frame_sim[this_time_index]            =  current_ground_truth["py_closest_in_body_frame"]

    if (should_save_look_ahead_results):
        look_ahead_x_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,0]
        look_ahead_y_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,1]
        look_ahead_road_curvatures[:,this_time_index]         =  current_ground_truth["look_ahead_road_curvatures"]

    if (should_save_observations):
        obs_distance_to_closest_point[this_time_index]       =  observation.get("distance_to_closest_point", np.nan)
        obs_heading_angle_relative_to_line[this_time_index]  =  observation.get("heading_angle_relative_to_line", np.nan)
        obs_road_curvature_at_closest_point[this_time_index] =  observation.get("road_curvature_at_closest_point", np.nan)

    # Display that we are starting this simulation run
    if (verbose > 0):
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

        # Get the actions from the unwrapped environment to ensure they are not adjusted
        # by an Action Scaling wrapper, or similar.
        this_drive_cmd, this_delta_req = env.unwrapped.car.get_action_requests()

        # Store the action applied at this time step
        drive_command_sim[this_time_index]  =  this_drive_cmd
        delta_request_sim[this_time_index]  =  this_delta_req

        # Store the states results from the step
        this_time_index = this_time_index+1
        px_sim[this_time_index]     =  current_ground_truth["px"]
        py_sim[this_time_index]     =  current_ground_truth["py"]
        theta_sim[this_time_index]  =  current_ground_truth["theta"]
        vx_sim[this_time_index]     =  current_ground_truth["vx"]
        vy_sim[this_time_index]     =  current_ground_truth["vy"]
        omega_sim[this_time_index]  =  current_ground_truth["omega"]
        delta_sim[this_time_index]  =  current_ground_truth["delta"]

        road_progress_at_closest_point_sim[this_time_index]      =  current_ground_truth["road_progress_at_closest_point"]
        distance_to_closest_point_sim[this_time_index]           =  current_ground_truth["distance_to_closest_point"]
        heading_angle_relative_to_line_sim[this_time_index]      =  current_ground_truth["heading_angle_relative_to_line"]
        road_curvature_at_closest_point_sim[this_time_index]     =  current_ground_truth["road_curvature_at_closest_point"]
        speed_limit_at_closest_point_sim[this_time_index]        =  current_ground_truth["speed_limit_at_closest_point"]
        recommended_speed_at_closest_point_sim[this_time_index]  =  current_ground_truth["recommended_speed_at_closest_point"]

        px_closest_sim[this_time_index]                          =  current_ground_truth["px_closest"]
        py_closest_sim[this_time_index]                          =  current_ground_truth["py_closest"]
        px_closest_in_body_frame_sim[this_time_index]            =  current_ground_truth["px_closest_in_body_frame"]
        py_closest_in_body_frame_sim[this_time_index]            =  current_ground_truth["py_closest_in_body_frame"]

        if (should_save_look_ahead_results):
            look_ahead_x_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,0]
            look_ahead_y_coords_in_body_frame[:,this_time_index]  =  current_ground_truth["look_ahead_line_coords_in_body_frame"][:,1]
            look_ahead_road_curvatures[:,this_time_index]         =  current_ground_truth["look_ahead_road_curvatures"]

        if (should_save_observations):
            obs_distance_to_closest_point[this_time_index]       =  observation.get("distance_to_closest_point", np.nan)
            obs_heading_angle_relative_to_line[this_time_index]  =  observation.get("heading_angle_relative_to_line", np.nan)
            obs_road_curvature_at_closest_point[this_time_index] =  observation.get("road_curvature_at_closest_point", np.nan)

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
    if (verbose > 0):
        print("Simulation finished")
        print("\n")

    # Compute the time and time indicies for returning
    time_index_sim = np.arange(start=0, stop=N_sim+1, step=1)
    time_in_seconds_sim = time_index_sim * env.unwrapped.get_integration_Ts()

    # Put all the time series into a dictionary
    sim_time_series_dict = {
        "terminated"  :  sim_terminated,
        "truncated"   :  sim_truncated,

        "time_in_seconds"  :  time_in_seconds_sim,
        "time_index"       :  time_index_sim,

        "reward"  : reward_sim,

        "px"     :  px_sim,
        "py"     :  py_sim,
        "theta"  :  theta_sim,
        "vx"     :  vx_sim,
        "vy"     :  vy_sim,
        "omega"  :  omega_sim,
        "delta"  :  delta_sim,

        "road_progress_at_closest_point"      :  road_progress_at_closest_point_sim,
        "distance_to_closest_point"           :  distance_to_closest_point_sim,
        "heading_angle_relative_to_line"      :  heading_angle_relative_to_line_sim,
        "road_curvature_at_closest_point"     :  road_curvature_at_closest_point_sim,
        "speed_limit_at_closest_point"        :  speed_limit_at_closest_point_sim,
        "recommended_speed_at_closest_point"  :  recommended_speed_at_closest_point_sim,

        "px_closest"                          :  px_closest_sim,
        "py_closest"                          :  py_closest_sim,
        "px_closest_in_body_frame"            :  px_closest_in_body_frame_sim,
        "py_closest_in_body_frame"            :  py_closest_in_body_frame_sim,

        "look_ahead_x_coords_in_body_frame"  :  look_ahead_x_coords_in_body_frame,
        "look_ahead_y_coords_in_body_frame"  :  look_ahead_y_coords_in_body_frame,
        "look_ahead_road_curvatures"         :  look_ahead_road_curvatures,

        "obs_distance_to_closest_point"        :  obs_distance_to_closest_point,
        "obs_heading_angle_relative_to_line"   :  obs_heading_angle_relative_to_line,
        "obs_road_curvature_at_closest_point"  :  obs_road_curvature_at_closest_point,

        "drive_command"  :  drive_command_sim,
        "delta_request"  :  delta_request_sim,
    }

    # Return the results dictionary
    return sim_time_series_dict



## -----------------
#  SIMULATE RL MODEL
#  -----------------
def simulate_rl_model(env, N_sim, rl_model, seed=None, should_save_look_ahead_results=False, should_save_observations=False, verbose=0):
    # Put the RL model into a policy class
    rl_policy = RLPolicy(rl_model)
    # Call the simluate policy function
    sim_time_series_dict = simulate_policy(env, N_sim, rl_policy, seed, should_save_look_ahead_results, should_save_observations, verbose)
    # Return the results dictionary
    return sim_time_series_dict



## -------------
#  PLOT THE ROAD
#  -------------
def plot_road_from_list_of_road_elements(road_elements_list, path_for_saving_figures, file_name_suffix, width_btw_cones=None, mean_length_btw_cones=None, stddev_of_length_btw_cones=None):

    # Initialize the road
    road = Road(road_elements_list=road_elements_list)

    # Check if cones should be plotted
    should_plot_cones = True
    if ((width_btw_cones is None) or (mean_length_btw_cones is None) or (stddev_of_length_btw_cones is None)):
        should_plot_cones = False

    # Call the function to generate cone locations
    if should_plot_cones:
        road.generate_cones(width_btw_cones, mean_length_btw_cones, stddev_of_length_btw_cones)

    # Open the figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Auto-scale the axis limits
    axs.set_xlim(auto=True)
    axs.set_ylim(auto=True)

    # Call the function to render the road
    road_handles = road.render_road(axs)

    # Plot the cones
    if should_plot_cones:
        # > Get the cones coordinates
        cones_left_side_coords  = road.get_cones_left_side()
        cones_right_side_coords = road.get_cones_right_side()
        # > Plot the left-side cones in yellow
        cone_handles_left_side  = axs.scatter(x=cones_left_side_coords[:,0],  y=cones_left_side_coords[:,1],  s=8.0, marker="o", facecolor="y", alpha=1.0, linewidths=0, edgecolors="k")
        # > Plot the right-side cones in blue
        cone_handles_right_side = axs.scatter(x=cones_right_side_coords[:,0], y=cones_right_side_coords[:,1], s=8.0, marker="o", facecolor="b", alpha=1.0, linewidths=0, edgecolors="k")

    # Ensure the aspect ratio stays as 1:1
    axs.set_aspect('equal', adjustable='box')

    # Add a title
    if should_plot_cones:
        fig.suptitle('The road and cones', fontsize=12)
    else:
        fig.suptitle('The road, i.e., the center of the lane to be followed', fontsize=12)
    # Save the figure
    if isinstance(file_name_suffix, str) and file_name_suffix == "":
        path_and_file_name = path_for_saving_figures + "/" + "ad_road" + ".pdf"
    else:
        path_and_file_name = path_for_saving_figures + "/" + "ad_road" + "_" + file_name_suffix + ".pdf"
    fig.savefig(path_and_file_name)
    print("Saved figure: " + path_and_file_name)

    # Put together the details of this plot
    plot_details_list = {
        "name"     :  "road center line",
        "fig"      :  fig,
        "axs"      :  axs,
        "handles"  :  road_handles,
    }

    if should_plot_cones:
        plot_details_list.update({
            "cone_handles_left"   :  cone_handles_left_side,
            "cone_handles_right"  :  cone_handles_right_side,
        })

    # Return the list of plot details
    return plot_details_list



## -------------
#  PLOT THE ROAD
#  -------------
def plot_current_state_zoomed_in(env, path_for_saving_figures, file_name_suffix):

    # Initialize the figure for plotting
    env.unwrapped.render_matplotlib_init_figure()
    # Plot the road
    env.unwrapped.render_matplotlib_plot_road()
    # If the road has cones, then plot the cones
    # > Get the cones coordinates
    cones_left_side_coords  = env.unwrapped.road.get_cones_left_side()
    cones_right_side_coords = env.unwrapped.road.get_cones_right_side()
    # > Plot the left-side cones in yellow
    if (len(cones_left_side_coords)>0):
        cone_handles_left_side  = env.unwrapped.axis.scatter(x=cones_left_side_coords[:,0],  y=cones_left_side_coords[:,1],  s=8.0, marker="o", facecolor="y", alpha=1.0, linewidths=0, edgecolors="k")
    # > Plot the right-side cones in blue
    if (len(cones_right_side_coords)>0):
        cone_handles_right_side = env.unwrapped.axis.scatter(x=cones_right_side_coords[:,0], y=cones_right_side_coords[:,1], s=8.0, marker="o", facecolor="b", alpha=1.0, linewidths=0, edgecolors="k")
    # Zoom into the start position
    env.unwrapped.render_matplotlib_zoom_to(px=env.unwrapped.car.px,py=env.unwrapped.car.py,x_width=20,y_height=20)
    # Add a title
    env.unwrapped.figure.suptitle('Zoom in of the road and car', fontsize=12)
    # Save the figure
    if isinstance(file_name_suffix, str) and file_name_suffix == "":
        path_and_file_name = path_for_saving_figures + "/" + "ad_vehicle" + "_" + ".pdf"
    else:
        path_and_file_name = path_for_saving_figures + "/" + "ad_vehicle" + "_" + file_name_suffix + ".pdf"
    env.unwrapped.figure.savefig(path_and_file_name)
    print("Saved figure: " + path_and_file_name)

    # # Put together the details of this plot
    # plot_details_list = {
    #     "name"     :  "current state zoomed in",
    #     "fig"      :  fig,
    #     "axs"      :  axs,
    #     "handles"  :  road_handles,
    # }

    # # Return the list of plot details
    # return plot_details_list





def plot_results_from_time_series_dict(env, sim_time_series_results, path_for_saving_figures, file_name_suffix, should_plot_reward):

    plot_details_list = []

    ## -------------------------------------------------
    #  PLOT THE RESULTS - TRAJECTORY IN CARTESIAN COORDS
    #  -------------------------------------------------

    # Open the figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.94,"bottom":0.05})

    # Set the figure size
    #fig.set_size_inches(7, 10)

    # Initialize a list for the legend
    legend_lines = []

    # Render the road onto the axis
    road_handles = env.unwrapped.road.render_road(axs)

    # Add a road line to the legend handles
    road_handles[0].set_label("road")
    legend_lines.append(road_handles[0])

    # If the road has cones, then plot the cones
    # > Get the cones coordinates
    cones_left_side_coords  = env.unwrapped.road.get_cones_left_side()
    cones_right_side_coords = env.unwrapped.road.get_cones_right_side()
    # > Plot the left-side cones in yellow
    if (len(cones_left_side_coords)>0):
        cone_handles_left_side  = axs.scatter(x=cones_left_side_coords[:,0],  y=cones_left_side_coords[:,1],  s=8.0, marker="o", facecolor="y", alpha=1.0, linewidths=0, edgecolors="k")
    # > Plot the right-side cones in blue
    if (len(cones_right_side_coords)>0):
        cone_handles_right_side = axs.scatter(x=cones_right_side_coords[:,0], y=cones_right_side_coords[:,1], s=8.0, marker="o", facecolor="b", alpha=1.0, linewidths=0, edgecolors="k")

    # Plot the (px,py) trajectory
    this_line, = axs.plot(sim_time_series_results["px"],sim_time_series_results["py"], color="red")
    # > Add the legend entry
    this_line.set_label("trajectory of vehicle")
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
    if isinstance(file_name_suffix, str) and file_name_suffix == "":
        path_and_file_name = path_for_saving_figures + "/" + "ad_cartesian_coords" + ".pdf"
    else:
        path_and_file_name = path_for_saving_figures + "/" + "ad_cartesian_coords" + "_" + file_name_suffix + ".pdf"
    fig.savefig(path_and_file_name)
    print('Saved figure: ' + path_and_file_name)

    plot_details_list.append({
        "name"  :  "trajectory in cartesian coordinates",
        "fig"   :  fig,
        "axs"   :  axs,
    })


    ## ------------------------------------------------
    #  PLOT THE RESULTS - TIME SERIES
    #  ------------------------------------------------

    if (should_plot_reward):
        num_of_subplots = 7
    else:
        num_of_subplots = 5

    # Open the figure
    fig, axs = plt.subplots(num_of_subplots, 1, sharex=True, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.94,"bottom":0.05})

    if (should_plot_reward):
        fig.set_size_inches(7, 10)
    else:
        fig.set_size_inches(7, 8)

    this_ax_idx = -1

    # Plot the reward time series (if requested)
    if (should_plot_reward):
        this_ax_idx = this_ax_idx + 1
        this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["reward"])
        # > Add the legend entry
        this_line.set_label("reward")
        legend_lines = []
        legend_lines.append(this_line)
        # Set the labels:
        #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
        axs[this_ax_idx].set_ylabel('[reward units]', fontsize=10)
        # Add grid lines
        axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
        # Show a legend
        axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)

    # Plot the cumulative reward time series (if requested)
    if (should_plot_reward):
        this_ax_idx = this_ax_idx + 1
        reward_cumulative_sum = np.cumsum(sim_time_series_results["reward"], dtype=np.float32)
        this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],reward_cumulative_sum)
        # > Add the legend entry
        this_line.set_label("cumulative reward")
        legend_lines = []
        legend_lines.append(this_line)
        # Set the labels:
        #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
        axs[this_ax_idx].set_ylabel('[reward units]', fontsize=10)
        # Add grid lines
        axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
        # Show a legend
        axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # # Plot the "progress" time series
    # this_ax_idx = this_ax_idx + 1
    # this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["road_progress_at_closest_point"])
    # # > Add the legend entry
    # this_line.set_label("progress along line")
    # legend_lines = []
    # legend_lines.append(this_line)
    # # Set the labels:
    # #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    # axs[this_ax_idx].set_ylabel('[meters]', fontsize=10)
    # # Add grid lines
    # axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # # Show a legend
    # axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Plot the "vx" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["vx"]*3.6)
    # > Add the legend entry
    this_line.set_label("forwards velocity (vx)")
    legend_lines = []
    legend_lines.append(this_line)
    # Plot the "recommended speed" time series
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["recommended_speed_at_closest_point"]*3.6)
    this_line.set_label("recommended speed")
    legend_lines.append(this_line)
    # Set the labels:
    #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('[km/h]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Plot the "distance to line" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["distance_to_closest_point"])
    # > Add the legend entry
    this_line.set_label("distance to line")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('[meters]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Plot the "heading relative to line" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["heading_angle_relative_to_line"]*180.0/3.141)
    # > Add the legend entry
    this_line.set_label("heading relative to line")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('[degrees]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Plot the "drive command" time series
    this_ax_idx = this_ax_idx + 1
    this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["drive_command"])
    # > Add the legend entry
    this_line.set_label("drive command")
    legend_lines = []
    legend_lines.append(this_line)
    # Set the labels:
    #axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('[%]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Plot the "delta request" time series
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
    legend_lines.append(this_line)
    # Set the labels:
    axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
    axs[this_ax_idx].set_ylabel('[degrees]', fontsize=10)
    # Add grid lines
    axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
    # Show a legend
    axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


    # Add an overall figure title
    fig.suptitle("Time series results", fontsize=12)

    # Save the plot
    if isinstance(file_name_suffix, str) and file_name_suffix == "":
        path_and_file_name = path_for_saving_figures + "/" + "ad_time_series" + ".pdf"
    else:
        path_and_file_name = path_for_saving_figures + "/" + "ad_time_series" + "_" + file_name_suffix + ".pdf"
    fig.savefig(path_and_file_name)
    print('Saved figure: ' + path_and_file_name)

    # Append the details of this plot
    plot_details_list.append({
        "name"  :  "time series data",
        "fig"   :  fig,
        "axs"   :  axs,
    })

    # Return the list of plot details
    return plot_details_list