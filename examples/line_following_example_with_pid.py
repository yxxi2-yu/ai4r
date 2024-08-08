#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium
import ai4rgym
from policies.pid_policy_for_autonomous_driving import PIDPolicyForAutonomousDriving





## ----------------------
#  PRINT THE RUNNING PATH
#  ----------------------
import os
# get the current working directory
current_working_directory = os.getcwd()
# print output to the console
print('This script is running from the following path:')
print(current_working_directory)



## -----------------------------------
#  SPECIFY THE PATH FOR SAVING FIGURES
#  -----------------------------------
path_for_saving_figures = 'examples/saved_figures'



## ------------------------------
#  SPECIFY THE VEHICLE PARAMETERS
#  ------------------------------

# Dictionary with car specifications,
# in the form of a dynamic bicycle model
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

# Note:
# The "v_transition_min" and "v_transition_max" specifications
# have defaults values of 3.0 and 5.0 m/s respectively.
# Set these values to be exceesively high to use a purely
# kinematic model of the vehicle.

# The model parameters above are based on a Telsa Model 3:
# > Source: https://www.tesla.com/ownersmanual/model3/en_cn/GUID-E414862C-CFA1-4A0B-9548-BE21C32CAA58.html
# > Source: https://www.tesla.com/sites/default/files/blog_attachments/the-slipperiest-car-on-the-road.pdf
#
# The Pacejka's tyre formula coefficients default to the value from here
# > Source: https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
# > https://au.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html



## ----------------
#  SPECIFY THE ROAD
#  ----------------

# Specified as a list of dictionaries, where each
# element in the list specifies a segment of the road.
# Example segment dictionaries:
# > {"type":"straight", "length":3.0}
# > {"type":"curved", "curvature":1/50.0, "angle_in_degrees":45.0}
# > {"type":"curved", "curvature":1/50.0, "length":30.0}
road_elements_list = [
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":1/2000.0, "angle_in_degrees":10.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/2000.0, "angle_in_degrees":10.0},
    {"type":"straight", "length":100.0},
]



## -----------------------------------------
#  SPECIFY THE NUMERICAL INTEGRATION DETAILS
#  -----------------------------------------

# The options available for the numerical
# integration method are:
# ["euler", "huen", "midpoint", "rk4", "rk45"]

numerical_integration_parameters = {
    "method" : "rk4",
    "Ts" : 0.05,
    "num_steps_per_Ts" : 1,
}



## ----------------------------------
#  SPECIFY THE TRUNCATION PARAMETERS
#  ----------------------------------
truncation_parameters = {
    "speed_lower_bound"  :  (10.0/3.6),
    "speed_upper_bound"  :  (200.0/3.6),
    "distance_to_closest_point_upper_bound"  :  20.0,
}



## -------------------------------------
#  SPECIFY THE INITIAL STATE DISTRUBTION
#  -------------------------------------

# The initial state is sampled from a uniform
# distribution between the minimum and maximum
# (i.e., between lower and upper bounds)
# > Note: a factor of (1/3.6) converts from units
#   of [km/h] to [m/s]

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



## ----------------------------------
#  SPECIFY THE OBSERVATION PARAMETERS
#  ----------------------------------
observation_parameters = {
    "should_include_ground_truth_px"                       :  "info",
    "should_include_ground_truth_py"                       :  "info",
    "should_include_ground_truth_theta"                    :  "info",
    "should_include_ground_truth_vx"                       :  "obs",
    "should_include_ground_truth_vy"                       :  "obs",
    "should_include_ground_truth_omega"                    :  "info",
    "should_include_ground_truth_delta"                    :  "info",
    "should_include_road_progress_at_closest_point"        :  "info",
    "should_include_vx_sensor"                             :  "info",
    "should_include_distance_to_closest_point"             :  "obs",
    "should_include_heading_angle_relative_to_line"        :  "info",
    "should_include_heading_angular_rate_gyro"             :  "info",
    "should_include_closest_point_coords_in_body_frame"    :  "info",
    "should_include_look_ahead_line_coords_in_body_frame"  :  "info",
    "should_include_road_curvature_at_closest_point"       :  "obs",
    "should_include_look_ahead_road_curvatures"            :  "obs",

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
    "distance_to_closest_point_stddev"  :  0.1,

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

## ---------------------------------------------
#  INITIALIZE THE AUTONOMOUS DRIVING ENVIRONMENT
#  ---------------------------------------------

# Options available for the "render_mode" are:
# ["matplotlib", None]

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



## ----------------------------
#  PLOT (i.e., RENDER) THE ROAD
#  ----------------------------

# Reset the environment
env.reset()
# Initialize the figure for plotting
env.unwrapped.render_matplotlib_init_figure()
# Plot the road
env.unwrapped.render_matplotlib_plot_road()
# Add a title
env.unwrapped.figure.suptitle('The road, i.e., the center of the lane to be followed', fontsize=12)
# Save the figure
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_road.pdf')

# Zoom into the start position
env.unwrapped.render_matplotlib_zoom_to(px=0,py=0,x_width=20,y_height=20)
# Add a title
env.unwrapped.figure.suptitle('Zoom in of the road and car', fontsize=12)
# Save the figure
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road_zoom.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_road_zoom.pdf')



## -------------------
#  PERFORM SIMULATIONS
#  -------------------

# Specify simulation length by:
# > Number of steps
N_sim = 850

# Create the policy
pid_policy = PIDPolicyForAutonomousDriving()

# Run the simulation
sim_time_series_results = env.unwrapped.simulate_policy(N_sim, pid_policy)



## ------------------------------------------------
#  PLOT THE RESULTS - IN CARTESIAN COORDINATE SPACE
#  ------------------------------------------------

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


## ------------------------------------------------
#  CREATE AN ANIMATION
#  ------------------------------------------------
ani = env.unwrapped.render_matplotlib_animation_of_trajectory(sim_time_series_results["px"], sim_time_series_results["py"], sim_time_series_results["theta"], sim_time_series_results["delta"], numerical_integration_parameters["Ts"], traj_increment=3, figure_title="Animation of car trajectory")

ani.save(path_for_saving_figures + '/ad_animation.gif')
print('Saved animation: ' + path_for_saving_figures + '/ad_animation.gif')

#from IPython.display import HTML
#HTML(ani.to_jshtml())

#plt.show()