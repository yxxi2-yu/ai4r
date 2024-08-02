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
    {"type":"straight", "length":30.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":120.0},
    {"type":"straight", "length":30.0},
    {"type":"curved", "curvature":-1/40.0, "angle_in_degrees":120.0},
    {"type":"straight", "length":20.0},
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
    "should_include_obs_for_ground_truth_state"                    :  False,
    "should_include_obs_for_vx_sensor"                             :  True,
    "should_include_obs_for_closest_distance_to_line"              :  True,
    "should_include_obs_for_heading_angle_relative_to_line"        :  True,
    "should_include_obs_for_heading_angle_gyro"                    :  False,
    "should_include_obs_for_accel_in_body_frame_x"                 :  False,
    "should_include_obs_for_accel_in_body_frame_y"                 :  False,
    "should_include_obs_for_look_ahead_line_coords_in_body_frame"  :  True,
    "should_include_obs_for_gps_line_coords_in_world_frame"        :  False,

    "scaling_for_ground_truth_state"                    :  0.0,
    "scaling_for_vx_sensor"                             :  0.0,
    "scaling_for_closest_distance_to_line"              :  0.0,
    "scaling_for_heading_angle_relative_to_line"        :  0.0,
    "scaling_for_heading_angle_gyro"                    :  0.0,
    "scaling_for_accel_in_body_frame_x"                 :  0.0,
    "scaling_for_accel_in_body_frame_y"                 :  0.0,
    "scaling_for_look_ahead_line_coords_in_body_frame"  :  0.0,
    "scaling_for_gps_line_coords_in_world_frame"        :  0.0,

    "vx_sensor_bias"   : 0.0,
    "vx_sensor_stddev" : 0.1,

    "closest_distance_to_line_bias"    :  0.0,
    "closest_distance_to_line_stddev"  :  0.05,

    "heading_angle_relative_to_line_bias"    :  0.0,
    "heading_angle_relative_to_line_stddev"  :  0.01,

    "heading_angle_gyro_bias"    :  0.0,
    "heading_angle_gyro_stddev"  :  0.0,

    "look_ahead_line_coords_in_body_frame_distance"             :  50.0,
    "look_ahead_line_coords_in_body_frame_num_points"           :  10,
    "look_ahead_line_coords_in_body_frame_stddev_lateral"       :  0.0,
    "look_ahead_line_coords_in_body_frame_stddev_longitudinal"  :  0.0,
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
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road_minimal.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_road_minimal.pdf')

# Zoom into the start position
env.unwrapped.render_matplotlib_zoom_to(px=0,py=0,x_width=20,y_height=20)
# Add a title
env.unwrapped.figure.suptitle('Zoom in of the road and car', fontsize=12)
# Save the figure
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road_zoom_minimal.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_road_zoom_minimal.pdf')



## -------------------
#  PERFORM SIMULATIONS
#  -------------------

# Specify simulation length by:
# > Number of steps
N_sim = 850
# > Time increment per simulation step (units: seconds)
Ts_sim = 0.05

# Specify the integration method to simulate
integration_method = "rk4"

# Specify the "progress queries" to look ahead 50 meters
progress_queries = np.array([100.0], dtype=np.float32)

# Initialise array for storing (px,py) trajectory:
px_traj    = np.empty([N_sim+1,], dtype=np.float32)
py_traj    = np.empty([N_sim+1,], dtype=np.float32)
theta_traj = np.empty([N_sim+1,], dtype=np.float32)
delta_traj = np.empty([N_sim+1,], dtype=np.float32)

# Set the integration method and Ts of the gymnasium
env.unwrapped.set_integration_method(integration_method)
env.unwrapped.set_integration_Ts(Ts_sim)

# Set the progress queries
env.unwrapped.set_progress_queries_for_generating_observations(progress_queries)

# Reset the gymnasium
# > which also returns the first observation
observation, info_dict = env.reset()

# Put the initial condition into the first entry of the state trajectory results
this_time_index = 0
px_traj[this_time_index]    = observation["px"][0]
py_traj[this_time_index]    = observation["py"][0]
theta_traj[this_time_index] = observation["theta"][0]
delta_traj[this_time_index] = observation["delta"][0]

# Display that we are starting this simulation run
print("\n")
print("Now starting simulation.")

# Initialize the flag to when the car reaches
# the end of the road
run_terminated = False


pid_policy = PIDPolicyForAutonomousDriving()

# ITERATE OVER THE TIME STEPS OF THE SIMULATION
for i_step in range(N_sim):

    # Set the road condition
    env.unwrapped.set_road_condition(road_condition="wet")



    ## --------------------
    #  START OF POLICY CODE

    action = pid_policy.compute_action(observation, info_dict, run_terminated)

    #  END OF POLICY CODE
    ## --------------------



    # Step forward the gymnasium
    observation, reward, terminated, truncated, info_dict = env.step(action)

    # Store the results
    this_time_index = this_time_index+1
    px_traj[this_time_index]    = observation["px"][0]
    py_traj[this_time_index]    = observation["py"][0]
    theta_traj[this_time_index] = observation["theta"][0]
    delta_traj[this_time_index] = observation["delta"][0]

    # Check whether the car reached the end of the road
    if terminated:
        if not(run_terminated):
            run_terminated = True

# FINISHED ITERATING OVER THE SIMULATION TIME

# Display that the simulation is finished
print("Simulation finished")
print("\n")


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
this_line, = axs.plot(px_traj,py_traj)
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
fig.savefig(path_for_saving_figures + '/ad_cartesian_coords_minimal.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_cartesian_coords_minimal.pdf')


## ------------------------------------------------
#  CREATE AN ANIMATION
#  ------------------------------------------------
ani = env.unwrapped.render_matplotlib_animation_of_trajectory(px_traj, py_traj, theta_traj, delta_traj, numerical_integration_parameters["Ts"], traj_increment=3, figure_title="Animation of car trajectory")

ani.save(path_for_saving_figures + '/ad_animation_minimal.gif')
print('Saved animation: ' + path_for_saving_figures + '/ad_animation_minimal.gif')

#from IPython.display import HTML
#HTML(ani.to_jshtml())

#plt.show()