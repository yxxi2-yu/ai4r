#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import gymnasium
import ai4rgym



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
}

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
    {"type":"straight", "length":200.0},
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":180.0},
    {"type":"straight", "length":200.0},
    {"type":"curved", "curvature":-1/50.0, "angle_in_degrees":180.0},
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
    #"num_steps_per_Ts" : 1,
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

v_init_min_in_kmh = 25.0
v_init_max_in_kmh = 35.0

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
env.unwrapped.figure.savefig('saved_figures/ad_road_minimal.pdf')



## -------------------
#  PERFORM SIMULATIONS
#  -------------------

# Specify simulation length by:
# > Number of steps
N_sim = 10000
# > Time increment per simulation step (units: seconds)
Ts_sim = 0.05

# Specify the integration method to simulate
integration_method = "rk4"

# Initialise array for storing (px,py) trajectory:
px_traj    = np.empty([N_sim+1,], dtype=np.float32)
py_traj    = np.empty([N_sim+1,], dtype=np.float32)

# Set the integration method and Ts of the gymnasium
env.unwrapped.set_integration_method(integration_method)
env.unwrapped.set_integration_Ts(Ts_sim)

# Reset the gymnasium
# > which also returns the first observation
observation, info_dict = env.reset()

# Put the initial condition into the first entry of the state trajectory results
this_time_index = 0
px_traj[this_time_index] = observation["px"]
py_traj[this_time_index] = observation["py"]

# Display that we are starting this simulation run
print("\n")
print("Now starting simulation.")

# Initialize the flag to when the car reaches
# the end of the road
run_terminated = False



# ITERATE OVER THE TIME STEPS OF THE SIMULATION
for i_step in range(N_sim):

    # Set the road condition
    env.unwrapped.set_road_condition(road_condition="wet")



    ## --------------------
    #  START OF POLICY CODE
    
    if (run_terminated):
        # Zero speed reference after reaching the end of the road
        speed_ref = 0.0
    else:
        # Constant speed reference while on the road
        speed_ref = 30.0/3.6

    # Get the "info_dict" observation of the distance to the line
    closest_distance = info_dict["closest_distance"]
    side_of_the_road_line = info_dict["side_of_the_road_line"]

    # Compute the speed of the car
    speed = np.sqrt( observation["vx"][0]**2 + observation["vy"][0]**2 )
    # Compute the error between the reference and actual speed
    speed_error = speed_ref - speed
    # Compute the drive command action
    drive_command_raw = 2.0 * speed_error
    # Clip the drive command action to be in the range [-100,100] percent
    drive_command_clipped = max(-100.0, min(drive_command_raw, 100.0))

    # Compute the steering angle request action
    delta_request = 4.0*(np.pi/180.0) * closest_distance * -side_of_the_road_line

    # Construct the action dictionary expected by the gymnasium
    action = {
        "drive_command" : drive_command_clipped,
        "delta_request" : delta_request
    }

    # Zero steering after reaching the end of the road
    if (run_terminated):
        action["delta_request"] = 0.0

    #  END OF POLICY CODE
    ## --------------------



    # Step forward the gymnasium
    observation, reward, terminated, truncated, info_dict = env.step(action)

    # Store the results
    this_time_index = this_time_index+1
    px_traj[this_time_index] = observation["px"]
    py_traj[this_time_index] = observation["py"]

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
fig.savefig('saved_figures/ad_cartesian_coords_minimal.pdf') 
