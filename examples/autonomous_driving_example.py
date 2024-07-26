#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import gymnasium
import ai4rgym



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
    "Dp_dry"  :  1.0,
    "Cp_dry"  :  1.9,
    "Bp_dry"  : 10.0,
    "Ep_dry"  :  0.9,
    "Dp_wet"  :  0.8,
    "Cp_wet"  :  2.3,
    "Bp_wet"  : 12.0,
    "Ep_wet"  :  1.0,
    "Dp_snow" :  0.3,
    "Cp_snow" :  2.0,
    "Bp_snow" :  5.0,
    "Ep_snow" :  1.0,
    "Dp_ice"  :  0.1,
    "Cp_ice"  :  2.0,
    "Bp_ice"  :  4.0,
    "Ep_ice"  :  1.0,
}

# The model parameters above are based on a Telsa Model 3:
# > Source: https://www.tesla.com/ownersmanual/model3/en_cn/GUID-E414862C-CFA1-4A0B-9548-BE21C32CAA58.html
# > Source: https://www.tesla.com/sites/default/files/blog_attachments/the-slipperiest-car-on-the-road.pdf
# > WEIGHT AND SIZE:
#   > Wheel base = 2.875 [meters]
#   > Mass (empty)  = { 1779 (f:845, r:934 ), 1900 (f:948, r:952 ), 1897 (f:946, r:951 ), 1617 (f:750, r:867 ) } [kg]
#   > Mass (loaded) = { 2184 (f:975, r:1209), 2300 (f:1075,r:1225), 2300 (f:1075,r:1225), 2017 (f:878, r:1139) } [kg]
#   > Length overall = 4.692 [m]
#   > Width (w/o mirrors) = 1.850 [m]
#   => Mass moment of inertia (Iz) approx. = (1/12) * mass * width^2 * length^2
# > DRIVE FORCE
#   > Max torque (per motor) = {219, 326, 404} [Nm]
#   > Motor speed at max torque = {6380, 6000, 5000} [RPM]
#   > Gearbox ratio: 1:9
#   > Number of motors = 2
#   > Wheel diameter = {18, 19} [inches]
#     => radius = {0.2286, 0.2413} [meters]
#   => "Cm" = (1/100) * "max torque" / "wheel radius"
# > AREODYNAMIC DRAG
#   > Drag coefficient = {0.24, 0.26}
#   > Frontal drag area = {23.9, 25.2} [ft^2]
#     => 0.3048   [m/ft]
#     => 0.3048^2 [m^2 / ft^2]
#     => {2.2204, 2.3412} [m^2]
#   > Air density = 1.202 [kg/m^3]
#   > Drag force = 0.5 * Cd * Area * rho_air * v^2
#   => "Cd" = 0.5 * Cd * Area * rho_air

# The model parameters for tire-to-road interaction are based on 
# Pacejka's tyre formula coefficients (peak, shape, stiffness, curvature)
# > Source: https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
# > https://au.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html
# -----------------------------------------------------------------------
#     Name         Typical range   Typical values for longitudinal forces
#                                  Dry tarmac   Wet tarmac   Snow   Ice
# -----------------------------------------------------------------------
# D   Peak         0.1 -  1.9       1.0          0.82        0.3    0.1
# C*  Shape        1.0 -  2.0       1.9          2.3         2.0    2.0
# B   Stiffness    4.0 - 12.0      10.0         12.0         5.0    4.0
# E   Curvature  -10.0 -  1.0       0.97         1.0         1.0    1.0
# -----------------------------------------------------------------------
# * The Pacejka model specifies the shape as C=1.65 for the longitudinal
#   force and C=1.3 for the lateral force.



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
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":45.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/100.0, "angle_in_degrees":45.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":90.0},
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/200.0, "angle_in_degrees":20.0},
    {"type":"straight", "length":40.0},
    {"type":"curved", "curvature":-1/30.0, "angle_in_degrees":90.0},
    {"type":"curved", "curvature":-1/30.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":40.0},
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":30.0},
    {"type":"straight", "length":200.0},
    {"type":"curved", "curvature":-1/50.0, "angle_in_degrees":40.0},
    {"type":"straight", "length":30.0},
    {"type":"curved", "curvature":1/50.0, "angle_in_degrees":20.0},
    {"type":"straight", "length":40.0},
    {"type":"curved", "curvature":1/20.0, "angle_in_degrees":160.0},
    {"type":"straight", "length":70.0},
    {"type":"curved", "curvature":-1/30.0, "angle_in_degrees":150.0},
    {"type":"straight", "length":50.0},
    {"type":"curved", "curvature":1/40.0, "angle_in_degrees":160.0},
    {"type":"straight", "length":300.0},
    {"type":"curved", "curvature":1/60.0, "angle_in_degrees":20.0},
    {"type":"straight", "length":10.0},
    {"type":"curved", "curvature":-1/20.0, "angle_in_degrees":90.0},
    {"type":"curved", "curvature":-1/20.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":20.0},
    {"type":"curved", "curvature":1/80.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":50.0},
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":200.0},
    {"type":"curved", "curvature":1/100.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":300.0},
    {"type":"curved", "curvature":-1/150.0, "angle_in_degrees":45.0},
    {"type":"straight", "length":20.0},
    {"type":"curved", "curvature":1/150.0, "angle_in_degrees":45.0},
    {"type":"straight", "length":32},
]
# Side note for interest, this road is vaguely based on
# the road to Albispass (plus a bit of loop-back):
# https://maps.app.goo.gl/aedWreLkR41Xyf1R8



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
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road.pdf')

# Zoom into the start position
env.unwrapped.render_matplotlib_zoom_to(px=0,py=0,x_width=20,y_height=20)
# Add a title
env.unwrapped.figure.suptitle('Zoom in of the road and car', fontsize=12)
# Save the figure
env.unwrapped.figure.savefig(path_for_saving_figures + '/ad_road_zoom.pdf')

#env.unwrapped.car.render_car(axis,px,py,theta,delta,scale=1.0)



## ----------------------------------------------
#  DISPLAY SOME INFROMATION ABOUT THE ENVIRONMENT
#  ----------------------------------------------

# To assist with constructing the road, we can print out
# some details of each road segment
# print("\n")
# print("start_points = ")
# print(env.unwrapped.road.get_start_points())

print("\n")
print("end_points = ")
print(env.unwrapped.road.get_end_points())

# print("\n")
# print("start_angles = ")
# print(env.unwrapped.road.get_start_angles())

# print("\n")
# print("end_angles = ")
# print(env.unwrapped.road.get_end_angles())

# print("\n")
# print("l_total_at_end = ")
# print(env.unwrapped.road.get_l_total_at_end())


# Display a few more details
print("\nEnvironment:")
print(env)
print("\nAction space:")
print(env.action_space)
print("\nObservation space:")
print(env.observation_space)
print("\n")



## -------------------
#  PERFORM SIMULATIONS
#  -------------------

# Specify simulation length by:
# > Number of steps
N_sim = 10000
# > Time increment per simulation step (units: seconds)
Ts_sim = 0.05

# Specify the integration methods to simulate
#integration_methods = ["euler", "midpoint", "huen", "rk4", "rk45"]
integration_methods = ["euler", "rk4"]

# Specify the "progress queries"
progress_queries = np.array([0.0,1.0,2.0,3.0,4.0,5.0], dtype=np.float32)
env.unwrapped.set_progress_queries_for_generating_observations(progress_queries)
# The "progress_queries" are defines as:
#   Specifies the values of progress-along-the-road,
#   relative to the current position of the car, at
#   which the observations should be generated. These
#   observations are returned in the "info_dict".
#   (Units: meters)

# Initialise array for storing:
# > The state trajectories
px_traj    = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
py_traj    = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
theta_traj = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
vx_traj    = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
vy_traj    = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
omega_traj = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
delta_traj = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
# > The slip angles of the rear and front wheel
alpha_r_traj = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
alpha_f_traj = np.empty([len(integration_methods),N_sim+1], dtype=np.float32)
# > The action trajectories
drive_command_traj = np.empty([len(integration_methods),N_sim], dtype=np.float32)
delta_request_traj = np.empty([len(integration_methods),N_sim], dtype=np.float32)
# > The trajectories of the closest point on the road
px_closest_traj  = np.zeros([len(integration_methods),N_sim+1],dtype=float)
py_closest_traj  = np.zeros([len(integration_methods),N_sim+1],dtype=float)
# > The computation time to perform the simulations
process_time = np.zeros([len(integration_methods),],dtype=float)



# ITERATE OVER THE INTEGRATION METHODS
for integration_method in integration_methods:
    # Get the integration method index for this iteration
    this_method_index = integration_methods.index(integration_method)

    # Set the integration method and Ts of the gymnasium
    env.unwrapped.set_integration_method(integration_method)
    env.unwrapped.set_integration_Ts(Ts_sim)

    # Reset the gymnasium
    # > which also returns the first observation
    observation, info_dict = env.reset()

    # Put the initial condition into the first entry of the state trajectory results
    this_time_index = 0
    px_traj[this_method_index,this_time_index]    = observation["px"]
    py_traj[this_method_index,this_time_index]    = observation["py"]
    theta_traj[this_method_index,this_time_index] = observation["theta"]
    vx_traj[this_method_index,this_time_index]    = observation["vx"]
    vy_traj[this_method_index,this_time_index]    = observation["vy"]
    omega_traj[this_method_index,this_time_index] = observation["omega"]
    delta_traj[this_method_index,this_time_index] = observation["delta"]

    px_closest_traj[this_method_index,this_time_index]  = info_dict["px_closest"]
    py_closest_traj[this_method_index,this_time_index]  = info_dict["py_closest"]

    # Display that we are starting this simulation run
    print("\n")
    print("Now using method: " + str(integration_method) + " with index = " + str(this_method_index) )
    print("Initial state       = [ " + "{:8.2f}".format(observation["px"][0]) + " , " + "{:8.2f}".format(observation["py"][0]) + " , " + "{:8.2f}".format(observation["theta"][0]*180/np.pi) + " ], in units [m,m,deg]")

    # "Start" a wall-clock timer
    this_start_time = time.process_time()

    # Initialize the flag to when the car reaches
    # the end of the road
    run_terminated = False



    # ITERATE OVER THE TIME STEPS OF THE SIMULATION
    for i_step in range(N_sim):

        # Set the road condition
        env.unwrapped.set_road_condition(road_condition="wet")
        # Note: reasons for changing the road condition
        # during a simulation:
        # > Represents that it start raining after some time.
        # > Represent that a part of the road is slippery
        #   (e.g., ice), hence road condition can change as a
        #   function of progress.

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
        # > For the actions at this time step
        drive_command_traj[this_method_index,this_time_index] = action["drive_command"]
        delta_request_traj[this_method_index,this_time_index] = action["delta_request"]
        # Increment that time index counter
        this_time_index = this_time_index+1
        # > For the state:
        px_traj[this_method_index,this_time_index]    = observation["px"]
        py_traj[this_method_index,this_time_index]    = observation["py"]
        theta_traj[this_method_index,this_time_index] = observation["theta"]
        vx_traj[this_method_index,this_time_index]    = observation["vx"]
        vy_traj[this_method_index,this_time_index]    = observation["vy"]
        omega_traj[this_method_index,this_time_index] = observation["omega"]
        delta_traj[this_method_index,this_time_index] = observation["delta"]
        
        # > For the closest point
        px_closest_traj[this_method_index,this_time_index] = info_dict["px_closest"]
        py_closest_traj[this_method_index,this_time_index] = info_dict["py_closest"]

        # Check whether the car reached the end of the road
        if terminated:
            if not(run_terminated):
                run_terminated = True
                #observation, info = env.reset()
                print("Reached the end of the road after " + str(i_step) + " time steps.")
                print("(px,py,progess) = ( " + str(observation["px"]) + " , " + str(observation["py"]) + " , " + str(info_dict["progress_at_closest_p"]) + " )" )
                print("total road length = " + str(env.unwrapped.road.get_total_length()) )

    # FINISHED ITERATING OVER THE SIMULATION TIME

    # Record the end time of this simulation run
    this_end_time = time.process_time()

    # Compute and store the simulation time
    this_sim_time = this_end_time - this_start_time
    process_time[this_method_index] = this_sim_time

    # Compute and store the slip angles for the front and rear wheels
    temp_Lr = env.unwrapped.car.Lr
    temp_Lf = env.unwrapped.car.Lf
    alpha_r_traj[this_method_index,:] = -np.arctan( (vy_traj[this_method_index,:]-temp_Lr*omega_traj[this_method_index,:])/vx_traj[this_method_index,:])
    alpha_f_traj[this_method_index,:] = -np.arctan( (vy_traj[this_method_index,:]+temp_Lf*omega_traj[this_method_index,:])/vx_traj[this_method_index,:]) + delta_traj[this_method_index,:]

    if not(run_terminated):
        print("Did not reach the end of the road after " + str(i_step) + " time steps.")
        print("(px,py,progess) = ( " + str(observation["px"]) + " , " + str(observation["py"]) + " , " + str(info_dict["progress_at_closest_p"]) + " )" )
        print("total road length = " + str(env.unwrapped.road.get_total_length()) )

# FINISHED ITERATING OVER THE INTEGRATION METHODS

# Display some simple information about the simulation
print("\n")
print("[px,py,theta] at the last time step for each method:")
for integration_method in integration_methods:
    this_method_index = integration_methods.index(integration_method)
    print("[ " + "{:6.2f}".format(px_traj[this_method_index,-1]) + " , " + "{:6.2f}".format(py_traj[this_method_index,-1]) + " , " + "{:6.2f}".format(theta_traj[this_method_index,-1]*180/np.pi) + " ], for " + str(integration_method))

print("\n")
print("Process execution time [seconds] for each method")
for integration_method in integration_methods:
    this_method_index = integration_methods.index(integration_method)
    print("{:10.3f}".format(process_time[this_method_index]*1000) + " [msec], for " + str(integration_method))



## --------------------------------------------
#  PLOT THE RESULTS - OF TRAJECTORIES OVER TIME
#  --------------------------------------------

print("\n")
print("Now plotting things")

# Specify the number of time steps (backwards from the end) to plot
number_of_steps_to_plot = N_sim

# Construct a vector for the time [in seconds] at each step
time_steps  = np.linspace(0, N_sim*Ts_sim, num=N_sim+1, dtype=float)

# Open the figure
fig, axs = plt.subplots(7, 1, sharex=True, sharey=False,gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.12})

# > Make the figure taller
temp_size = fig.get_size_inches()
fig.set_size_inches(temp_size[0], temp_size[1]*1.5)

# Initialize a list for the legend
legend_lines = []

# ITERATE OVER THE INTEGRATION METHODS
for integration_method in integration_methods:
    # Get the integration method index for this iteration
    this_method_index = integration_methods.index(integration_method)

    # Plot the px time-series result
    this_px_to_plot = px_traj[this_method_index,:]
    this_line, = axs[0].plot(time_steps[-number_of_steps_to_plot:-1],this_px_to_plot[-number_of_steps_to_plot:-1])
    # > Add the legend entry
    this_line.set_label(integration_method)
    legend_lines.append(this_line)

    # Plot the py time-series result
    this_py_to_plot = py_traj[this_method_index,:]
    axs[1].plot(time_steps[-number_of_steps_to_plot:-1],this_py_to_plot[-number_of_steps_to_plot:-1])

    # Plot the theta time-series result
    this_theta_to_plot = theta_traj[this_method_index,:]*(180/np.pi)
    axs[2].plot(time_steps[-number_of_steps_to_plot:-1],this_theta_to_plot[-number_of_steps_to_plot:-1])

    # Plot the delta time-series result
    this_delta_to_plot = delta_traj[this_method_index,:]*(180/np.pi)
    axs[3].plot(time_steps[-number_of_steps_to_plot:-1],this_delta_to_plot[-number_of_steps_to_plot:-1])

    # Plot the rear slip angle (alpha_r) time-series result
    this_alpha_r_to_plot = alpha_r_traj[this_method_index,:]*(180/np.pi)
    axs[4].plot(time_steps[-number_of_steps_to_plot:-1],this_alpha_r_to_plot[-number_of_steps_to_plot:-1])

    # Plot the rear slip angle (alpha_r) time-series result
    this_alpha_f_to_plot = alpha_f_traj[this_method_index,:]*(180/np.pi)
    axs[5].plot(time_steps[-number_of_steps_to_plot:-1],this_alpha_f_to_plot[-number_of_steps_to_plot:-1])

    # Plot the drive command time-series result
    this_drive_command_to_plot = drive_command_traj[this_method_index,:]
    axs[6].plot(time_steps[(-number_of_steps_to_plot-1):-2],this_drive_command_to_plot[-number_of_steps_to_plot:-1])

# Set the labels:
# > X axis labels
axs[6].set_xlabel('time [sec]', fontsize=10)
# > Y axis labels
axs[0].set_ylabel("px\n[m]", fontsize=10)
axs[1].set_ylabel("py\n[m]", fontsize=10)
axs[2].set_ylabel("theta\n[deg]", fontsize=10)
axs[3].set_ylabel("delta\n[deg]", fontsize=10)
axs[4].set_ylabel("alpha_r\n[deg]", fontsize=10)
axs[5].set_ylabel("alpha_f\n[deg]", fontsize=10)
axs[6].set_ylabel("drive\n[%]", fontsize=10)

# Add grid lines
for axis in axs:
    axis.grid(visible=True, which="both", axis="both", linestyle='--')

# Show a legend for the whole figure
fig.legend(handles=legend_lines, loc="lower center", ncol=4, labelspacing=0.1)

# Add an overall figure title
fig.suptitle("Comparing numerical integration methods on closed loop simulations", fontsize=12)

# Save the plot
fig.savefig(path_for_saving_figures + '/ad_time_series.pdf')



## ------------------------------------------------
#  PLOT THE RESULTS - IN CARTESIAN COORDINATE SPACE
#  ------------------------------------------------

# Open the figure
fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

# Render the road onto the axis
env.unwrapped.road.render_road(axs)

# Initialize a list for the legend
legend_lines = []

# ITERATE OVER THE INTEGRATION METHODS
for integration_method in integration_methods:
    # Get the integration method index for this iteration
    this_method_index = integration_methods.index(integration_method)

    # Plot the (px,py) trajectory
    this_px_to_plot = px_traj[this_method_index,:]
    this_py_to_plot = py_traj[this_method_index,:]
    this_line, = axs.plot(this_px_to_plot[-number_of_steps_to_plot:-1],this_py_to_plot[-number_of_steps_to_plot:-1])
    # > Add the legend entry
    this_line.set_label(integration_method)
    legend_lines.append(this_line)

# For only the last integration method, plot the (px,py)
# trajectory of the closest point on the road.
this_px_to_plot = px_closest_traj[this_method_index,:]
this_py_to_plot = py_closest_traj[this_method_index,:]
this_line, = axs.plot(this_px_to_plot[-number_of_steps_to_plot:-1],this_py_to_plot[-number_of_steps_to_plot:-1])
# > Add the legend entry
this_line.set_label("closest road points for " + integration_method)
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
fig.suptitle("Showing the road and the (px,py) trajectories", fontsize=12)

# Save the plot
fig.savefig(path_for_saving_figures + '/ad_cartesian_coords.pdf')
