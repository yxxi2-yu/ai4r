#!/usr/bin/env python

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium
import ai4rgym
#from policies.pid_policy_for_autonomous_driving import PIDPolicyForAutonomousDriving
from evaluation.evaluation_for_autonomous_driving import simulate_policy
from evaluation.evaluation_for_autonomous_driving import plot_results_from_time_series_dict
from evaluation.evaluation_for_autonomous_driving import plot_road_from_list_of_road_elements
from evaluation.evaluation_for_autonomous_driving import plot_current_state_zoomed_in



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
    {"type":"straight", "length":10.0},
    {"type":"curved", "curvature":1/5.0, "angle_in_degrees":45.0},
    {"type":"straight", "length":8.0},
]



## -------------
#  PLOT THE ROAD
#  -------------
# Specify a suffix for the plot file names
# > This is useful if the plot is called within a loop,
#   for example, the suffix could be an identifer for
#   the particular road.
file_name_suffix = "with_cones"

# Specify the cone placement parameters
width_btw_cones = 1.0
mean_length_btw_cones = 0.5
stddev_of_length_btw_cones = 0.00

# Call the plotting function
road_plot_details = plot_road_from_list_of_road_elements(road_elements_list, path_for_saving_figures, file_name_suffix, width_btw_cones=width_btw_cones, mean_length_btw_cones=mean_length_btw_cones, stddev_of_length_btw_cones=stddev_of_length_btw_cones)



## ------------------------------
#  SPECIFY THE VEHICLE PARAMETERS
#  ------------------------------

# Dictionary with car specifications,
# in the form of a dynamic bicycle model
bicycle_model_parameters = {
    "Lf" : 0.60*0.33,
    "Lr" : 0.40*0.33,
    "m"  : 3.0,
    "Iz" : (1.0/12.0) * 3.0 * (0.40**2+0.25**2),
    "Cm" : (1.0/100.0) * 10.0,
    "Cd" : 1.0,
    "delta_offset" : 0.0 * np.pi/180,
    "delta_request_max" : 45 * np.pi/180,
    "Ddelta_lower_limit" : -90 * np.pi/180,
    "Ddelta_upper_limit" :  90 * np.pi/180,
    "v_transition_min" : 500.0 / 3.6,
    "v_transition_max" : 600.0 / 3.6,
    "body_len_f" : (0.60*0.33) * 1.5,
    "body_len_r" : (0.40*0.33) * 1.5,
    "body_width" : 0.25,
}

# Note:
# The "v_transition_min" and "v_transition_max" specifications
# have defaults values of 3.0 and 5.0 m/s respectively.
# Set these values to be exceesively high to use a purely
# kinematic model of the vehicle.

# The model parameters above are very loosely intended to match a 1/10 scale
# Traxxas Slash RC car.
#
# The Pacejka's tyre formula coefficients default to the value from here
# > Source: https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
# > https://au.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html



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
#  SPECIFY THE TERMINATION PARAMETERS
#  ----------------------------------
termination_parameters = {
    "speed_lower_bound"  :  0.0,
    "speed_upper_bound"  :  (40.0/3.6),
    "distance_to_closest_point_upper_bound"  :  20.0,

    "reward_speed_lower_bound"  :  -100.0,
    "reward_speed_upper_bound"  :  -100.0,
    "reward_distance_to_closest_point_upper_bound"  :  -100.0,
}



## -------------------------------------
#  SPECIFY THE INITIAL STATE DISTRUBTION
#  -------------------------------------

# The initial state is sampled from a uniform
# distribution between the minimum and maximum
# (i.e., between lower and upper bounds)
# > Note: a factor of (1/3.6) converts from units
#   of [km/h] to [m/s]

py_init_min = -0.1
py_init_max =  0.1

v_init_min_in_meters_per_sec = 1.0
v_init_max_in_meters_per_sec = 1.0

initial_state_bounds = {
    "px_init_min" : 0.0,
    "px_init_max" : 0.0,
    "py_init_min" : py_init_min,
    "py_init_max" : py_init_max,
    "theta_init_min" : 0.0,
    "theta_init_max" : 0.0,
    "vx_init_min" : v_init_min_in_meters_per_sec,
    "vx_init_max" : v_init_min_in_meters_per_sec,
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
    "should_include_ground_truth_vx"                       :  "info",
    "should_include_ground_truth_vy"                       :  "info",
    "should_include_ground_truth_omega"                    :  "info",
    "should_include_ground_truth_delta"                    :  "info",
    "should_include_road_progress_at_closest_point"        :  "info",
    "should_include_vx_sensor"                             :  "obs",
    "should_include_distance_to_closest_point"             :  "info",
    "should_include_heading_angle_relative_to_line"        :  "info",
    "should_include_heading_angular_rate_gyro"             :  "info",
    "should_include_closest_point_coords_in_body_frame"    :  "info",
    "should_include_look_ahead_line_coords_in_body_frame"  :  "neither",
    "should_include_road_curvature_at_closest_point"       :  "neither",
    "should_include_look_ahead_road_curvatures"            :  "neither",
    "should_include_cone_detections"                       :  "obs",

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
    "scaling_for_cone_detections"                       :  1.0,

    "vx_sensor_bias"    : 0.0,
    "vx_sensor_stddev"  : 0.0, #0.1

    "distance_to_closest_point_bias"    :  0.0,
    "distance_to_closest_point_stddev"  :  0.00, #0.01

    "heading_angle_relative_to_line_bias"    :  0.0,
    "heading_angle_relative_to_line_stddev"  :  0.00, #0.01

    "heading_angular_rate_gyro_bias"    :  0.0,
    "heading_angular_rate_gyro_stddev"  :  0.00,  #0.01

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

    "cone_detections_width_btw_cones"             :  1.0,
    "cone_detections_mean_length_btw_cones"       :  0.5,
    "cone_detections_stddev_of_length_btw_cones"  :  0.01,

    "cone_detections_fov_horizontal_degrees"         :  80.0,
    "cone_detections_body_x_upper_bound"             :  4.0,
    "cone_detections_stddev_of_detection_in_body_x"  :  0.01,
    "cone_detections_stddev_of_detection_in_body_y"  :  0.01,
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
    termination_parameters=termination_parameters,
    initial_state_bounds=initial_state_bounds,
    observation_parameters=observation_parameters,
)



## -------------------------
#  PLOT AN INITIAL CONDITION
#  -------------------------

# Reset the environment
obs, info = env.reset()
# Specify a suffix for the plot file names
file_name_suffix = "with_cones"
# Call the plotting function
plot_current_state_zoomed_in(env, path_for_saving_figures, file_name_suffix)
# Display the observation for interest
print(obs)



## ---------------------
#  DEFINE A POLICY CLASS
#  ---------------------
class PIDPolicyForConeFollowing:
    """
    This class implements a policy for Autonomus Driving
    that mostly matches the policy provided in the policy_node
    """

    def __init__(self):
        """
        Initialization function for the "PIDPolicyForAutonomousDriving" class.
        """
        self.speed_ref = 1.0
        self.drive_command_baseline = 10.0
        self.kp_cruise_control = 4.0



    def compute_action(self, observation, info_dict, terminated, truncated):
        # Set the speed reference
        speed_ref = self.speed_ref
        # Zero speed reference after reaching the end of the road
        if (terminated or truncated):
            speed_ref = 0.0

        # Simple Proportional-only controll for speed control
        # > Get the speed observation of the car
        speed = observation["vx_sensor"][0]
        # > Compute the error between the reference and actual speed
        speed_error = speed_ref - speed
        # > Compute the drive command action
        drive_command_raw = self.drive_command_baseline + self.kp_cruise_control * speed_error
        # > Clip the drive command action to be in the range [-100,100] percent
        drive_command_clipped = max(-100.0, min(drive_command_raw, 100.0))

        # Initialise the steering request to zero
        delta_request = 0.0



        # Extract the number of cones
        num_cones = observation["cone_detections_num_cones"][0]
        # > On the real car, this is:
        #num_cones = msg.n

        # Extract the coordinates and colour of each cone
        # > Note: multiplying by 1000.0 to convert from meters to millimeters
        x_coords_np_array = observation["cone_detections_x"][:num_cones] * 1000.0
        y_coords_np_array = observation["cone_detections_y"][:num_cones] * 1000.0
        x_coords = x_coords_np_array.tolist()
        y_coords = y_coords_np_array.tolist()
        cone_colour = observation["cone_detections_side_of_road"][:num_cones].tolist()

        #print(x_coords)

        # > On the real car, this is:
        #x_coords = msg.x
        #y_coords = msg.y
        #cone_colour = msg.c
        
        # =======================================
        # START OF: INSERT POLICY CODE BELOW HERE
        # =======================================

        # OBSERVATIONS:
        # > The "x_coords", "y_coords" and "cone_colour" variable are:
        #   - Lists with length equal to num_cones (the number of cones detected).
        #   - "x_coords" and "y_coords" give the x and y world coordinates of the cones respectively
        #   - "cone_colour" gives the colour of the cones (0 for yellow, 1 for blue)
        #   - A cone represents a single index of all three lists e.g. x_coords[i], y_coords[i], cone_colour[i] represent the ith cone

        # ACTIONS:
        # > The "esc_action" is:
        #   - The action for driving the main motor (esc := electronic speed contrller).
        #   - In units of signed percent, with valid range [-100,100]
        # > The "steering_action" is:
        #   - The action for changing the position of the steering servo.
        #   - In units of signed percent, with valid range [-100,100]

        # ACRONYMS:
        # "esc" := Electronic Speed Controller
        #   - This device on the Traxxas car does NOT control the speed.
        #   - The "esc" device set the voltage to the motor within the
        #     range [-(max voltage),+(max voltage)]


        # =====================================
        # START OF SIMPLE POLICY
        # =====================================
        ROAD_WIDTH = 1000           # Width of the Road (in mm)
        P_STEERING = 0.005            # P value for Steering Action
        STEER_OFFSET = 0            # Offset to adjust for biased steering
        #FIXED_ESC_VALUE = 33.0      # Fixed ESC value for steering tests

        yellow_line = False         # Flag to signal yellow line detection 
        blue_line = False           # Flag to signal blue line detection

        # Convert to arrays for convenience
        x_array = np.array(x_coords)
        y_array = np.array(y_coords)
        c_array = np.array(cone_colour)

        # Create boolean masks based on c_np for yellow (0) and blue (1)
        yellow_mask = (c_array == 0)
        blue_mask = (c_array == 1)

        # Filter co-ordinates based on color
        bx = x_array[blue_mask]
        by = y_array[blue_mask]
        yx = x_array[yellow_mask]
        yy = y_array[yellow_mask]

        # Fit a line to yellow points if there are enough points (>=2 points)
        if len(yx) > 1:
            try:
                yellow_fit = np.polyfit(yx, yy, 1)  # Linear fit (y = mx + b)
                yellow_line = True
                #self.get_logger().info("Yellow line fit: gradient = " + str(yellow_fit[0]) + "y-intercept + " + str(yellow_fit[1]))
            except:
                print("Could not fit yellow line")
                #self.get_logger().info("Could not fit yellow line")
                
        else:
            print("Not enough yellow cones detected")
            #self.get_logger().info("Not enough yellow cones detected")

        # Fit a line to blue points if there are enough points
        if len(bx) > 1:
            try:
                blue_fit = np.polyfit(bx, by, 1)  # Linear fit (y = mx + b)
                blue_line = True
                #self.get_logger().info("Blue line fit: gradient = " + str(blue_fit[0]) + "y-intercept + " + str(blue_fit[1]))
            except:
                print("Could not fit blue line")
                #self.get_logger().info("Could not fit blue line")
        else:
            print("Not enough blue cones detected")
            #self.get_logger().info("Not enough blue cones detected")
        
        if yellow_line and blue_line:
            # Average the coefficients of the yellow and blue lines
            m = (yellow_fit[0] + blue_fit[0]) / 2
            b = (yellow_fit[1] + blue_fit[1]) / 2
        elif yellow_line:
            m = yellow_fit[0]
            b = yellow_fit[1] - ROAD_WIDTH / 2  # Offset calculation if only yellow line found
        elif blue_line:
            m = blue_fit[0]
            b = blue_fit[1] + ROAD_WIDTH / 2    # Offset calculation if only blue line found
        else:
            m, b = False, False

        if m and b:
            # Evaluate the distance-to-line at some look ahead distance in front of the car
            look_ahead_dist = 1000.0 # (Units: millimeters)
            distance_to_target_line = m * look_ahead_dist + b
            steering_action = P_STEERING * distance_to_target_line + STEER_OFFSET
            print(f"Dist-to-line = {distance_to_target_line:.2f}, steer action = {steering_action:.2f}")
            # self.get_logger().info(f"Steering action: {steering_action:.2f}")
            #esc_action = FIXED_ESC_VALUE
        else:
            print("Could not fit any line")
            #self.get_logger().info("Could not fit any line")
            steering_action = 0.0
            drive_command_clipped = 0.0
            #esc_action = 0.0
            # LOGIC FOR TERMINATION

        # =====================================
        # END OF SIMPLE POLICY
        # =====================================

        # Convert the [-100,100] percentage stering action
        # into a steering angle
        steering_action_clipped = max(-100.0, min(steering_action, 100.0))
        delta_request = steering_action_clipped * bicycle_model_parameters["delta_request_max"]
    


        # =====================================
        # END OF: INSERT POLICY CODE ABOVE HERE
        # =====================================



        # Construct the action vector expected by the gymnasium
        action = np.array([drive_command_clipped,delta_request], dtype=np.float32)

        # Return the action
        return action


## -------------------
#  PERFORM SIMULATIONS
#  -------------------

# Specify simulation length by:
# > Number of steps
N_sim = 400

# Create the policy
pid_policy = PIDPolicyForConeFollowing()

# Specify whether or not to save the look head results
# > Note that this is more data that all other parts of the results combined
should_save_look_ahead_results = False

# Run the simulation
sim_time_series_results = simulate_policy(env, N_sim, pid_policy, should_save_look_ahead_results, should_save_observations=True)



## ----------------
#  PLOT THE RESULTS
#  ----------------

# Specify a suffix for the plot file names
# > This is useful if the plot is called within a loop,
#   for example, the suffix could be the number of RL
#   training iterations performed thus far.
file_name_suffix = "cone_following_example"

# Specify a flag for whether to plot the reward or not
# > For some policy or policy synthesis method it is
#   NOT convenient to use the reward coming from the
#   environment.
#   - This is usually the case when the reward function
#     is explicitly a (hyper-) parameter for the policy
#     synthesis method.
#   - For example, the reward function is directly used
#     in MPC, and hence the per-time-step reward values
#     from the environment are not necessary for MPC
#     (and hence the reward values from the environment
#     are often out-of-sync with the reward function
#     bring used for MPC).
should_plot_reward = True

# Call the plotting function
plot_details_list = plot_results_from_time_series_dict(env, sim_time_series_results, path_for_saving_figures, file_name_suffix, should_plot_reward)


## ------------------------------------------------
#  CREATE AN ANIMATION
#  ------------------------------------------------
if (True):
    ani_zoom_width=5
    ani_zoom_height=5
    ani = env.unwrapped.render_matplotlib_animation_of_trajectory(sim_time_series_results["px"], sim_time_series_results["py"], sim_time_series_results["theta"], sim_time_series_results["delta"], numerical_integration_parameters["Ts"], traj_increment=3, figure_title="Animation of car trajectory", zoom_width=ani_zoom_width, zoom_height=ani_zoom_height)

    ani.save(path_for_saving_figures + '/ad_animation.gif')
    print('Saved animation: ' + path_for_saving_figures + '/ad_animation.gif')

    #from IPython.display import HTML
    #HTML(ani.to_jshtml())


## ------------------------------------------------
#  PLOT THE RESULTS - TIME SERIES
#  ------------------------------------------------

# Open the figure
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.94,"bottom":0.05})

fig.set_size_inches(7, 8)

this_ax_idx = -1

# Plot the ...
this_ax_idx = this_ax_idx + 1
this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["obs_distance_to_closest_point"])
# > Add the legend entry
this_line.set_label("dist to closest")
legend_lines = []
legend_lines.append(this_line)
# Set the labels:
#axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
axs[this_ax_idx].set_ylabel('[meters]', fontsize=10)
# Add grid lines
axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
# Show a legend
axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)


# Plot the ...
this_ax_idx = this_ax_idx + 1
this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["obs_heading_angle_relative_to_line"]*(180.0/np.pi))
# > Add the legend entry
this_line.set_label("heading angle to line")
legend_lines = []
legend_lines.append(this_line)
# Set the labels:
#axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
axs[this_ax_idx].set_ylabel('[degrees]', fontsize=10)
# Add grid lines
axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
# Show a legend
axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)

# Plot the ...
this_ax_idx = this_ax_idx + 1
this_line, = axs[this_ax_idx].plot(sim_time_series_results["time_in_seconds"],sim_time_series_results["obs_road_curvature_at_closest_point"])
# > Add the legend entry
this_line.set_label("curvature at closest")
legend_lines = []
legend_lines.append(this_line)
# Set the labels:
#axs[this_ax_idx].set_xlabel('time [seconds]', fontsize=10)
axs[this_ax_idx].set_ylabel('[1/meters]', fontsize=10)
# Add grid lines
axs[this_ax_idx].grid(visible=True, which="both", axis="both", linestyle='--')
# Show a legend
axs[this_ax_idx].legend(handles=legend_lines, loc="upper right", ncol=1, labelspacing=0.1)

fig.savefig(path_for_saving_figures + '/ad_obs_time_series.pdf')
print('Saved figure: ' + path_for_saving_figures + '/ad_obs_time_series.pdf')
