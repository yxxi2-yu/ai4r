#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from evaluation.evaluation_for_autonomous_driving import plot_road_from_list_of_road_elements
from ai4rgym.envs.road import Road



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
    {"type":"straight", "length":2.0},
    {"type":"curved", "curvature":1/2.0, "angle_in_degrees":150.0},
    {"type":"straight", "length":2.0},
    {"type":"curved", "curvature":-1/2.0, "angle_in_degrees":120.0},
    {"type":"straight", "length":2.0},
    {"type":"curved", "curvature":-1/2.0, "angle_in_degrees":90.0},
    {"type":"straight", "length":4.0},
    {"type":"curved", "curvature":1/2.0, "angle_in_degrees":60.0},
    {"type":"straight", "length":2.0},
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




## -------------------------------------------------------
#  PLOT THE VISIBLE CONES IN WORLD FRAME AND IN BODY FRAME
#  -------------------------------------------------------

# Specify the cone placement parameters
width_btw_cones = 1.0
mean_length_btw_cones = 0.5
stddev_of_length_btw_cones = 0.00

# Initialize the road
road = Road(road_elements_list=road_elements_list)

# Call the function to generate cone locations
road.generate_cones(width_btw_cones, mean_length_btw_cones, stddev_of_length_btw_cones)

# Open the figure
fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

# Auto-scale the axis limits
axs.set_xlim(auto=True)
axs.set_ylim(auto=True)

# Call the function to render the road
road_handles = road.render_road(axs)

# Plot the cones
if True:
    # > Get the cones coordinates
    cones_left_side_coords  = road.get_cones_left_side()
    cones_right_side_coords = road.get_cones_right_side()
    # > Plot the left-side cones in yellow
    cone_handles_left_side  = axs.scatter(x=cones_left_side_coords[:,0],  y=cones_left_side_coords[:,1],  s=8.0, marker="o", facecolor="y", alpha=1.0, linewidths=0, edgecolors="k")
    # > Plot the right-side cones in blue
    cone_handles_right_side = axs.scatter(x=cones_right_side_coords[:,0], y=cones_right_side_coords[:,1], s=8.0, marker="o", facecolor="b", alpha=1.0, linewidths=0, edgecolors="k")


# Get the cone info
# > Specify the details of the location from which to detect cones
px_car = 2.0
py_car = 4.0
theta_car = 135.0 * (np.pi/180.0)
camera_fov = 80.0
detection_distance_limit = 4.0
# > Call the detection function
cone_info = road.cone_info_at_given_pose_and_fov(px_car, py_car, theta_car, fov_horizontal_degrees=camera_fov, body_x_upper_bound=detection_distance_limit)

print(f"num_cones = {cone_info["num_cones"]}")

# Plot the detected cones
# > For the left side
x_for_left_cones = cone_info["px"][cone_info["side_of_road"]<0.0]
y_for_left_cones = cone_info["py"][cone_info["side_of_road"]<0.0]
cone_detections_left_side  = axs.scatter(x=x_for_left_cones,  y=y_for_left_cones,  s=16.0, marker="x", facecolor="k", alpha=1.0, linewidths=1.0)
# > For the right side
x_for_right_cones = cone_info["px"][cone_info["side_of_road"]>0.0]
y_for_right_cones = cone_info["py"][cone_info["side_of_road"]>0.0]
cone_detections_left_side  = axs.scatter(x=x_for_right_cones,  y=y_for_right_cones,  s=16.0, marker="*", facecolor="k", alpha=1.0, linewidths=1.0)

# Ensure the aspect ratio stays as 1:1
axs.set_aspect('equal', adjustable='box')

# Add a title
fig.suptitle('The road with visible cones', fontsize=12)
# Save the figure
path_and_file_name = path_for_saving_figures + "/" + "ad_road" + "_" + "with_visible_cones" + ".pdf"
fig.savefig(path_and_file_name)
print("Saved figure: " + path_and_file_name)



# PLOTTING CONES IN BODY FRAME

# Open the figure
fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

# Auto-scale the axis limits
axs.set_xlim(auto=True)
axs.set_ylim(auto=True)

# Plot the detected cones
# > For the left side
x_for_left_cones = cone_info["px_in_body_frame"][cone_info["side_of_road"]<0.0]
y_for_left_cones = cone_info["py_in_body_frame"][cone_info["side_of_road"]<0.0]
cone_detections_left_side  = axs.scatter(x=x_for_left_cones,  y=y_for_left_cones,  s=16.0, marker="x", facecolor="k", alpha=1.0, linewidths=1.0)
# > For the right side
x_for_right_cones = cone_info["px_in_body_frame"][cone_info["side_of_road"]>0.0]
y_for_right_cones = cone_info["py_in_body_frame"][cone_info["side_of_road"]>0.0]
cone_detections_left_side  = axs.scatter(x=x_for_right_cones,  y=y_for_right_cones,  s=16.0, marker="*", facecolor="k", alpha=1.0, linewidths=1.0)

# Ensure the aspect ratio stays as 1:1
axs.set_aspect('equal', adjustable='box')

# Add a title
fig.suptitle('The visible cones in the body frame', fontsize=12)
# Save the figure
path_and_file_name = path_for_saving_figures + "/" + "ad" + "_" + "visible_cones_in_body_frame" + ".pdf"
fig.savefig(path_and_file_name)
print("Saved figure: " + path_and_file_name)

