#!/usr/bin/env python

import numpy as np

from ai4rgym.envs.road import Road
from unit_tests.road_unit_tests import plot_closest_point_for_whole_space
from unit_tests.road_unit_tests import look_ahead_is_valid



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
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":1/2000.0, "angle_in_degrees":225.0},
    {"type":"straight", "length":100.0},
    {"type":"curved", "curvature":-1/2000.0, "angle_in_degrees":225.0},
    {"type":"straight", "length":100.0},
]

road = Road(epsilon_c=1/10000, road_elements_list=road_elements_list)


## ------------------------------
#  TEST THE CLOSEST POINT MAPPING
#  ------------------------------

# Specify the path for saving the figure
closest_point_figure_path_and_name = path_for_saving_figures + "/road_test_of_closest_points.pdf"
plot_closest_point_for_whole_space(road, closest_point_figure_path_and_name, grid_spacing=None)


## ------------------------------
#  TEST THE LOOK AHEAD VALIDITY
#  ------------------------------

# Specify the path for saving the figure
look_ahead_validity_figure_path_and_name = path_for_saving_figures + "/road_test_of_look_ahead_validity.pdf"
look_ahead_is_valid(road, look_ahead_validity_figure_path_and_name, grid_spacing=None)
