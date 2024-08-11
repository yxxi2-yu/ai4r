#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def plot_closest_point_for_whole_space(road, plot_path_and_name, grid_spacing=None):

    print("[UNIT TESTING] Starting test for computing the road's closest point for the whole space")

    # Open a figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Initialize a list for the legend
    #legend_lines = []

    # Render the road
    road.render_road(axs)

    # Get the extent of the road from the plot area
    x_min, x_max = axs.get_xlim()
    y_min, y_max = axs.get_ylim()

    # Compute the extent for testing
    x_test_min = x_min - 0.2 * (x_max - x_min)
    x_test_max = x_max + 0.2 * (x_max - x_min)
    y_test_min = y_min - 0.2 * (y_max - y_min)
    y_test_max = y_max + 0.2 * (y_max - y_min)

    # Check if the "grid_spacing" argument was supplied
    if (grid_spacing is not None):
        # Get the step for gridding
        grid_test_step = grid_spacing
    else:
        # Compute the step for gridding
        x_step_nominal = (x_test_max-x_test_min) / 50.0
        y_step_nominal = (x_test_max-x_test_min) / 50.0

        # Compute the minimum of the two
        grid_test_step = min(x_step_nominal, y_step_nominal)

    # Create the grid space for testing
    x_grid_test = np.arange(start=x_test_min, stop=x_test_max, step=grid_test_step, dtype=np.float32)
    y_grid_test = np.arange(start=y_test_min, stop=y_test_max, step=grid_test_step, dtype=np.float32)

    # Initialize arrays for storing the results
    num_test_points = x_grid_test.size * y_grid_test.size
    plot_points_x = np.empty((2,num_test_points))
    plot_points_y = np.empty((2,num_test_points))

    print("[UNIT TESTING] Now computing the road's closest point for " + str(num_test_points) + " grid points across the whole space.")

    # Iterate through all the points
    this_point_idx = 0
    for x in x_grid_test:
        for y in y_grid_test:
            # Set the progress queries
            progress_queries = np.array([0.0], dtype=np.float32)
            # Get the road info for this point
            road_info = road.road_info_at_given_pose_and_progress_queries(px=x, py=y, theta=0.0, progress_queries=progress_queries)
            # Put the results into the array
            plot_points_x[0,this_point_idx] = road_info["px"]
            plot_points_x[1,this_point_idx] = road_info["px_closest"]
            plot_points_y[0,this_point_idx] = road_info["py"]
            plot_points_y[1,this_point_idx] = road_info["py_closest"]
            # Increment the counter
            this_point_idx = this_point_idx + 1

    print("[UNIT TESTING] Finished computing closest points.")

    print("[UNIT TESTING] Now plotting the results")
    # Plot all the lines from test point to closest points
    line_handles = axs.plot(plot_points_x, plot_points_y, color="blue", linewidth=0.2, linestyle="-", marker="o", markersize=1, markerfacecolor="k", markeredgewidth=0.0)

    # Set the labels:
    axs.set_xlabel('px [meters]', fontsize=10)
    axs.set_ylabel('py [meters]', fontsize=10)

    # Add grid lines
    axs.grid(visible=True, which="both", axis="both", linestyle='--')

    # Set the aspect ratio for equally scaled axes
    axs.set_aspect('equal', adjustable='box')

    # Show a legend
    #fig.legend(handles=legend_lines, loc="lower center", ncol=4, labelspacing=0.1)

    # Add an overall figure title
    fig.suptitle("Testing for the closest points on the road", fontsize=12)

    # Save the plot
    fig.savefig(plot_path_and_name)
    print("[UNIT TESTING] Finished plotting results, figure saved at: " + plot_path_and_name)



def look_ahead_is_valid(road, plot_path_and_name, grid_spacing=None):

    print("[UNIT TESTING] Starting test for checking that the look ahead points from a progress query are alway valid")

    # Open a figure
    fig, axs = plt.subplots(1, 1, sharex=False, sharey=False, gridspec_kw={"left":0.15, "right": 0.95, "top":0.92,"bottom":0.18})

    # Initialize a list for the legend
    #legend_lines = []

    # Render the road
    road_handles = road.render_road(axs)

    # Make the road line thicker
    for handle in road_handles:
        handle.set_linewidth(5.0)

    # Get the extent of the road from the plot area
    x_min, x_max = axs.get_xlim()
    y_min, y_max = axs.get_ylim()

    # Compute the extent for testing
    x_test_min = x_min - 0.2 * (x_max - x_min)
    x_test_max = x_max + 0.2 * (x_max - x_min)
    y_test_min = y_min - 0.2 * (y_max - y_min)
    y_test_max = y_max + 0.2 * (y_max - y_min)

    # Check if the "grid_spacing" argument was supplied
    if (grid_spacing is not None):
        # Get the step for gridding
        grid_test_step = grid_spacing
    else:
        # Compute the step for gridding
        x_step_nominal = (x_test_max-x_test_min) / 50.0
        y_step_nominal = (x_test_max-x_test_min) / 50.0

        # Compute the minimum of the two
        grid_test_step = min(x_step_nominal, y_step_nominal)

    # Create the grid space for testing
    x_grid_test = np.arange(start=x_test_min, stop=x_test_max, step=grid_test_step, dtype=np.float32)
    y_grid_test = np.arange(start=y_test_min, stop=y_test_max, step=grid_test_step, dtype=np.float32)

    # Initialize arrays for storing the results
    num_test_points = x_grid_test.size * y_grid_test.size
    plot_points_x = np.empty((2,num_test_points))
    plot_points_y = np.empty((2,num_test_points))

    print("[UNIT TESTING] Now computing the road's closest point for " + str(num_test_points) + " grid points across the whole space.")

    # Overall flag
    found_nan_somewhere = False

    # Specify the progress queries
    temp_increment = 10.0
    look_ahead_progress_queries = np.linspace(temp_increment , 100.0, num=10, endpoint=True)

    # Initialize arrays for storing the results
    num_look_ahead_points = look_ahead_progress_queries.size
    plot_points_x = np.empty((num_look_ahead_points,num_test_points))
    plot_points_y = np.empty((num_look_ahead_points,num_test_points))

    # Iterate through all the points
    this_point_idx = 0
    for x in x_grid_test:
        for y in y_grid_test:
            # Get the road info for this point
            road_info = road.road_info_at_given_pose_and_progress_queries(px=x, py=y, theta=0.0, progress_queries=look_ahead_progress_queries)

            this_points_contains_nan = np.any(np.isnan(road_info["road_points_in_body_frame"]))
            this_angles_contains_nan = np.any(np.isnan(road_info["road_angles_relative_to_body_frame"]))
            this_curvatures_contains_nan = np.any(np.isnan(road_info["curvatures"]))

            if (this_points_contains_nan):
                print("[UNIT TESTING] nan found in the \"points\" array for test coordinate (x,y) = ( " + str(x) + " , " + str(y) + " )")
                found_nan_somewhere = True
            if (this_angles_contains_nan):
                print("[UNIT TESTING] nan found in the \"angles\" array for test coordinate (x,y) = ( " + str(x) + " , " + str(y) + " )")
                found_nan_somewhere = True
            if (this_curvatures_contains_nan):
                print("[UNIT TESTING] nan found in the \"curvatures\" array for test coordinate (x,y) = ( " + str(x) + " , " + str(y) + " )")
                found_nan_somewhere = True

            # Transform the look-ahead points into world frame
            # > Rotation matrix
            theta_rotate = 0.0
            R_mat = np.array([[np.cos(theta_rotate),np.sin(theta_rotate)],[-np.sin(theta_rotate),np.cos(theta_rotate)]])
            # Translation vector
            this_p = np.array([[x],[y]], dtype=np.float32)
            # Transpose the road point
            road_points = np.transpose( road_info["road_points_in_body_frame"] )
            # Transform the look-ahead points
            look_ahead_point_in_world_frame = np.add( this_p , np.matmul( R_mat , road_points ) )

            # Put the results into the array
            plot_points_x[:,this_point_idx] = look_ahead_point_in_world_frame[0,:]
            plot_points_y[:,this_point_idx] = look_ahead_point_in_world_frame[1,:]
            # Increment the counter
            this_point_idx = this_point_idx + 1

    print("[UNIT TESTING] Finished checking the look ahead progress queries.")

    if not(found_nan_somewhere):
        print("[UNIT TESTING] Test PASSED.")
    else:
        print("[UNIT TESTING] Test FAILED.")

    print("[UNIT TESTING] Now plotting the results")
    # Plot all the look ahead points
    line_handles = axs.plot(plot_points_x, plot_points_y, color="red", linewidth=0, marker="o", markersize=1, markerfacecolor="red", markeredgewidth=0.0)

    # Set the labels:
    axs.set_xlabel('px [meters]', fontsize=10)
    axs.set_ylabel('py [meters]', fontsize=10)

    # Add grid lines
    axs.grid(visible=True, which="both", axis="both", linestyle='--')

    # Set the aspect ratio for equally scaled axes
    axs.set_aspect('equal', adjustable='box')

    # Show a legend
    #fig.legend(handles=legend_lines, loc="lower center", ncol=4, labelspacing=0.1)

    # Add an overall figure title
    fig.suptitle("Testing for the validity of roda look-ahead points", fontsize=12)

    # Save the plot
    fig.savefig(plot_path_and_name)
    print("[UNIT TESTING] Finished plotting results, figure saved at: " + plot_path_and_name)