#!/usr/bin/env python

import numpy as np

class PIDPolicyForAutonomousDriving:
    """
    This class implements a controller for Autonomus Driving
    that is a combination of a:
        - A PID (Proportional, Intergral, Derivative) controller for
          controlling the vehicle's speed using the drive command.
        - Rules adjusting the speed reference based on road curvature.
        - A PID (Proportional, Intergral, Derivative) controller for
          tracking the road using the steering.
        - Rules for adjusting the PID gains of the tracking controller
          based on the actual speed of the car, which is sometime called
          "gain scheduling".

    Class variables:
    - integral_of_speed_error   :  integral of the speed error over time (units: m)
    - integral_of_offset_error  :  integral of the offset error over time (units: ms)
    """

    def __init__(self):
        """
        Initialization function for the "PIDPolicyForAutonomousDriving" class.
        """
        self.integral_of_speed_error = 0.0
        self.integral_of_offset_error = 0.0


    def compute_action(self, observation, info_dict, run_terminated):
        # Get the "info_dict" observation of the curvature at the closest point on the road
        curvature_at_closest = observation["road_curvature_at_closest_point"][0]
        # Get the "info_dict" observation of the curvature at the progress queries to the line
        look_ahead_curvature = observation["look_ahead_road_curvatures"][0]
        
        if (run_terminated):
            # Zero speed reference after reaching the end of the road
            speed_ref = 0.0
        else:
            # Speed reference relative to look-ahead curvature
            if np.isnan(look_ahead_curvature):
                speed_ref = 20.0/3.6
            else:
                curvature_for_speed_ref = abs(look_ahead_curvature)
                if ( abs(curvature_at_closest) >= abs(look_ahead_curvature) ):
                    curvature_for_speed_ref = abs(curvature_at_closest)

                speed_ref = 60.0/3.6 - curvature_for_speed_ref * (50.0) * (40/3.6)
                speed_ref = max(speed_ref,20.0/3.6)

        # Get the "info_dict" observation of the distance to the line
        #closest_distance = info_dict["closest_distance"]
        #side_of_the_road_line = info_dict["side_of_the_road_line"]
        dist_to_line = observation["distance_to_closest_point"][0]

        # Compute the speed of the car
        speed = np.sqrt( observation["ground_truth_vx"][0]**2 + observation["ground_truth_vy"][0]**2 ) * np.sign(observation["ground_truth_vx"][0])
        # Compute the error between the reference and actual speed
        speed_error = speed_ref - speed
        # Compute the drive command action
        drive_command_raw = 2.0 * speed_error
        # Clip the drive command action to be in the range [-100,100] percent
        drive_command_clipped = max(-100.0, min(drive_command_raw, 100.0))

        # Adjust the steering gain based on the speed
        #kp_steering = 2.0
        kp_steering = -(2.0 / (40.0/3.6)) * speed + 0.5 + ((2.0 / (40.0/3.6)) * (60.0/3.6))
        kp_steering = max( 0.5 , min( 4.0, kp_steering ) )

        # Compute the steering angle request action
        #delta_request = kp_steering*(np.pi/180.0) * closest_distance * (-side_of_the_road_line)
        delta_request = kp_steering*(np.pi/180.0) * (-dist_to_line)

        # Construct the action vector expected by the gymnasium
        action = np.array([drive_command_clipped,delta_request], dtype=np.float32)

        # Construct the action dictionary expected by the gymnasium
        #action = {
        #    "drive_command" : drive_command_clipped,
        #    "delta_request" : delta_request
        #}

        # Zero input if go too fast
        if (speed > 50.0):
            action[0] = 0.0
            action[1] = 0.0

        # Zero steering and drive after reaching the end of the road
        if (run_terminated):
            action[1] = 0.0

        # Return the action
        return action

