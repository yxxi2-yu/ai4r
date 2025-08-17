#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ai4rgym.envs.bicycle_model_dynamic import BicycleModelDynamic
from ai4rgym.envs.road import Road

import gymnasium as gym
from gymnasium import spaces


class AutonomousDrivingEnv(gym.Env):
    """
    This class is a Gymnasium environment that simulates the
    interaction of:
    - A car (as modelled by a dynamic bicycle model).
    - Driving around on a road (as modelled by a sequnece of straight lines and
      circles).

    THE CAR:
        car : BicycleModelDynamic
            An instance of the dynamic bicycle model class.

    THE ROAD:
        road_env : RoadEnv
            An instance of the kinematic bicycle model class.

    INTEGRATION SPECIFICATIONS:
        integration_method : string
            The numerical integration method that is used when simulating the
            evolution of the car.
        integration_Ts : float
            The numerical integration time step that the environment evolves
            by each time the "step" function is called.
        integration_num_steps_per_Ts : int
            The number of step that "integration_Ts" is split up into.

    INITIAL CONDITION SPECIFICATIONS:
        px_init_min,    px_init_max    : float
        py_init_min,    py_init_max    : float
        theta_init_min, theta_init_max : float
        vx_init_min,    vx_init_max    : float
        vy_init_min,    vy_init_max    : float
        omega_init_min, omega_init_max : float
        delta_init_min, delta_init_max : float
            The minimum and maximum (i.e., lower and upper bound) to use when
            generating randomized initial conditions. Each element of the state
            is sampled independently from a uniform distribution between the
            minimum and maximum values. For a fixed (i.e not random) initial
            state, set both the minimum and maximum to the same value.

    DEFAULT PROGRESS QUERIES:
        progress_queries_default : numpy array, 1-dimensionsl
            The default values for progress-along-the-road, relative to
            the current position of the car, at which the road information
            should be generated.
            Units: meters

    DEFAULT ROAD CONDITION:
        default_road_condition : string
            Specifies the default for which parameter values to use for
            determining the tire forces that arise due to slippage at high
            speeds.
            Options: { "dry" , "wet" , "snow" , "ice"

    RENDERING:
        render_mode : string
            The style of rendering to perform
        figure : matplotlib.figure
            For matplotlib rendering
        axis : matplotlib.axis
            For matplotlib rendering
        car_handles : list of matplotlib handles
            List of handles to the plotted elements that render the car
        road_handles : list of matplotlib handles
            List of handles to the plotted elements that render the road
        
        To plot the road environment and car, you can use the following
        functions:
        - `render_matplotlib_init_figure()`
          This open a figure
        - `render_matplotlib_plot_road()`
          This plots the road
        - `render_matplotlib_plot_car(px,py,theta,delta)`
          This plots the car for the given pose values
        - `render_matplotlib_zoom_to(px,py,x_width,y_height)`
          This adjust the axis limits of the plot
    
    You can save the figure in the usual fashion for a matplotlb figure:
    - `env.figure.savefig('autonomous_driving.pdf')`
    """

    metadata = {
        "render_modes": ["matplotlib"],
        "integration_methods" : ["euler", "huen", "midpoint", "rk4", "rk45"],
    }

    def __init__(
        self,
        render_mode,
        bicycle_model_parameters,
        road_elements_list,
        numerical_integration_parameters,
        termination_parameters,
        initial_state_bounds,
        observation_parameters,
    ):
        """
        Initialization function for the "AutonomousDrivingGym" class.

        Parameters
        ----------
            render_mode : string
                Specifies how the visualization of the environment should be
                render as part of the "step" function.

            bicycle_model_parameters : dictionary
                Parameters necessary for initializing a BicycleModelDynamic
                object

            road_elements_list : list of dictionaries
                Specifies the road

            numerical_integration_parameters : dictionary
                Specifies the integration details used in the "step" function.
                This dictionary should contain the folllowing keys:
                - method : string
                    The numerical integration method that is used when simulating the
                    evolution of the car.
                - Ts : float
                    The numerical integration time step that the environment evolves
                    by each time the "step" function is called.
                - num_steps_per_Ts : int
                    The number of step that "integration_Ts" is split up into.

            termination_parameters : dictionary
                Specifies the bounds for when to raise the termination flag.
                This dictionary should contain the folllowing keys:
                - speed_lower_bound : float
                    Termination is flagged when the speed of the car drops below
                    this lower bound value. If this is set to zero, then there is
                    effecively no lower bound.
                - reward_for_speed_lower_bound : float
                    The reward when this termination criteria triggers.
                - speed_upper_bound : float
                    Termination is flagged when the speed of the car goes above
                    this upper bound value.
                - reward_for_speed_upper_bound : float
                    The reward when this termination criteria triggers.
                - distance_to_closest_point_upper_bound : float
                    Termination is flagged when the car deviates from the road line
                    by more thatn this upper bound value.
                - reward_for_distance_to_closest_point_upper_bound : float
                    The reward when this termination criteria triggers.

            initial_state_bounds : dictionary
                Specifies the minimum and maximum (i.e., lower and upper bounds) us for
                drawing each element of the initial state from independent uniform
                distributions.
                This dictionary should contain the folllowing keys:
                - px_init_min,    px_init_max    : float
                - py_init_min,    py_init_max    : float
                - theta_init_min, theta_init_max : float
                - vx_init_min,    vx_init_max    : float
                - vy_init_min,    vy_init_max    : float
                - omega_init_min, omega_init_max : float
                - delta_init_min, delta_init_max : float

            observation_parameters : dictionary
                Specifies the sensor measurements that sholud be included in the
                observations. Sensor meansurements not included in the observations
                are included in the "info_dict".
                This dictionary can contain the following keys:
                (Flags for which sensor measurements to include in the observations)
                - should_include_obs_for_ground_truth_state                    :  bool
                - should_include_obs_for_vx_sensor                             :  bool
                - should_include_obs_for_closest_distance_to_line              :  bool
                - should_include_obs_for_heading_angle_relative_to_line        :  bool
                - should_include_obs_for_heading_angle_gyro                    :  bool
                - should_include_obs_for_accel_in_body_frame_x                 :  bool
                - should_include_obs_for_accel_in_body_frame_y                 :  bool
                - should_include_obs_for_look_ahead_line_coords_in_body_frame  :  bool
                - should_include_obs_for_look_ahead_road_curvatures            :  bool
                - should_include_obs_for_road_progress_at_closest_point        :  bool
                - should_include_obs_for_road_curvature_at_closest_point       :  bool
                - should_include_obs_for_gps_line_coords_in_world_frame        :  bool
                - should_include_cone_detections                               :  bool

                (Scaling value for each sensor measurement)
                - scaling_for_ground_truth_state                    :  float
                - scaling_for_vx_sensor                             :  float
                - scaling_for_closest_distance_to_line              :  float
                - scaling_for_heading_angle_relative_to_line        :  float
                - scaling_for_heading_angle_gyro                    :  float
                - scaling_for_accel_in_body_frame_x                 :  float
                - scaling_for_accel_in_body_frame_y                 :  float
                - scaling_for_look_ahead_line_coords_in_body_frame  :  float
                - scaling_for_look_ahead_road_curvatures            :  float
                - scaling_for_road_progress_at_closes_point         :  float
                - scaling_for_road_curvature_at_closes_point        :  float
                - scaling_for_cone_detections                       :  float

                (Specifications for each sensor measurement)
                - vx_sensor_bias   : float
                - vx_sensor_stddev : float

                - closest_distance_to_line_bias    :  float
                - closest_distance_to_line_stddev  :  float

                - heading_angle_relative_to_line_bias    :  float
                - heading_angle_relative_to_line_stddev  :  float

                - heading_angle_gyro_bias    :  float
                - heading_angle_gyro_stddev  :  float

                - look_ahead_line_coords_in_body_frame_distance             :  float
                - look_ahead_line_coords_in_body_frame_num_points           :  float
                - look_ahead_line_coords_in_body_frame_stddev_lateral       :  float
                - look_ahead_line_coords_in_body_frame_stddev_longitudinal  :  float

                - cone_detections_width_btw_cones             :  float
                - cone_detections_mean_length_btw_cones       :  float
                - cone_detections_stddev_of_length_btw_cones  :  float

        Returns
        -------
        Nothing
        """

        # ACTION SPACE
        # > Actions are the:
        #   - Drive command (aka., force command) to the motor.
        #     Range [-100,100]
        #     Units: percent
        #   - Requested steering angle.
        #     Range between plus/minus car_parameters["delta_request_max"]
        #     Units: radians
        self.action_space = spaces.Box(
            low =np.array([-100.0, -bicycle_model_parameters["delta_request_max"]]),
            high=np.array([ 100.0,  bicycle_model_parameters["delta_request_max"]]),
            shape=(2,), dtype=np.float32
        )

        # > For readability, would we use a dictionary,...
        #   HOWEVER, action space dictionaries are not compatible with
        #   the Stable Baselines 3 RL training library.
        #self.action_space = spaces.Dict(
        #    {
        #        "drive_command": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
        #        "delta_request": spaces.Box(low=-bicycle_model_parameters["delta_request_max"], high=bicycle_model_parameters["delta_request_max"], shape=(1,), dtype=np.float32),
        #   }
        #)

        # To still provide some readability for the action space box:
        self.action_space_labels = ["Drive command","Requested steering angle"]

        # THE CAR:
        # Create an instance of the bicycle model
        self.car = BicycleModelDynamic(bicycle_model_parameters)

        # THE ROAD:
        # Instantiate a RoadEnv object
        self.road = Road(epsilon_c=(1/10000), road_elements_list=road_elements_list)

        # Get the total road length into a variable for this class
        self.total_road_length = self.road.get_total_length()

        # A length slightly shorter than the total for more
        # reliably detecting termination
        self.total_road_length_for_termination = max( self.total_road_length-0.1 , 0.9999 * self.total_road_length )

        # INTEGRATION SPECIFICATIONS:
        # Set the numerical integration parameters
        self.integration_method = numerical_integration_parameters["method"]
        self.integration_Ts = numerical_integration_parameters["Ts"]
        self.integration_num_steps_per_Ts =  1  if ("num_steps_per_Ts" not in numerical_integration_parameters) else numerical_integration_parameters["num_steps_per_Ts"]

        # Termination SPECIFICATIONS:
        # Set the termination parameters
        self.termination_speed_lower_bound  =  0.0          if ("speed_lower_bound" not in termination_parameters) else termination_parameters["speed_lower_bound"]
        self.termination_speed_upper_bound  =  (200.0/3.6)  if ("speed_upper_bound" not in termination_parameters) else termination_parameters["speed_upper_bound"]
        self.termination_distance_to_closest_point_upper_bound  = 20.0  if ("distance_to_closest_point_upper_bound" not in termination_parameters) else termination_parameters["distance_to_closest_point_upper_bound"]

        self.termination_reward_for_speed_lower_bound                      =  0.0  if ("reward_for_speed_lower_bound"                     not in termination_parameters) else termination_parameters["reward_for_speed_lower_bound"]
        self.termination_reward_for_speed_upper_bound                      =  0.0  if ("reward_for_speed_upper_bound"                     not in termination_parameters) else termination_parameters["reward_for_speed_upper_bound"]
        self.termination_reward_for_distance_to_closest_point_upper_bound  =  0.0  if ("reward_for_distance_to_closest_point_upper_bound" not in termination_parameters) else termination_parameters["reward_for_distance_to_closest_point_upper_bound"]



        # INITIAL CONDITION SPECIFICATIONS:
        # Set the initial condition bound
        self.px_init_min    = initial_state_bounds["px_init_min"]
        self.px_init_max    = initial_state_bounds["px_init_max"]
        self.py_init_min    = initial_state_bounds["py_init_min"]
        self.py_init_max    = initial_state_bounds["py_init_max"]
        self.theta_init_min = initial_state_bounds["theta_init_min"]
        self.theta_init_max = initial_state_bounds["theta_init_max"]
        self.vx_init_min    = initial_state_bounds["vx_init_min"]
        self.vx_init_max    = initial_state_bounds["vx_init_max"]
        self.vy_init_min    = initial_state_bounds["vy_init_min"]
        self.vy_init_max    = initial_state_bounds["vy_init_max"]
        self.omega_init_min = initial_state_bounds["omega_init_min"]
        self.omega_init_max = initial_state_bounds["omega_init_max"]
        self.delta_init_min = initial_state_bounds["delta_init_min"]
        self.delta_init_max = initial_state_bounds["delta_init_max"]



        # OBSERVATION PARAMETERS
        # Speficify the default values of the observation parameters
        # > Flags for which sensor measurements to include in the observations or info dict
        # > Scaling value for each sensor measurement
        # > Specifications for each sensor measurement
        observation_parameters_defaults = {
            "should_include_ground_truth_px"                       :  "info",
            "should_include_ground_truth_py"                       :  "info",
            "should_include_ground_truth_theta"                    :  "info",
            "should_include_ground_truth_vx"                       :  "info",
            "should_include_ground_truth_vy"                       :  "info",
            "should_include_ground_truth_omega"                    :  "info",
            "should_include_ground_truth_delta"                    :  "info",
            "should_include_road_progress_at_closest_point"        :  "info",
            "should_include_vx_sensor"                             :  "info",
            "should_include_distance_to_closest_point"             :  "obs",
            "should_include_heading_angle_relative_to_line"        :  "obs",
            "should_include_heading_angular_rate_gyro"             :  "info",
            "should_include_accel_in_body_frame_x"                 :  "neither",
            "should_include_accel_in_body_frame_y"                 :  "neither",
            "should_include_closest_point_coords_in_body_frame"    :  "info",
            "should_include_look_ahead_line_coords_in_body_frame"  :  "info",
            "should_include_road_curvature_at_closest_point"       :  "obs",
            "should_include_look_ahead_road_curvatures"            :  "info",
            "should_include_gps_line_coords_in_world_frame"        :  "neither",
            "should_include_cone_detections"                       :  "neither",

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
            "scaling_for_accel_in_body_frame_x"                 :  1.0,
            "scaling_for_accel_in_body_frame_y"                 :  1.0,
            "scaling_for_closest_point_coords_in_body_frame"    :  1.0,
            "scaling_for_look_ahead_line_coords_in_body_frame"  :  1.0,
            "scaling_for_road_curvature_at_closest_point"       :  1.0,
            "scaling_for_look_ahead_road_curvatures"            :  1.0,
            "scaling_for_gps_line_coords_in_world_frame"        :  1.0,
            "scaling_for_cone_detections"                       :  1.0,

            "vx_sensor_bias"   : 0.0,
            "vx_sensor_stddev" : 0.0, #0.1

            "distance_to_closest_point_bias"    :  0.0,
            "distance_to_closest_point_stddev"  :  0.00, #0.01

            "heading_angle_relative_to_line_bias"    :  0.0,
            "heading_angle_relative_to_line_stddev"  :  0.00, #0.01

            "heading_angular_rate_gyro_bias"    :  0.0,
            "heading_angular_rate_gyro_stddev"  :  0.0,

            "closest_point_coords_in_body_frame_bias"    :  0.0,
            "closest_point_coords_in_body_frame_stddev"  :  0.0,

            "look_ahead_line_coords_in_body_frame_bias"    :  0.0,
            "look_ahead_line_coords_in_body_frame_stddev"  :  0.0,

            "road_curvature_at_closest_point_bias"    :  0.0,
            "road_curvature_at_closest_point_stddev"  :  0.0,

            "look_ahead_road_curvatures_bias"    :  0.0,
            "look_ahead_road_curvatures_stddev"  :  0.0,

            "look_ahead_line_coords_in_body_frame_distance"             :  100.0,
            "look_ahead_line_coords_in_body_frame_num_points"           :  10,
            "look_ahead_line_coords_in_body_frame_stddev_lateral"       :  0.0,
            "look_ahead_line_coords_in_body_frame_stddev_longitudinal"  :  0.0,

            "cone_detections_width_btw_cones"             :  1.0,
            "cone_detections_mean_length_btw_cones"       :  0.5,
            "cone_detections_stddev_of_length_btw_cones"  :  0.00,

            "cone_detections_fov_horizontal_degrees"         :  80.0,
            "cone_detections_body_x_upper_bound"             :  4.0,
            "cone_detections_stddev_of_detection_in_body_x"  :  0.00,
            "cone_detections_stddev_of_detection_in_body_y"  :  0.00,
        }


        # Set the observation parameters that are provided by the input argument
        # Update defaults with provided params
        processed_observation_parameters = {key: observation_parameters.get(key, default) for key, default in observation_parameters_defaults.items()}

        # Put every observation parameter into a class variable using the
        # key as the name for the class variable
        for key, value in processed_observation_parameters.items():
            setattr(self, key, value)



        # OBSERVATION NAMES AND BOX-SPACE SPECIFICATIONS
        # Specify the names for observations
        observations_name_low_high_shape = [
            ["ground_truth_px"                      , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_py"                      , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_theta"                   , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_vx"                      , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_vy"                      , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_omega"                   , -np.inf    , np.inf    , (1,) ],
            ["ground_truth_delta"                   , -0.5*np.pi , 0.5*np.pi , (1,) ],
            ["road_progress_at_closest_point"       , -np.inf    , np.inf    , (1,) ],
            ["vx_sensor"                            , -np.inf    , np.inf    , (1,) ],
            ["distance_to_closest_point"            , -np.inf    , np.inf    , (1,) ],
            ["heading_angle_relative_to_line"       , -np.pi     , np.pi     , (1,) ],
            ["heading_angular_rate_gyro"            , -np.inf    , np.inf    , (1,) ],
            ["accel_in_body_frame_x"                , -np.inf    , np.inf    , (1,) ],
            ["accel_in_body_frame_y"                , -np.inf    , np.inf    , (1,) ],
            ["closest_point_coords_in_body_frame"   , -np.inf    , np.inf    , (2,) ],
            ["look_ahead_line_coords_in_body_frame" , -np.inf    , np.inf    , (2,self.look_ahead_line_coords_in_body_frame_num_points) ],
            ["road_curvature_at_closest_point"      , -np.inf    , np.inf    , (1,) ],
            ["look_ahead_road_curvatures"           , -np.inf    , np.inf    , (self.look_ahead_line_coords_in_body_frame_num_points,) ],
            ["gps_line_coords_in_world_frame"       , -np.inf    , np.inf    , (1,) ]
        ]



        # OBSERVATION SPACE
        # Construct a normal dictionary for the sensor measurements
        # that are to be included as observations.

        # > Initialise an empty dictionary
        obs_space_dict = {}
        self.obs_dict_blank = {}
        self.info_dict_blank = {}

        # Iterate over the observation names
        for obs_details in observations_name_low_high_shape:
            # Extract the specs of this observation
            obs_name  = obs_details[0]
            obs_low   = obs_details[1]
            obs_high  = obs_details[2]
            obs_shape = obs_details[3]
            # Get the "should include" string for this name
            should_include_string = processed_observation_parameters.get("should_include_"+obs_name)
            # Process for should include in observation
            if should_include_string.lower() == "obs":
                setattr(self, "should_include_obs_for_"+obs_name, True)
                obs_space_dict.update({obs_name: spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32)})
                self.obs_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.float32)})
            else:
                setattr(self, "should_include_obs_for_"+obs_name, False)
            # Process for should include in info dict
            if should_include_string.lower() == "info":
                setattr(self, "should_include_info_for_"+obs_name, True)
                self.info_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.float32)})
            else:
                setattr(self, "should_include_info_for_"+obs_name, False)



        # OBSERVATION OF CONE DETECTIONS
        # Note: This needs to be processed separately because there are multiple
        # observation spaces created by the one "should_include" flag.

        # Convert the "should_include_cone_detections" string to booleans
        # > for observation
        if self.should_include_cone_detections.lower() == "obs":
            self.should_include_obs_for_cone_detections = True
        else:
            self.should_include_obs_for_cone_detections = False
        # > For info:
        if self.should_include_cone_detections.lower() == "info":
            self.should_include_info_for_cone_detections = True
        else:
            self.should_include_info_for_cone_detections = False

        # Generate the cones for the road
        if (self.should_include_obs_for_cone_detections or self.should_include_info_for_cone_detections):
            self.road.generate_cones(self.cone_detections_width_btw_cones, self.cone_detections_mean_length_btw_cones, self.cone_detections_stddev_of_length_btw_cones)

        # > Compute a reasonable upper bound on the number of cones that could be detected
        self.num_cones_max = int(3.0 * (1.0 + self.cone_detections_body_x_upper_bound / self.cone_detections_mean_length_btw_cones))
        # > Specify the names for each part of the cone detection observations:
        observations_name_low_high_shape_for_cone_detections = [
            ["cone_detections_x"                   , -np.inf    , np.inf             , (self.num_cones_max,)],
            ["cone_detections_y"                   , -np.inf    , np.inf             , (self.num_cones_max,)],
        ]

        # Iterate over the observation names
        for obs_details in observations_name_low_high_shape_for_cone_detections:
            # Extract the specs of this observation
            obs_name  = obs_details[0]
            obs_low   = obs_details[1]
            obs_high  = obs_details[2]
            obs_shape = obs_details[3]

            # Process for should include in observation
            if self.should_include_obs_for_cone_detections:
                obs_space_dict.update({obs_name: spaces.Box(low=obs_low, high=obs_high, shape=obs_shape, dtype=np.float32)})
                self.obs_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.float32)})
            # Process for should include in info dict
            if self.should_include_info_for_cone_detections:
                self.info_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.float32)})

        # Add the observation for the side of the road
        obs_name = "cone_detections_side_of_road"
        obs_shape = (self.num_cones_max,)
        if self.should_include_obs_for_cone_detections:
            obs_space_dict.update({obs_name: spaces.MultiDiscrete(nvec=[3]*self.num_cones_max, dtype=np.int32, start=[0]*self.num_cones_max)})
            self.obs_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.int32)})
        if self.should_include_info_for_cone_detections:
            self.info_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.int32)})

        # Add the observation for the number of cones
        obs_name = "cone_detections_num_cones"
        obs_shape = (1,)
        if self.should_include_obs_for_cone_detections:
            obs_space_dict.update({obs_name: spaces.MultiDiscrete(nvec=[self.num_cones_max+1], dtype=np.int32, start=[0])})
            self.obs_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.int32)})
        if self.should_include_info_for_cone_detections:
            self.info_dict_blank.update({obs_name: np.zeros(obs_shape, dtype=np.int32)})



        # > Create the observation space
        self.observation_space = spaces.Dict(obs_space_dict)


        # LOOK-AHEAD QUERIES
        # Based on the observation parameters, construct the progress queries
        # that are used for getting look-ahead details about the line.
        temp_increment = self.look_ahead_line_coords_in_body_frame_distance / self.look_ahead_line_coords_in_body_frame_num_points
        self.look_ahead_progress_queries = np.linspace(temp_increment , self.look_ahead_line_coords_in_body_frame_distance, num=(self.look_ahead_line_coords_in_body_frame_num_points), endpoint=True)

        # PROGRESS TRACKING
        # Initialize the "previous progress" variable to zero
        # > It is set to the correct value in the reset function
        self.previous_progress_at_closest_p = 0.0

        # GROUND TRUTH RECORDING
        # Initialize the "current_ground_truth" dictionary
        # > This is used to ensure access to these quantities
        #   without noise or scaling
        self.current_ground_truth = {
            "px"     :  0.0,
            "py"     :  0.0,
            "theta"  :  0.0,
            "vx"     :  0.0,
            "vy"     :  0.0,
            "omega"  :  0.0,
            "delta"  :  0.0,

            "road_progress_at_closest_point"   :  0.0,
            "distance_to_closest_point"        :  0.0,
            "heading_angle_relative_to_line"   :  0.0,
            "road_curvature_at_closest_point"  :  0.0,

            "px_closest"  :  0.0,
            "py_closest"  :  0.0,

            "px_closest_in_body_frame"  :  0.0,
            "py_closest_in_body_frame"  :  0.0,

            "look_ahead_line_coords_in_body_frame"  :  0.0,
            "look_ahead_road_curvatures"            :  0.0,
        }

        # PROGRESS QUERIES:
        # Set the progress queries used for filling
        # the "info_dict" with look-ahead observations.
        #self.progress_queries = np.array([0.0,1.0,2.0,3.0,4.0,5.0], dtype=np.float32)

        # ROAD CONDITION:
        # Set the road condition
        self.road_condition = "dry"
        # Pacejka tire model parameters, only used
        # when the road condition is "other"
        self.D_Pacejka = None
        self.C_Pacejka = None
        self.B_Pacejka = None
        self.E_Pacejka = None

        # RENDERING:
        # Check that the requested render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Variables for the matplotlib display
        self.figure = None
        self.axis   = None  
        self.car_handles = []



    def _get_observation_and_info_and_update_ground_truth(self):
        """
        Puts the current state of the car in a dictionary

        Parameters
        ----------
        Nothing

        Returns
        -------
        dictionary
            Containing keys for each state of the car
        """
        # Initialise empty dictionaries
        obs_dict  = self.obs_dict_blank.copy()
        info_dict = self.info_dict_blank.copy()

        # Get the road info for the current pose of the car
        # and at the "look_ahead_progress_queries"
        road_info_dict = self._get_road_info(self.look_ahead_progress_queries)
        #print(road_info_dict)

        # Get the cone info for the current pose of the car (if required)
        if (self.should_include_obs_for_cone_detections or self.should_include_info_for_cone_detections):
            cone_info = self.road.cone_info_at_given_pose_and_fov(px=self.car.px, py=self.car.py, theta=self.car.theta, fov_horizontal_degrees=self.cone_detections_fov_horizontal_degrees, body_x_upper_bound=self.cone_detections_body_x_upper_bound)
        else:
            cone_info = []

        # Update the dictionary of ground truth values
        self.current_ground_truth["px"]     =  self.car.px
        self.current_ground_truth["py"]     =  self.car.py
        self.current_ground_truth["theta"]  =  self.car.theta
        self.current_ground_truth["vx"]     =  self.car.vx
        self.current_ground_truth["vy"]     =  self.car.vy
        self.current_ground_truth["omega"]  =  self.car.omega
        self.current_ground_truth["delta"]  =  self.car.delta

        self.current_ground_truth["road_progress_at_closest_point"]   =  road_info_dict["progress_at_closest_p"]
        self.current_ground_truth["distance_to_closest_point"]        =  (road_info_dict["closest_distance"] * road_info_dict["side_of_the_road_line"])
        self.current_ground_truth["heading_angle_relative_to_line"]   =  road_info_dict["road_angle_relative_to_body_frame_at_closest_p"]
        self.current_ground_truth["road_curvature_at_closest_point"]  =  road_info_dict["curvature_at_closest_p"]

        self.current_ground_truth["px_closest"]  =  road_info_dict["px_closest"]
        self.current_ground_truth["py_closest"]  =  road_info_dict["py_closest"]

        self.current_ground_truth["px_closest_in_body_frame"]  =  road_info_dict["px_closest_in_body_frame"]
        self.current_ground_truth["py_closest_in_body_frame"]  =  road_info_dict["py_closest_in_body_frame"]

        self.current_ground_truth["look_ahead_line_coords_in_body_frame"]  =  road_info_dict["road_points_in_body_frame"]
        self.current_ground_truth["look_ahead_road_curvatures"]            =  road_info_dict["curvatures"]

        # Compute the measurement values
        road_progress_at_closest_point = road_info_dict["progress_at_closest_p"] * self.scaling_for_road_progress_at_closest_point

        vx_noise = np.random.normal(self.vx_sensor_bias, self.vx_sensor_stddev)
        vx_sensor = (self.car.vx + vx_noise) * self.scaling_for_vx_sensor

        dist_to_line_noise = np.random.normal(self.distance_to_closest_point_bias, self.distance_to_closest_point_stddev)
        dist_to_line = (road_info_dict["closest_distance"] * road_info_dict["side_of_the_road_line"] + dist_to_line_noise) * self.scaling_for_distance_to_closest_point

        heading_rel_to_line_noise = np.random.normal(self.heading_angle_relative_to_line_bias, self.heading_angle_relative_to_line_stddev)
        heading_rel_to_line = (road_info_dict["road_angle_relative_to_body_frame_at_closest_p"] + heading_rel_to_line_noise) * self.scaling_for_heading_angle_relative_to_line

        heading_gyro_noise = np.random.normal(self.heading_angular_rate_gyro_bias, self.heading_angular_rate_gyro_stddev)
        heading_gyro = (self.car.omega + heading_gyro_noise) * self.scaling_for_heading_angular_rate_gyro

        # Compute body-frame accelerations if any accel observation/info is requested
        accel_x_meas = None
        accel_y_meas = None
        if (
            self.should_include_obs_for_accel_in_body_frame_x or self.should_include_info_for_accel_in_body_frame_x or
            self.should_include_obs_for_accel_in_body_frame_y or self.should_include_info_for_accel_in_body_frame_y
        ):
            ax, ay = self.car.compute_body_frame_acceleration(
                road_condition=self.road_condition,
                Dp=self.D_Pacejka, Cp=self.C_Pacejka, Bp=self.B_Pacejka, Ep=self.E_Pacejka
            )
            accel_x_meas = ax * self.scaling_for_accel_in_body_frame_x
            accel_y_meas = ay * self.scaling_for_accel_in_body_frame_y

        px_noise = np.random.normal(self.closest_point_coords_in_body_frame_bias, self.closest_point_coords_in_body_frame_stddev)
        py_noise = np.random.normal(self.closest_point_coords_in_body_frame_bias, self.closest_point_coords_in_body_frame_stddev)
        px_closest_in_body_frame = (road_info_dict["px_closest_in_body_frame"] + px_noise) * self.scaling_for_closest_point_coords_in_body_frame
        py_closest_in_body_frame = (road_info_dict["py_closest_in_body_frame"] + py_noise) * self.scaling_for_closest_point_coords_in_body_frame

        look_ahead_line_coords_noise = np.array( np.random.normal(self.look_ahead_line_coords_in_body_frame_bias, self.look_ahead_line_coords_in_body_frame_stddev, (2,self.look_ahead_line_coords_in_body_frame_num_points)) , dtype=np.float32)
        look_ahead_line_coords_in_body_frame = (np.transpose( np.array(road_info_dict["road_points_in_body_frame"], dtype=np.float32)) + look_ahead_line_coords_noise) * self.scaling_for_look_ahead_line_coords_in_body_frame

        road_curvature_noise = np.random.normal(self.road_curvature_at_closest_point_bias, self.road_curvature_at_closest_point_stddev)
        road_curvature_at_closest_point = (road_info_dict["curvature_at_closest_p"] + road_curvature_noise) * self.scaling_for_road_curvature_at_closest_point

        look_ahead_road_curvatures_noise = np.array( np.random.normal(self.look_ahead_road_curvatures_bias, self.look_ahead_road_curvatures_stddev, (self.look_ahead_line_coords_in_body_frame_num_points,)) , dtype=np.float32)
        look_ahead_road_curvatures = (np.array(road_info_dict["curvatures"], dtype=np.float32) + look_ahead_road_curvatures_noise) * self.scaling_for_look_ahead_road_curvatures

        if (self.should_include_obs_for_cone_detections or self.should_include_info_for_cone_detections):
            # > Get the number of cones
            num_cones = cone_info["num_cones"]
            # > Prepare observation placeholder for zero cones
            if (num_cones == 0):
                cone_detections_x = np.zeros((self.num_cones_max,), dtype=np.float32)
                cone_detections_y = np.zeros((self.num_cones_max,), dtype=np.float32)
                cone_detections_side_of_road = np.full(shape=(self.num_cones_max,), fill_value=2, dtype=np.int32)
                num_cones_in_obs = 0
            # > Prepare observation for more than the maximum allowed size of the observation
            elif (num_cones > self.num_cones_max):
                # > Sample the noise
                cone_detections_x_noise = np.array( np.random.normal(0.0, self.cone_detections_stddev_of_detection_in_body_x, (self.num_cones_max,)) , dtype=np.float32)
                cone_detections_y_noise = np.array( np.random.normal(0.0, self.cone_detections_stddev_of_detection_in_body_y, (self.num_cones_max,)) , dtype=np.float32)
                # > Build the observations
                cone_detections_x = (cone_info["px_in_body_frame"][0:self.num_cones_max] + cone_detections_x_noise) * self.scaling_for_cone_detections
                cone_detections_y = (cone_info["py_in_body_frame"][0:self.num_cones_max] + cone_detections_y_noise) * self.scaling_for_cone_detections
                cone_detections_side_of_road = cone_info["side_of_road"][0:self.num_cones_max]
                num_cones_in_obs = self.num_cones_max
                # > Update the left-hand side of road from -1 to 0
                cone_detections_side_of_road[cone_detections_side_of_road==-1] = 0
             # > Prepare observation for a "situation normal" detection
            else:
                # > Sample the noise
                cone_detections_x_noise = np.array( np.random.normal(0.0, self.cone_detections_stddev_of_detection_in_body_x, (num_cones,)) , dtype=np.float32)
                cone_detections_y_noise = np.array( np.random.normal(0.0, self.cone_detections_stddev_of_detection_in_body_y, (num_cones,)) , dtype=np.float32)
                # > Build the observations
                padding_length = self.num_cones_max - num_cones
                cone_detections_x = np.pad((cone_info["px_in_body_frame"]+cone_detections_x_noise) * self.scaling_for_cone_detections, (0, padding_length), mode='constant', constant_values=0.0)
                cone_detections_y = np.pad((cone_info["py_in_body_frame"]+cone_detections_y_noise) * self.scaling_for_cone_detections, (0, padding_length), mode='constant', constant_values=0.0)
                cone_detections_side_of_road = np.pad(cone_info["side_of_road"], (0, padding_length), mode='constant', constant_values=2)
                num_cones_in_obs = num_cones
                # > Update the left-hand side of road from -1 to 0
                cone_detections_side_of_road[cone_detections_side_of_road==-1] = 0



        # Put the measurements values into the appropriate dictionary
        if (self.should_include_obs_for_ground_truth_px):
            obs_dict["ground_truth_px"][0]  = self.car.px * self.scaling_for_ground_truth_px
        if (self.should_include_info_for_ground_truth_px):
            info_dict["ground_truth_px"][0] = self.car.px * self.scaling_for_ground_truth_px

        if (self.should_include_obs_for_ground_truth_py):
            obs_dict["ground_truth_py"][0]  = self.car.py * self.scaling_for_ground_truth_py
        if (self.should_include_info_for_ground_truth_py):
            info_dict["ground_truth_py"][0] = self.car.py * self.scaling_for_ground_truth_py

        if (self.should_include_obs_for_ground_truth_theta):
            obs_dict["ground_truth_theta"][0]  = self.car.theta * self.scaling_for_ground_truth_theta
        if (self.should_include_info_for_ground_truth_theta):
            info_dict["ground_truth_theta"][0] = self.car.theta * self.scaling_for_ground_truth_theta

        if (self.should_include_obs_for_ground_truth_vx):
            obs_dict["ground_truth_vx"][0]  = self.car.vx * self.scaling_for_ground_truth_vx
        if (self.should_include_info_for_ground_truth_vx):
            info_dict["ground_truth_vx"][0] = self.car.vx * self.scaling_for_ground_truth_vx

        if (self.should_include_obs_for_ground_truth_vy):
            obs_dict["ground_truth_vy"][0]  = self.car.vy * self.scaling_for_ground_truth_vy
        if (self.should_include_info_for_ground_truth_vy):
            info_dict["ground_truth_vy"][0] = self.car.vy * self.scaling_for_ground_truth_vy

        if (self.should_include_obs_for_ground_truth_omega):
            obs_dict["ground_truth_omega"][0]  = self.car.omega * self.scaling_for_ground_truth_omega
        if (self.should_include_info_for_ground_truth_omega):
            info_dict["ground_truth_omega"][0] = self.car.omega * self.scaling_for_ground_truth_omega

        if (self.should_include_obs_for_ground_truth_delta):
            obs_dict["ground_truth_delta"][0]  = self.car.delta * self.scaling_for_ground_truth_delta
        if (self.should_include_info_for_ground_truth_delta):
            info_dict["ground_truth_delta"][0] = self.car.delta * self.scaling_for_ground_truth_delta

        if (self.should_include_obs_for_road_progress_at_closest_point):
            obs_dict["road_progress_at_closest_point"][0]  = road_progress_at_closest_point
        if (self.should_include_info_for_road_progress_at_closest_point):
            info_dict["road_progress_at_closest_point"][0] = road_progress_at_closest_point

        if (self.should_include_obs_for_vx_sensor):
            obs_dict["vx_sensor"][0]  = vx_sensor
        if (self.should_include_info_for_vx_sensor):
            info_dict["vx_sensor"][0] = vx_sensor

        if (self.should_include_obs_for_distance_to_closest_point):
            obs_dict["distance_to_closest_point"][0]  = dist_to_line
        if (self.should_include_info_for_distance_to_closest_point):
            info_dict["distance_to_closest_point"][0] = dist_to_line

        if (self.should_include_obs_for_heading_angle_relative_to_line):
            obs_dict["heading_angle_relative_to_line"][0]  = heading_rel_to_line
        if (self.should_include_info_for_heading_angle_relative_to_line):
            info_dict["heading_angle_relative_to_line"][0] = heading_rel_to_line

        if (self.should_include_obs_for_heading_angular_rate_gyro):
            obs_dict["heading_angular_rate_gyro"][0]  = heading_gyro
        if (self.should_include_info_for_heading_angular_rate_gyro):
            info_dict["heading_angular_rate_gyro"][0] = heading_gyro

        if (self.should_include_obs_for_accel_in_body_frame_x):
            obs_dict["accel_in_body_frame_x"][0]  = 0.0 if accel_x_meas is None else accel_x_meas
        if (self.should_include_info_for_accel_in_body_frame_x):
            info_dict["accel_in_body_frame_x"][0] = 0.0 if accel_x_meas is None else accel_x_meas

        if (self.should_include_obs_for_accel_in_body_frame_y):
            obs_dict["accel_in_body_frame_y"][0]  = 0.0 if accel_y_meas is None else accel_y_meas
        if (self.should_include_info_for_accel_in_body_frame_y):
            info_dict["accel_in_body_frame_y"][0] = 0.0 if accel_y_meas is None else accel_y_meas

        if (self.should_include_obs_for_closest_point_coords_in_body_frame):
            obs_dict["closest_point_coords_in_body_frame"][0]  = px_closest_in_body_frame
            obs_dict["closest_point_coords_in_body_frame"][1]  = py_closest_in_body_frame
        if (self.should_include_info_for_closest_point_coords_in_body_frame):
            info_dict["closest_point_coords_in_body_frame"][0] = px_closest_in_body_frame
            info_dict["closest_point_coords_in_body_frame"][1] = py_closest_in_body_frame

        if (self.should_include_obs_for_look_ahead_line_coords_in_body_frame):
            obs_dict["look_ahead_line_coords_in_body_frame"]  = look_ahead_line_coords_in_body_frame
        if (self.should_include_info_for_look_ahead_line_coords_in_body_frame):
            info_dict["look_ahead_line_coords_in_body_frame"] = look_ahead_line_coords_in_body_frame

        if (self.should_include_obs_for_road_curvature_at_closest_point):
            obs_dict["road_curvature_at_closest_point"][0]  = road_curvature_at_closest_point
        if (self.should_include_info_for_road_curvature_at_closest_point):
            info_dict["road_curvature_at_closest_point"][0] = road_curvature_at_closest_point

        if (self.should_include_obs_for_look_ahead_road_curvatures):
            obs_dict["look_ahead_road_curvatures"]  = look_ahead_road_curvatures
        if (self.should_include_info_for_look_ahead_road_curvatures):
            info_dict["look_ahead_road_curvatures"] = look_ahead_road_curvatures

        if (self.should_include_obs_for_gps_line_coords_in_world_frame):
            obs_dict["gps_line_coords_in_world_frame"][0]  = 0.0
        if (self.should_include_info_for_gps_line_coords_in_world_frame):
            info_dict["gps_line_coords_in_world_frame"][0] = 0.0

        if (self.should_include_obs_for_cone_detections):
            obs_dict["cone_detections_x"]  = cone_detections_x
            obs_dict["cone_detections_y"]  = cone_detections_y
            obs_dict["cone_detections_side_of_road"]  = cone_detections_side_of_road
            obs_dict["cone_detections_num_cones"][0]  = num_cones_in_obs
        if (self.should_include_info_for_cone_detections):
            info_dict["cone_detections_x"]  = cone_detections_x
            info_dict["cone_detections_y"]  = cone_detections_y
            info_dict["cone_detections_side_of_road"]  = cone_detections_side_of_road
            info_dict["cone_detections_num_cones"][0]  = num_cones_in_obs


        # Return the two dictionaries
        return obs_dict, info_dict

    def _get_road_info(self, progress_queries):
        """
        Gets the details of the upcoming section of road,
        relative to the car, into a dictionary.

        Parameters
        ----------
            progress_queries : numpy arrray, 1-dimensional
                Specifies the values of progress-along-the-road, relative to
                the current position of the car, at which the road information
                should be generated.
                Units: meters

        Returns
        -------
            road_info_dict : dictionary
                Containing details for the road relative to the current
                state of the car.
                The properties of the info_dict are:
                - "px", "py" : float
                    World-frame (x,y) coordinate of the car.
                - "px_closest", "py_closest" : float
                    World-frame (x,y) coordinate of the closest point on the road.
                - "closest_distance" : float
                    Euclidean distance from the car to the closest point on the road.
                - "side_of_the_road_line" : int
                    The side of the road that the car is on (1:=left, -1=right).
                - "progress_at_closest_p" : float
                    The total length of road from the start of the road to the closest point.
                - "road_angle_at_closest_p" : float
                    The angle of the road at the closest point (relative to the world-frame x-axis).
                - "road_angles_relative_to_body_frame_at_closest_p" : float
                    Angle of the road, relative to the body frame, at the closest point.
                - "curvature_at_closest_p" : float
                    The curvature of the road at the closest point.
                - "closest_element_idx" : int
                    The index of the road element that is closest to the car.
                - "progress_queries" : numpy array, 1-dimensional
                    A repeat of the input parameter that specifies the values of progress-along-the-road,
                    relative to the current position of the car, at which the observations should be generated. 
                - "road_points_in_body_frame" : numpy array, 2-dimensional
                    (px,py) coordinates in the body frame of the progress query points.
                    A 2-dimensional numpy array with: size = number of query points -by- 2.
                - "road_angles_relative_to_body_frame" : numpy array, 1-dimensional
                    Angle of the road, relative to the body frame, at each of the progress query points.
                    A 1-dimensional numpy array with: size = number of query points.
                - "curvatures" : numpy array, 1-dimensional
                    Curvature of the road at each of the progress query points.
                    A 1-dimensional numpy array with: size = number of query points.

                Units: all length in meters, all angles in radians.
        """
        return self.road.road_info_at_given_pose_and_progress_queries(px=self.car.px, py=self.car.py, theta=self.car.theta, progress_queries=progress_queries)



    @staticmethod
    def compute_default_reward_for_distance_to_line(d: float):
        """
        Calculate the default reward for encouraging the car to stay on or
        close to the line-to-follow, i.e., encourage zero distance to the
        center of the road.

        Parameters
        ----------
            d : float
                Distance to the center of the road in meters. This distance
                should be a positive number regardless of the side of the road.

        Returns
        -------
            float : reward value
                Maximum value of reward: 3, goes to negative for distances > 2
        """
        # Define the coefficients
        a = 3.0
        b = 1.0
        c = 2.0

        # Calculate the reward separately for each interval of the distance
        # > If (distance < 0.5); then reward with higher curvature
        if d < 0.5:
            return a * (1 - d**2)
        # > Else If (0.5 <= distance < 2); then reward with lower curvature
        elif 0.5 <= d < 2:
            return b * (2 - d)**2
        # > Else (2 <= distance); reward has high curvature
        else:       # If
            return -c * (d - 2)**3



    @staticmethod
    def compute_default_reward_for_speed(speed_in_kmph: float):
        """
        Default reward function for encouraging the car to maintain
        a speed of 60 km/h

        Parameters
        ----------
            speed_in_kmph : float
            The speed of the car in kilometers per hour (i.e., kmph).

        Returns
        -------
            float: reward value
                Maximum value of reward: 300
        """
        # Coefficients derived from solving the equations
        a = 1.0/12.0  # Coefficient for the quadratic function in the first and second segments
        b = 300.0     # Offset for the quadratic function in the first segment
        e = 10.0      # Coefficient for the linear function in the third segment

        # Calculate the reward separately for each interval of the speed
        if 0 <= speed_in_kmph < 120:
            return -a * (speed_in_kmph - 60)**2 + b  # Quadratic function for 0 to 120 kmph with max at 60kph
        else:
            return -e * (speed_in_kmph - 120)  # Linear decreasing function for speed above 120 kmph

    def undo_scaling(self, dict):
        # Take a copy so are not to change the input dictionary
        unscaled_dict = dict.copy()
        # Unscale the value for each key in the dictionary
        for key, value in dict.items():
            unscaled_dict.update({key : (value / getattr(self, "scaling_for_"+key))})
        # Return the unscaled dictionary
        return unscaled_dict



    def reset(self, seed=None, options=None):
        """
        Resets the position of the car and generates an observation

        Parameters
        ----------
            seed : int
                For seeding the numpy random number generator
            options : dictionary
                Can be used to specify any one or more of the minimum
                and maximum bounds from which each element of the
                initial state is randomly drawn.
                Hence, this dictionary can contain any one or more of
                the following keys:
                - px_init_min,    px_init_max    : float
                - py_init_min,    py_init_max    : float
                - theta_init_min, theta_init_max : float
                - vx_init_min,    vx_init_max    : float
                - vy_init_min,    vy_init_max    : float
                - omega_init_min, omega_init_max : float
                - delta_init_min, delta_init_max : float

        Returns
        -------
            observation : dictionary
                Containing keys for each state of the car,
                i.e., "px", "py", "theta", "vx", "vy", "omega", "delta"
            info_dict : dictionary
                Same definition as for the "_get_road_info" function
        """

        # Reset the seed of self.np_random
        super().reset(seed=seed)

        # Extract any options for the initial conditions
        if (options is None):
            options = {}

        px_init_min    = options["px_init_min"]    if ("px_init_min"    in options) else self.px_init_min
        px_init_max    = options["px_init_max"]    if ("px_init_max"    in options) else self.px_init_max
        py_init_min    = options["py_init_min"]    if ("py_init_min"    in options) else self.py_init_min
        py_init_max    = options["py_init_max"]    if ("py_init_max"    in options) else self.py_init_max
        theta_init_min = options["theta_init_min"] if ("theta_init_min" in options) else self.theta_init_min
        theta_init_max = options["theta_init_max"] if ("theta_init_max" in options) else self.theta_init_max
        vx_init_min    = options["vx_init_min"]    if ("vx_init_min"    in options) else self.vx_init_min
        vx_init_max    = options["vx_init_max"]    if ("vx_init_max"    in options) else self.vx_init_max
        vy_init_min    = options["vy_init_min"]    if ("vy_init_min"    in options) else self.vy_init_min
        vy_init_max    = options["vy_init_max"]    if ("vy_init_max"    in options) else self.vy_init_max
        omega_init_min = options["omega_init_min"] if ("omega_init_min" in options) else self.omega_init_min
        omega_init_max = options["omega_init_max"] if ("omega_init_max" in options) else self.omega_init_max
        delta_init_min = options["delta_init_min"] if ("delta_init_min" in options) else self.delta_init_min
        delta_init_max = options["delta_init_max"] if ("delta_init_max" in options) else self.delta_init_max

        # Sample the initial condition
        px_init    = self.np_random.uniform(low=px_init_min,    high=px_init_max)
        py_init    = self.np_random.uniform(low=py_init_min,    high=py_init_max)
        theta_init = self.np_random.uniform(low=theta_init_min, high=theta_init_max)
        vx_init    = self.np_random.uniform(low=vx_init_min,    high=vx_init_max)
        vy_init    = self.np_random.uniform(low=vy_init_min,    high=vy_init_max)
        omega_init = self.np_random.uniform(low=omega_init_min, high=omega_init_max)
        delta_init = self.np_random.uniform(low=delta_init_min, high=delta_init_max)
        
        # Reset the state of the car
        self.car.reset(
            px    = px_init,
            py    = py_init,
            theta = theta_init,
            vx    = vx_init,
            vy    = vy_init,
            omega = omega_init,
            delta = delta_init
        )

        # Get the observation
        observation, info_dict = self._get_observation_and_info_and_update_ground_truth()

        # Set the previous progress variable
        self.previous_progress_at_closest_p = self.current_ground_truth["road_progress_at_closest_point"]

        # Render, if necessary
        # ...

        # Return the observation and info dictionary
        return observation, info_dict



    def step(self, action):
        """
        Steps forward the car by one time increment,
        and generates an observation.

        Parameters
        ----------
            action : array
                Should match the format defined by self.action_space,
                i.e., with the following keys:
                - action[0] : float
                  Drive command to the motor in percent.
                  100.0 (percent) means maximum force is applied to the
                  bicycle model in the forwards direction (i.e., in the
                  body frame positive x-direction)
                  -100.0 (percent) means maximum force is applied in the 
                  backwards direction (i.e., body frame negative x-direction)
                - action[1] : float
                  Requested steering angle (units: radians)

        Returns
        -------
            observation : dictionary
                Containing keys for each state of the car,
                i.e., "px", "py", "theta", "vx", "vy", "omega", "delta"
            info_dict : dictionary
                Same definition as for the "_get_info" function.
        """

        # Set the action request for bicycle model
        self.car.set_action_requests(
            drive_command_request  = action[0],
            delta_request = action[1],
        )

        # Get the road condition
        road_condition = self.road_condition
        Dp = None
        Cp = None
        Bp = None
        Ep = None
        if (road_condition == "other"):
            Dp = self.D_Pacejka
            Cp = self.C_Pacejka
            Bp = self.B_Pacejka
            Ep = self.E_Pacejka

        # Perform one integration step
        self.car.perform_integration_step(
            Ts = self.integration_Ts,
            method = self.integration_method,
            num_steps = self.integration_num_steps_per_Ts,
            should_update_state = True,
            road_condition = road_condition,
            Dp=Dp, Cp=Cp, Bp=Bp, Ep=Ep
        )

        # Get the observation
        observation, info_dict = self._get_observation_and_info_and_update_ground_truth()

        # Extract the value of "progress at closest point"
        progress_at_closest_point = self.current_ground_truth["road_progress_at_closest_point"]

        # Extract the value of "distance to closest point"
        distance_to_closest_point = self.current_ground_truth["distance_to_closest_point"]

        # Compute the speed of the car
        car_speed = np.sqrt(self.car.vx**2 + self.car.vy**2)

        # Compute the "terminated" flag and reasons
        terminated = False
        termination_reward = 0.0
        terminated_due_to_finish = False
        terminated_due_to_speed_high = False
        terminated_due_to_speed_low = False
        terminated_due_to_offtrack = False
        if (progress_at_closest_point >= self.total_road_length_for_termination):
            terminated = True
            terminated_due_to_finish = True
        # > Terminate for being outside of the speed range
        if (car_speed > self.termination_speed_upper_bound):
            terminated = True
            terminated_due_to_speed_high = True
            # termination_reward += self.termination_reward_for_speed_upper_bound
        if (car_speed < self.termination_speed_lower_bound):
            terminated = True
            terminated_due_to_speed_low = True
            # termination_reward += self.termination_reward_for_speed_lower_bound
        # > Terminate for deviating too much from the line
        if (abs(distance_to_closest_point) > self.termination_distance_to_closest_point_upper_bound):
            terminated = True
            terminated_due_to_offtrack = True
            # termination_reward += self.termination_reward_for_distance_to_closest_point_upper_bound

        # Expose termination reasons and configured magnitudes for wrapper use
        info_dict["termination"] = {
            "finished": terminated_due_to_finish,
            "speed_high": terminated_due_to_speed_high,
            "speed_low": terminated_due_to_speed_low,
            "off_track": terminated_due_to_offtrack,
        }
        info_dict["termination_rewards"] = {
            "speed_upper_bound": self.termination_reward_for_speed_upper_bound,
            "speed_lower_bound": self.termination_reward_for_speed_lower_bound,
            "off_track": self.termination_reward_for_distance_to_closest_point_upper_bound,
        }

        # Compute the "truncated" flag
        truncated = False

        # Set the reward to zero; reward shaping to be done in a wrapper
        reward = 0.0
        self.previous_progress_at_closest_p = progress_at_closest_point

        # Render, if necessary
        #if self.render_mode in ["matplotlib"]:
        #    self._render_frame()

        # Return the observation and info dictionary
        return observation, reward, terminated, truncated, info_dict        


    def render_matplotlib_init_figure(self):
        # Close if there is an existing figure
        if (self.figure is not None):
            plt.close(self.figure)

        # Open a figure with an axis
        self.figure, self.axis = plt.subplots(1, 1)

        # Plot the car
        px  = self.car.px
        py  = self.car.py
        theta = self.car.theta
        delta = self.car.delta
        car_scale = 1.0
        car_handles = self.car.render_car(self.axis, px, py, theta, delta, scale=car_scale)
        self.car_handles.append(car_handles)

        # Compute a bounding radius for the car
        Lf = self.car.Lf * car_scale
        Lr = self.car.Lr * car_scale
        car_plot_length = 1.5 * np.sqrt(Lf**2 + Lr**2)
        
        # Set the axis limits
        self.axis.set_xlim(xmin=-0.7*car_plot_length, xmax=0.7*car_plot_length)
        self.axis.set_ylim(ymin=-0.7*car_plot_length, ymax=0.7*car_plot_length)

        # Add the grid lines
        self.axis.grid(visible=True, which="both", axis="both", linestyle='--')

        # Set the aspect ratio
        self.axis.set_aspect('equal', adjustable='box')

        # Label the axes
        self.axis.set_xlabel('x (World) [meters]')
        self.axis.set_ylabel('y (World) [meters]')
        
        # Save the figure
        #self.figure.savefig('service_environment.pdf')
        
        # Return the car
        # > This return is for compatibility with
        #   matplotlib.pyploy.FuncAnimation
        return self.car_handles

    def render_matplotlib_plot_road(self):
        # Initialize the figure, if it does not already exist
        if (self.figure is None):
            self.render_matplotlib_init_figure()

        # Auto-scale the axis limits
        self.axis.set_xlim(auto=True)
        self.axis.set_ylim(auto=True)

        # Call the function to render the road
        self.road_handles = self.road.render_road(self.axis)

        # Ensure the aspect ratio stays as 1:1
        self.axis.set_aspect('equal', adjustable='box')

        # Return the road handles
        return self.road_handles

    def render_matplotlib_zoom_to(self, px, py, x_width, y_height, axis_handle=None):
        if (axis_handle is not None):
            # Set the axis limits
            axis_handle.set_xlim(xmin=px-0.5*x_width,  xmax=px+0.5*x_width)
            axis_handle.set_ylim(ymin=py-0.5*y_height, ymax=py+0.5*y_height)
        elif (self.figure is not None):
            # Set the axis limits
            self.axis.set_xlim(xmin=px-0.5*x_width,  xmax=px+0.5*x_width)
            self.axis.set_ylim(ymin=py-0.5*y_height, ymax=py+0.5*y_height)

    def render_matplotlib_animation_of_trajectory(self, px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=1, figure_title=None, zoom_width=20, zoom_height=20):
        # Create a figure for the animation
        fig_4_ani, axis_4_ani = plt.subplots(1, 1)
        interval_btw_frames_ms = Ts * 1000

        # Plot the road
        self.road.render_road(axis_4_ani)

        # If the road has cones, then plot the cones
        # > Get the cones coordinates
        cones_left_side_coords  = self.road.get_cones_left_side()
        cones_right_side_coords = self.road.get_cones_right_side()
        # > Plot the left-side cones in yellow
        if (len(cones_left_side_coords)>0):
            cone_handles_left_side  = axis_4_ani.scatter(x=cones_left_side_coords[:,0],  y=cones_left_side_coords[:,1],  s=8.0, marker="o", facecolor="y", alpha=1.0, linewidths=0, edgecolors="k")
        # > Plot the right-side cones in blue
        if (len(cones_right_side_coords)>0):
            cone_handles_right_side = axis_4_ani.scatter(x=cones_right_side_coords[:,0], y=cones_right_side_coords[:,1], s=8.0, marker="o", facecolor="b", alpha=1.0, linewidths=0, edgecolors="k")

        # Add a title
        if (figure_title is None):
            fig_4_ani.suptitle('Animation of car', fontsize=12)
        else:
            fig_4_ani.suptitle(figure_title, fontsize=12)

        # Plot the start position
        car_handles = self.car.render_car(axis_4_ani,px_traj[0],py_traj[0],theta_traj[0],delta_traj[0],scale=1.0)

        # Zoom into the start position
        self.render_matplotlib_zoom_to(px=px_traj[0],py=py_traj[0],x_width=zoom_width,y_height=zoom_height,axis_handle=axis_4_ani)

        # Display that that animation creation is about to start
        print("Now creating the animation, this may take some time.")

        # Function for printing each individual frame of the animation
        def animate_one_trajectory_frame(i):
            # Update the car
            self.car.render_car(axis_4_ani,px_traj[i],py_traj[i],theta_traj[i],delta_traj[i],scale=1.0, plot_handles=car_handles)
            # Update the window
            self.render_matplotlib_zoom_to(px=px_traj[i],py=py_traj[i],x_width=zoom_width,y_height=zoom_height,axis_handle=axis_4_ani)

        # Check the length of the trajectory
        # > By checking for the first nan
        # > If the trajectory did not terminate nor truncate,
        #   then there should be no nans
        # Find the index of the first NaN
        nan_index = np.where(np.isnan(px_traj))[0]
        # Determine the length up to the first NaN
        if nan_index.size > 0:
            traj_length = nan_index[0]
        else:
            traj_length = len(px_traj)

        # Create the list of trajectory indicies to render
        traj_indicies_to_render = range(0,traj_length,traj_increment)

        # Create the animation
        ani = animation.FuncAnimation(fig_4_ani, animate_one_trajectory_frame, frames=traj_indicies_to_render, interval=interval_btw_frames_ms)

        # Return the animation object
        return ani


    #def render(self):
    #    if self.render_mode == "matplotlib":
    #        return self._render_frame()

    #def _render_frame(self):
    #    return



    def close(self):
    # Close the matplotlib figure
        if (self.figure is not None):
            plt.close(self.figure)



    # -------------------------
    # GETTER AND SETTER METHODS
    # -------------------------

    # GETTER FUNCTIONS
    def get_integration_method(self):
        return self.integration_method

    def get_integration_Ts(self):
        return self.integration_Ts

    def get_car_state(self):
        return {
            "px"    : self.car.px,
            "py"    : self.car.py,
            "theta" : self.car.theta,
            "vx"    : self.car.vx,
            "vy"    : self.car.vy,
            "omega" : self.car.omega,
            "delta" : self.car.delta,
        }

    def get_current_ground_truth(self):
        return self.current_ground_truth

    def set_integration_method(self, integration_method):
        # Set the method to be used for numerical integration of the equations of motion
        if (integration_method in self.metadata["integration_methods"]):
            self.integration_method = integration_method
        else:
            print("WARNING: The requested integration method \"" + str(integration_method) + "\" is not allowed.")
            print("The valid integration methods are: " + str(self.metadata["integration_methods"]))
            print("Exiting now.")
            exit()

    def set_integration_Ts(self, Ts):
        if (Ts > 0):
            self.integration_Ts = Ts
        else:
            print("WARNING: The requested integration time Ts \"" + str(Ts) + "\" is not allowed.")
            print("A valid integration time Ts must be greater than zero.")
            print("Exiting now.")
            exit()

    def set_road_condition(self, road_condition, Dp=None, Cp=None, Bp=None, Ep=None):
        """
        Parameters
        ----------
            "road_condition" : string
            Specifies which parameter values to use for determining the
            tire forces that arise due to slippage at high speeds.
            Options: { "dry" , "wet" , "snow" , "ice" , "other" }
            If "other" is specified, then must specify all four of the
            following keys
            
            "D_Pacejka" : float
            "C_Pacejka" : float
            "B_Pacejka" : float
            "E_Pacejka" : float
            The coefficients of Pacejka's Tire Formula, which are used when
            the "road_condition" key is set to "other"

        Returns
        -------
        Nothing
        """
        # Set the road condition
        self.road_condition = road_condition

        # If condition is "other", set also the
        # Pacejka tire model parameters
        self.D_Pacejka = Dp
        self.C_Pacejka = Cp
        self.B_Pacejka = Bp
        self.E_Pacejka = Ep

    # def set_progress_queries_for_generating_observations(self, progress_queries):
    #     """
    #     Parameters
    #     ----------
    #         "progress_queries" : numpy array, 1-dimensional
    #         Specifies the values of progress-along-the-road,
    #         relative to the current position of the car, at
    #         which the observations should be generated. These
    #         observations are returned in the "info_dict".
    #         (Units: meters)

    #     Returns
    #     -------
    #     Nothing
    #     """
    #     # Get the progress query values
    #     self.progress_queries = progress_queries
