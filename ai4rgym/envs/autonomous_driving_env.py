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
                Specifies the inegration details used in the "step" function.
                This dictionary should contain the folllowing keys:
                - method : string
                    The numerical integration method that is used when simulating the
                    evolution of the car.
                - Ts : float
                    The numerical integration time step that the environment evolves
                    by each time the "step" function is called.
                - num_steps_per_Ts : int
                    The number of step that "integration_Ts" is split up into.

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
                - should_include_obs_for_gps_line_coords_in_world_frame        :  bool

                (Scaling value for each sensor measurement)
                - scaling_for_ground_truth_state                    :  float
                - scaling_for_vx_sensor                             :  float
                - scaling_for_closest_distance_to_line              :  float
                - scaling_for_heading_angle_relative_to_line        :  float
                - scaling_for_heading_angle_gyro                    :  float
                - scaling_for_accel_in_body_frame_x                 :  float
                - scaling_for_accel_in_body_frame_y                 :  float
                - scaling_for_look_ahead_line_coords_in_body_frame  :  float
                - scaling_for_gps_line_coords_in_world_frame        :  float

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
        self.road = Road(epsilon_c=(1/500))

        # Add the road element
        for element in road_elements_list:
            if (element["type"] == "straight"):
                self.road.add_road_element_straight(element["length"])
            elif (element["type"] == "curved"):
                if ("angle_in_degrees" in element):
                    self.road.add_road_element_curved_by_angle(curvature=element["curvature"], angle_in_degrees=element["angle_in_degrees"])
                elif ("length" in element):
                    self.road.add_road_element_curved_by_length(curvature=element["curvature"], length=element["length"])
                else:
                    print("ERROR: curved road element specification is invalid, element = " + str(element))
            else:
                print("ERROR: road element type is invalid, element = " + str(element))

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
        # Set the observation parameters
        # > Flags for which sensor measurements to include in the observations
        self.should_include_obs_for_ground_truth_state                    =  False  if ("should_include_obs_for_ground_truth_state"                   not in observation_parameters) else observation_parameters["should_include_obs_for_ground_truth_state"]
        self.should_include_obs_for_vx_sensor                             =  True   if ("should_include_obs_for_vx_sensor"                            not in observation_parameters) else observation_parameters["should_include_obs_for_vx_sensor"]
        self.should_include_obs_for_closest_distance_to_line              =  True   if ("should_include_obs_for_closest_distance_to_line"             not in observation_parameters) else observation_parameters["should_include_obs_for_closest_distance_to_line"]
        self.should_include_obs_for_heading_angle_relative_to_line        =  True   if ("should_include_obs_for_heading_angle_relative_to_line"       not in observation_parameters) else observation_parameters["should_include_obs_for_heading_angle_relative_to_line"]
        self.should_include_obs_for_heading_angle_gyro                    =  True   if ("should_include_obs_for_heading_angle_gyro"                   not in observation_parameters) else observation_parameters["should_include_obs_for_heading_angle_gyro"]
        self.should_include_obs_for_accel_in_body_frame_x                 =  False  if ("should_include_obs_for_accel_in_body_frame_x"                not in observation_parameters) else observation_parameters["should_include_obs_for_accel_in_body_frame_x"]
        self.should_include_obs_for_accel_in_body_frame_y                 =  False  if ("should_include_obs_for_accel_in_body_frame_y"                not in observation_parameters) else observation_parameters["should_include_obs_for_accel_in_body_frame_y"]
        self.should_include_obs_for_look_ahead_line_coords_in_body_frame  =  True   if ("should_include_obs_for_look_ahead_line_coords_in_body_frame" not in observation_parameters) else observation_parameters["should_include_obs_for_look_ahead_line_coords_in_body_frame"]
        self.should_include_obs_for_gps_line_coords_in_world_frame        =  False  if ("should_include_obs_for_gps_line_coords_in_world_frame"       not in observation_parameters) else observation_parameters["should_include_obs_for_gps_line_coords_in_world_frame"]

        # > Scaling value for each sensor measurement
        self.scaling_for_ground_truth_state                    =  1.0  if ("scaling_include_obs_for_ground_truth_state"                   not in observation_parameters) else observation_parameters["scaling_for_ground_truth_state"]
        self.scaling_for_vx_sensor                             =  1.0  if ("scaling_include_obs_for_vx_sensor"                            not in observation_parameters) else observation_parameters["scaling_for_vx_sensor"]
        self.scaling_for_closest_distance_to_line              =  1.0  if ("scaling_include_obs_for_closest_distance_to_line"             not in observation_parameters) else observation_parameters["scaling_for_closest_distance_to_line"]
        self.scaling_for_heading_angle_relative_to_line        =  1.0  if ("scaling_include_obs_for_heading_angle_relative_to_line"       not in observation_parameters) else observation_parameters["scaling_for_heading_angle_relative_to_line"]
        self.scaling_for_heading_angle_gyro                    =  1.0  if ("scaling_include_obs_for_heading_angle_gyro"                   not in observation_parameters) else observation_parameters["scaling_for_heading_angle_gyro"]
        self.scaling_for_accel_in_body_frame_x                 =  1.0  if ("scaling_include_obs_for_accel_in_body_frame_x"                not in observation_parameters) else observation_parameters["scaling_for_accel_in_body_frame_x"]
        self.scaling_for_accel_in_body_frame_y                 =  1.0  if ("scaling_include_obs_for_accel_in_body_frame_y"                not in observation_parameters) else observation_parameters["scaling_for_accel_in_body_frame_y"]
        self.scaling_for_look_ahead_line_coords_in_body_frame  =  1.0  if ("scaling_include_obs_for_look_ahead_line_coords_in_body_frame" not in observation_parameters) else observation_parameters["scaling_for_look_ahead_line_coords_in_body_frame"]
        self.scaling_for_gps_line_coords_in_world_frame        =  1.0  if ("scaling_include_obs_for_gps_line_coords_in_world_frame"       not in observation_parameters) else observation_parameters["scaling_for_gps_line_coords_in_world_frame"]

        # > Specifications for each sensor measurement
        self.vx_sensor_bias    =  0.0  if ("vx_sensor_bias"    not in observation_parameters) else observation_parameters["vx_sensor_bias"]
        self.vx_sensor_stddev  =  0.1  if ("vx_sensor_stddev"  not in observation_parameters) else observation_parameters["vx_sensor_stddev"]

        self.closest_distance_to_line_bias    =  0.0  if ("closest_distance_to_line_bias"    not in observation_parameters) else observation_parameters["closest_distance_to_line_bias"]
        self.closest_distance_to_line_stddev  =  0.1  if ("closest_distance_to_line_stddev"  not in observation_parameters) else observation_parameters["closest_distance_to_line_stddev"]

        self.heading_angle_relative_to_line_bias    =  0.0  if ("heading_angle_relative_to_line_bias"    not in observation_parameters) else observation_parameters["heading_angle_relative_to_line_bias"]
        self.heading_angle_relative_to_line_stddev  =  0.01  if ("heading_angle_relative_to_line_stddev"  not in observation_parameters) else observation_parameters["heading_angle_relative_to_line_stddev"]

        self.heading_angle_gyro_bias    =  0.0  if ("heading_angle_gyro_bias"    not in observation_parameters) else observation_parameters["heading_angle_gyro_bias"]
        self.heading_angle_gyro_stddev  =  0.01  if ("heading_angle_gyro_stddev"  not in observation_parameters) else observation_parameters["heading_angle_gyro_stddev"]

        self.look_ahead_line_coords_in_body_frame_distance             =  50.0  if ("look_ahead_line_coords_in_body_frame_distance"             not in observation_parameters) else observation_parameters["look_ahead_line_coords_in_body_frame_distance"]
        self.look_ahead_line_coords_in_body_frame_num_points           =  10  if ("look_ahead_line_coords_in_body_frame_num_points"           not in observation_parameters) else observation_parameters["look_ahead_line_coords_in_body_frame_num_points"]
        self.look_ahead_line_coords_in_body_frame_stddev_lateral       =   0.0  if ("look_ahead_line_coords_in_body_frame_stddev_lateral"       not in observation_parameters) else observation_parameters["look_ahead_line_coords_in_body_frame_stddev_lateral"]
        self.look_ahead_line_coords_in_body_frame_stddev_longitudinal  =   0.0  if ("look_ahead_line_coords_in_body_frame_stddev_longitudinal"  not in observation_parameters) else observation_parameters["look_ahead_line_coords_in_body_frame_stddev_longitudinal"]

        # OBSERVATION SPACE
        # Construct a normal dictionary for the sensor measurements
        # that are to be included as observations.
        # Note that any measurements not included in the observations
        # is included in the "info_dict".
        # Hence, the combination of the observation and "info_dict"
        # provides all possible sensor measurements without duplication.

        # > Initialis an empty dictionary
        obs_space_dict = {}
        self.obs_dict_blank = {}
        self.info_dict_blank = {}

        if (self.should_include_obs_for_ground_truth_state):
            obs_space_dict.update({
                "gt_px":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_py":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_theta": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_vx":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_vy":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_omega": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "gt_delta": spaces.Box(low=-0.5*np.pi, high=0.5*np.pi, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({
                "gt_px":    0.0,
                "gt_py":    0.0,
                "gt_theta": 0.0,
                "gt_vx":    0.0,
                "gt_vy":    0.0,
                "gt_omega": 0.0,
                "gt_delta": 0.0,
            })
        else:
            self.info_dict_blank.update({
                "gt_px":    0.0,
                "gt_py":    0.0,
                "gt_theta": 0.0,
                "gt_vx":    0.0,
                "gt_vy":    0.0,
                "gt_omega": 0.0,
                "gt_delta": 0.0,
            })

        if (self.should_include_obs_for_vx_sensor):
            print("WARNING: the observation is not implemented (vx)")
            obs_space_dict.update({
                "vx":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"vx": 0.0})
        else:
            self.info_dict_blank.update({"vx": 0.0})

        if (self.should_include_obs_for_closest_distance_to_line):
            print("WARNING: the observation is not implemented (dist to line)")
            obs_space_dict.update({
                "dist_to_line":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"dist_to_line": 0.0})
        else:
            self.info_dict_blank.update({"dist_to_line": 0.0})

        if (self.should_include_obs_for_heading_angle_relative_to_line):
            print("WARNING: the observation is not implemented (heading relative to line)")
            obs_space_dict.update({
                "heading_rel_to_line":    spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"heading_rel_to_line": 0.0})
        else:
            self.info_dict_blank.update({"heading_rel_to_line": 0.0})

        if (self.should_include_obs_for_heading_angle_gyro):
            print("WARNING: the observation is not implemented (heading gyro)")
            obs_space_dict.update({
                "heading_gyro":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"heading_gyro": 0.0})
        else:
            self.info_dict_blank.update({"heading_gyro":0.0})

        if (self.should_include_obs_for_accel_in_body_frame_x):
            print("WARNING: the observation is not implemented (accelerometer in body frame x)")
            obs_space_dict.update({
                "accel_x":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"accel_x": 0.0})
        else:
            self.info_dict_blank.update({"accel_x": 0.0})

        if (self.should_include_obs_for_accel_in_body_frame_y):
            print("WARNING: the observation is not implemented (accelerometer in body frame y)")
            obs_space_dict.update({
                "accel_y":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"accel_y": 0.0})
        else:
            self.info_dict_blank.update({"accel_y": 0.0})

        if (self.should_include_obs_for_look_ahead_line_coords_in_body_frame):
            print("WARNING: the observation is not implemented (look ahead line coords)")
            obs_space_dict.update({
                "look_ahead_line_coords":    spaces.Box(low=-np.inf, high=np.inf, shape=(2,self.look_ahead_line_coords_in_body_frame_num_points), dtype=np.float32),
            })
            self.obs_dict_blank.update({"look_ahead_line_coords": 0.0})
        else:
            self.info_dict_blank.update({"look_ahead_line_coords": 0.0})

        if (self.should_include_obs_for_gps_line_coords_in_world_frame):
            print("WARNING: the observation is not implemented (gps line coords)")
            obs_space_dict.update({
                "gps_line_coords":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })
            self.obs_dict_blank.update({"gps_line_coords": 0.0})
        else:
            self.info_dict_blank.update({"gps_line_coords": 0.0})



        # > Create the observation space
        self.observation_space = spaces.Dict(obs_space_dict)



        # Initialize the "previous progress" variable to zero
        # > It is set to the correct value in the reset function
        self.previous_progress_at_closest_p = 0.0

        # PROGRESS QUERIES:
        # Set the progress queries used for filling
        # the "info_dict" with look-ahead observations.
        self.progress_queries = np.array([0.0,1.0,2.0,3.0,4.0,5.0], dtype=np.float32)

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



    def _get_observation_and_info(self):
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
        # Initialise an empty
        obs_dict  = self.obs_dict_blank
        info_dict = self.info_dict_blank

        if (self.should_include_obs_for_ground_truth_state):
            obs_dict.update({
                "gt_px":    np.array([self.car.px], dtype=np.float32),
                "gt_py":    np.array([self.car.py], dtype=np.float32),
                "gt_theta": np.array([self.car.theta], dtype=np.float32),
                "gt_vx":    np.array([self.car.vx], dtype=np.float32),
                "gt_vy":    np.array([self.car.vy], dtype=np.float32),
                "gt_omega": np.array([self.car.omega], dtype=np.float32),
                "gt_delta": np.array([self.car.delta], dtype=np.float32),
            })
        else:
            info_dict.update({
                "gt_px":    np.array([self.car.px], dtype=np.float32),
                "gt_py":    np.array([self.car.py], dtype=np.float32),
                "gt_theta": np.array([self.car.theta], dtype=np.float32),
                "gt_vx":    np.array([self.car.vx], dtype=np.float32),
                "gt_vy":    np.array([self.car.vy], dtype=np.float32),
                "gt_omega": np.array([self.car.omega], dtype=np.float32),
                "gt_delta": np.array([self.car.delta], dtype=np.float32),
            })

        if (self.should_include_obs_for_vx_sensor):
            obs_dict.update({"vx": np.array([self.car.vx], dtype=np.float32)})
        else:
            info_dict.update({"vx": np.array([self.car.vx], dtype=np.float32)})

        if (self.should_include_obs_for_closest_distance_to_line):
            obs_dict.update({"dist_to_line": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"dist_to_line": np.array([0.0], dtype=np.float32)})

        if (self.should_include_obs_for_heading_angle_relative_to_line):
            obs_dict.update({"heading_rel_to_line": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"heading_rel_to_line": np.array([0.0], dtype=np.float32)})

        if (self.should_include_obs_for_heading_angle_gyro):
            obs_dict.update({"heading_gyro": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"heading_gyro": np.array([0.0], dtype=np.float32)})

        if (self.should_include_obs_for_accel_in_body_frame_x):
            obs_dict.update({"accel_x": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"accel_x": np.array([0.0], dtype=np.float32)})

        if (self.should_include_obs_for_accel_in_body_frame_y):
            obs_dict.update({"accel_y": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"accel_y": np.array([0.0], dtype=np.float32)})

        if (self.should_include_obs_for_look_ahead_line_coords_in_body_frame):
            obs_dict.update({"look_ahead_line_coords": np.zeros((2,self.look_ahead_line_coords_in_body_frame_num_points), dtype=np.float32)})
        else:
            info_dict.update({"look_ahead_line_coords": np.zeros((2,self.look_ahead_line_coords_in_body_frame_num_points), dtype=np.float32)})

        if (self.should_include_obs_for_gps_line_coords_in_world_frame):
            obs_dict.update({"gps_line_coords": np.array([0.0], dtype=np.float32)})
        else:
            info_dict.update({"gps_line_coords": np.array([0.0], dtype=np.float32)})

        # Return the two dictionaries
        return obs_dict, info_dict

    def _get_info(self, progress_queries):
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
            info_dict : dictionary
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
                Same definition as for the "_get_info" function
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
        observation, info_dict = self._get_observation_and_info()

        # Get the info dictionary
        info_dict = self._get_info(progress_queries=self.progress_queries)

        # Set the reset point as the previous progress point
        self.previous_progress_at_closest_p = info_dict["progress_at_closest_p"]

        # Render, if necessary
        #if self.render_mode in ["matplotlib"]:
        #    self._render_frame()

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
        observation, info_dict = self._get_observation_and_info()

        # Get the info dictionary
        info_dict = self._get_info(progress_queries=self.progress_queries)
        #info_dict = {}

        # Compute the "terminated" flag
        terminated = False
        if (info_dict["progress_at_closest_p"] >= self.total_road_length_for_termination):
            #print("(prog,tot_len) = ( " + str(info_dict["progress_at_closest_p"]) + " , " + str(self.total_road_length) + " )" )
            terminated = True

        # Set the truncated flag
        truncated = False

        # Set the reward
        # > As the change in progress in this step
        reward = info_dict["progress_at_closest_p"] - self.previous_progress_at_closest_p
        self.previous_progress_at_closest_p = info_dict["progress_at_closest_p"]

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

    def render_matplotlib_animation_of_trajectory(self, px_traj, py_traj, theta_traj, delta_traj, Ts, traj_increment=1, figure_title=None):
        # Create a figure for the animation
        fig_4_ani, axis_4_ani = plt.subplots(1, 1)
        interval_btw_frames_ms = Ts * 1000

        # Plot the road
        self.road.render_road(axis_4_ani)

        # Add a title
        if (figure_title is None):
            fig_4_ani.suptitle('Animation of car', fontsize=12)
        else:
            fig_4_ani.suptitle(figure_title, fontsize=12)

        # Plot the start position
        car_handles = self.car.render_car(axis_4_ani,px_traj[0],py_traj[0],theta_traj[0],delta_traj[0],scale=1.0)

        # Zoom into the start position
        self.render_matplotlib_zoom_to(px=px_traj[0],py=py_traj[0],x_width=20,y_height=20,axis_handle=axis_4_ani)

        # Display that that animation creation is about to start
        print("Now creating the animation, this may take some time.")

        # Function for printing each individual frame of the animation
        def animate_one_trajectory_frame(i):
            # Update the car
            self.car.render_car(axis_4_ani,px_traj[i],py_traj[i],theta_traj[i],delta_traj[i],scale=1.0, plot_handles=car_handles)
            # Update the window
            self.render_matplotlib_zoom_to(px=px_traj[i],py=py_traj[i],x_width=20,y_height=20,axis_handle=axis_4_ani)

        # Create the list of trajectory indicies to render
        traj_idx_to_render = range(0,len(px_traj),traj_increment)

        # Create the animation
        ani = animation.FuncAnimation(fig_4_ani, animate_one_trajectory_frame, frames=traj_idx_to_render, interval=interval_btw_frames_ms)

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

    def set_progress_queries_for_generating_observations(self, progress_queries):
        """
        Parameters
        ----------
            "progress_queries" : numpy array, 1-dimensional
            Specifies the values of progress-along-the-road,
            relative to the current position of the car, at
            which the observations should be generated. These
            observations are returned in the "info_dict".
            (Units: meters)

        Returns
        -------
        Nothing
        """
        # Get the progress query values
        self.progress_queries = progress_queries
