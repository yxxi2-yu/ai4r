#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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
        initial_state_bounds
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

            initial_state_bounds
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

        Returns
        -------
        Nothing
        """

        # OBSERVATION SPACE
        # > Observations are state of the vehicle
        # > For readability, we use a dictionary
        self.observation_space = spaces.Dict(
            {
                "px":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "py":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "theta": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "vx":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "vy":    spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "omega": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "delta": spaces.Box(low=-0.5*np.pi, high=0.5*np.pi, shape=(1,), dtype=np.float32),
            }
        )

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
        if ("num_steps_per_Ts" in numerical_integration_parameters):
            self.integration_num_steps_per_Ts = numerical_integration_parameters["num_steps_per_Ts"]
        else:
            self.integration_num_steps_per_Ts = 1

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



    def _get_observation(self):
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
        return {
            "px":    np.array([self.car.px], dtype=np.float32),
            "py":    np.array([self.car.py], dtype=np.float32),
            "theta": np.array([self.car.theta], dtype=np.float32),
            "vx":    np.array([self.car.vx], dtype=np.float32),
            "vy":    np.array([self.car.vy], dtype=np.float32),
            "omega": np.array([self.car.omega], dtype=np.float32),
            "delta": np.array([self.car.delta], dtype=np.float32),
        }

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
        observation = self._get_observation()

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
        observation = self._get_observation()

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


#env.unwrapped.render_matplotlib_plot_car(px,py,theta,delta)

    def render_matplotlib_zoom_to(self, px, py, x_width, y_height):
        if (self.figure is not None):
            # Set the axis limits
            self.axis.set_xlim(xmin=px-0.5*x_width,  xmax=px+0.5*x_width)
            self.axis.set_ylim(ymin=py-0.5*y_height, ymax=py+0.5*y_height)

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
