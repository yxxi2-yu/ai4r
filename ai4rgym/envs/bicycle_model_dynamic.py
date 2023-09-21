#!/usr/bin/env python

import numpy as np
from scipy import integrate

class BicycleModelDynamic:
    """
    This class simulates the evolution of a dynamic bicycle model with:

    States:
    - px     :  x-position in the world frame (units: m)
    - py     :  y-position in the world frame (units: m)
    - theta  :  angle of car body relative to x-axis of the world frame (units: rad)
    - vx     :  x-velocity in the body frame (units: m/s)
    - vy     :  y-velocity in the body frame (units: m/s)
    - omega  :  angular velocity of car body (units: rad/s)
    - delta  :  steering angle of front wheel relative of the body of the car (units: rad)

    Actions of the dynamic model:
    - Fcmd    :  drive force command for force applied to the rear wheel (units: [-100,100] %)
    - Ddelta  :  steering rate of the front wheel (units: rad/s)

    Actions that can be requested of this simulator:
    - drive_command_request  :  directly the drive command input of the dynamic model (units: [-100,100] %)
    - delta_request          :  steering rate of the front wheel (units: [-100,100] %)

    Parameters:
    - L   :  wheel-base length, i.e., distance between front and rear axels (units: m)
    - Cm  :  motor constant that convert the drive command to force applied to the wheel (units: N/"100% command")
    - Cd  :  coefficient of aerodynamics drag that opposes the direction of motion (units: N / (m/s)^2))
    - Dp  :  Pacejka's tyre formala "peak" coefficient
    - Cp  :  Pacejka's tyre formala "shape" coefficient
    - Bp  :  Pacejka's tyre formala "stiffness" coefficient
    - Ep  :  Pacejka's tyre formala "curvature" coefficient

    - v_transition_min : the velocity below which the eom simulate a purely kinematic model (units: m/s)
    - v_transition_min : the velocity above which the eom simulate a purely dynamic model (units: m/s)

    Bounds on the states:
    - No explicit bounds. The request bounds below implicitly limit the v and delta states.

    Bounds on the actions:
    - delta_request_max   : upper limit of allowable delta request (lower limit is taken as the negative)
    - Ddelta_lower_bound  : lower limit of allowable Ddelta
    - Ddelta_upper_bound  : upper limit of allowable Ddelta
    """



    def __init__(self, model_parameters):
        """
        Initialization function for the "BicycleModelDynamic" class.

        The default value for the Pacejka's tire formala coefficients (peak,
        shape, stiffness, curvature) are taken from the following:
        > Source: https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
        > Source: https://au.mathworks.com/help/sdl/ref/tireroadinteractionmagicformula.html
        The details are repeated here for convenience:
        ------------------------------------------------------------------------
            Name         Typical range   Typical values for longitudinal forces
                                         Dry tarmac   Wet tarmac   Snow   Ice
        ------------------------------------------------------------------------
        D   Peak         0.1  -  1.9       1.0          0.82        0.3    0.1
        C*  Shape        1.0  -  2.0       1.9          2.3         2.0    2.0
        B   Stiffness    4.0  - 12.0      10.0         12.0         5.0    4.0
        E   Curvature  -10.0  -  1.0       0.97         1.0         1.0    1.0
        ------------------------------------------------------------------------
        * Note for coefficient C: The Pacekja model specifies the shape as
          C=1.65 for the longitudinal force and C=1.3 for the lateral force.

        Parameters
        ----------
            model_parameters : float
                A dictionary for the constant paramters that describe the
                dynamic bicycle model. See the class definition above for the
                notation and defintion of each parameter.

        Returns
        -------
        Nothing
        """
        # Extract the parameters
        # > For vehicle dimension
        self.Lf = model_parameters["Lf"]
        self.Lr = model_parameters["Lr"]
        # > For vehicle mass and inertia
        self.m  = model_parameters["m"]
        self.Iz = model_parameters["Iz"]
        # > For motor constant
        self.Cm = model_parameters["Cm"]
        # > For drag coefficient
        self.Cd = model_parameters["Cd"]
        # > For Pacejka's tyre formala coefficient (peak, shape, stiffness, curvature)
        self.Dp_dry =  1.0  if ("Dp_dry" not in model_parameters) else model_parameters["Dp_dry"]
        self.Cp_dry =  1.9  if ("Cp_dry" not in model_parameters) else model_parameters["Cp_dry"]
        self.Bp_dry = 10.0  if ("Bp_dry" not in model_parameters) else model_parameters["Bp_dry"]
        self.Ep_dry =  0.97 if ("Ep_dry" not in model_parameters) else model_parameters["Ep_dry"]

        self.Dp_wet =  0.82 if ("Dp_wet" not in model_parameters) else model_parameters["Dp_wet"]
        self.Cp_wet =  2.3  if ("Cp_wet" not in model_parameters) else model_parameters["Cp_wet"]
        self.Bp_wet = 12.0  if ("Bp_wet" not in model_parameters) else model_parameters["Bp_wet"]
        self.Ep_wet =  1.0  if ("Ep_wet" not in model_parameters) else model_parameters["Ep_wet"]

        self.Dp_snow =  0.3 if ("Dp_snow" not in model_parameters) else model_parameters["Dp_snow"]
        self.Cp_snow =  2.0 if ("Cp_snow" not in model_parameters) else model_parameters["Cp_snow"]
        self.Bp_snow =  5.0 if ("Bp_snow" not in model_parameters) else model_parameters["Bp_snow"]
        self.Ep_snow =  1.0 if ("Ep_snow" not in model_parameters) else model_parameters["Ep_snow"]

        self.Dp_ice =  0.1 if ("Dp_ice" not in model_parameters) else model_parameters["Dp_ice"]
        self.Cp_ice =  2.0 if ("Cp_ice" not in model_parameters) else model_parameters["Cp_ice"]
        self.Bp_ice =  4.0 if ("Bp_ice" not in model_parameters) else model_parameters["Bp_ice"]
        self.Ep_ice =  1.0 if ("Ep_ice" not in model_parameters) else model_parameters["Ep_ice"]

        # Syntax note:
        # value = <value_if_true> if <expression> else <value_if_false>


        # > For parametric uncertainty in the steering angle
        self.delta_offset = model_parameters["delta_offset"]
        # > For limits on the steering angle that can be requested
        self.delta_request_max = model_parameters["delta_request_max"]
        # > For limits on the rate-of-change of steering angle
        self.Ddelta_lower_limit = model_parameters["Ddelta_lower_limit"]
        self.Ddelta_upper_limit = model_parameters["Ddelta_upper_limit"]

        # > For the range ovre which to transition from kinematic to dynamic
        #   bicycle model
        if ("v_transition_min" in model_parameters):
            self.v_transition_min = model_parameters["v_transition_min"]
        else:
            self.v_transition_min = 3
        if ("v_transition_max" in model_parameters):
            self.v_transition_max = model_parameters["v_transition_max"]
        else:
            self.v_transition_max = 3


        # Construct a dictionary of the model parameters are needed for the equations of motion
        # > This allows the numerical integration function to be written in a general fashion that is essential identical for any system.
        self.model_params_for_eom = {
            "Lf" : self.Lf,
            "Lr" : self.Lr,
            "m"  : self.m,
            "Iz" : self.Iz,
            "Cm" : self.Cm,
            "Cd" : self.Cd,
            "Dp" : self.Dp_dry,
            "Cp" : self.Cp_dry,
            "Bp" : self.Bp_dry,
            "Ep" : self.Ep_dry,
            "v_transition_min" : self.v_transition_min,
            "v_transition_max" : self.v_transition_max,
        }

        # Initialize the state variables
        # > These are for readability when accessing this class
        self.px    = 0.0
        self.py    = 0.0
        self.theta = 0.0
        self.vx    = 0.0
        self.vy    = 0.0
        self.omega = 0.0
        self.delta = 0.0

        # Initialize the action variables
        self._Fcmd   = 0.0
        self._Ddelta = 0.0

        # Initialize the actions requested
        self._drive_command_request  = 0.0
        self._delta_request = -self.delta_offset



    def reset(self, px = 0, py = 0, theta = 0, vx = 0, vy = 0, omega = 0, delta = 0):
        """
        Reset the states, actions, and requested actions.

        Parameters
        ----------
            px : float
                Value to assign the "px" state for the reset (units: m)
            py : float
                Value to assign the "py" state for the reset (units: m)
            theta : float
                Value to assign the "theta" state for the reset (units: radians)
            vx : float
                Value to assign the "vx" state for the reset (units: m/s)
            vy : float
                Value to assign the "vy" state for the reset (units: m/s)
            omega : float
                Value to assign the "omega" state for the reset (units: rad/s)
            delta : float
                Value to assign the "delta" state for the reset (units: radians)

        Returns
        -------
        Nothing
        """
        # Set the states to the values provided
        # > Impose any lower and upper of the state elements
        self.px    = px
        self.py    = py
        self.theta = theta
        self.vx    = vx
        self.vy    = vy
        self.omega = omega
        self.delta = max( -self.delta_request_max, min( delta , self.delta_request_max ))

        # Reset the actions and requested actions to zero
        self._drive_command_request  = 0.0
        self._delta_request = -self.delta_offset



    def set_action_requests(self, drive_command_request = 0, delta_request = 0):
        """
        Set the action request values, clipping them to their allowable range.

        Parameters
        ----------
            drive_command_request : float
                The requested drive command for the force applied to the drive
                wheel, i.e., the accelerator position.
                (units: N/"100% command")
            delta_request : float
                The requested value for the steering angle, i.e., the target.
                (units: radians)

        Returns
        -------
        Nothing
        """
        # Clip the actions to their respective request limits
        self._drive_command_request  = max( -100.0, min( drive_command_request , 100.0 ))
        self._delta_request = max( -self.delta_request_max, min( delta_request, self.delta_request_max ))



    @staticmethod
    def eom_kinematic_bicycle(t, s, a, mp):
        """
        Compute the equations-of-motion (eom) for the dynamic bicycle model.

        The eom for the bicycle, derived for the drive force being applied at
        the rear wheel, are:

        px_dot    = vx * cos(theta) - vy * sin(theta)
        py_dot    = vx * sin(theta) + vy * cos(theta)
        theta_dot = omega
        vx_dot    = (1/m)  * (F   - Fyf sin(delta) + m v_y omega )
        vy_dot    = (1/m)  * (Fyr + Fyf cos(delta) - m v_x omega )
        omega_dot = (1/Iz) * (-Fyr lr + Fyf lf cos(delta))

        Where:
        F = (Fcmd/100.0) * Cm - Cd * vx^2
        Fyf = Fzf Dp sin( C atan( B alpha_f - E()))
        Fyr = Fzr Dp sin( C atan( B alpha_r - E()))
        alpha_f = atan( (vy + lf omega) / vx ) - delta
        alpha_r = atan( (vy - lf omega) / vx )

        See the class definition above for the meaning of this notation.

        Parameters
        ----------
            t : float
                Time for a time-varying eom. This is not used by this function
                because the kinematic bicycle model is time-invariant. However,
                "t" is required as an input parameters because it is expected by
                the scipy.integrate numercial integration functions.
            s : numpy array
                The states, with ordering convention:
                [ px  , py  , theta, vx  , vy  , omega, delta ], hence mapping as:
                [ s[0], s[1], s[2] , s[3], s[4], s[5] , s[6]  ]
                This is expected to be a one-dimensional array.
            a : numpy array
                The actions, with ordering convention:
                [ Fcmd, Ddelta ], hence mapping as:
                [ a[0], a[1]   ]
                This is expected to be a one-dimensional array.
            mp : dictionary
                A dictionary contain the constant model parameters that appear
                in the eom. For the dynamic bicycle model is the:
                - Car mass "m"
                - Car length center-of-gravity to front and rear axles "Lf", "Lr"
                - Drive motor coefficient "Cm"
                - Aerodynamic drag coefficient "Cd"
                - Pacejka Tyre Model coefficieints "Dp", "Cp", "Bp", "Ep"
                - The kinematic-to-dynamic model velocity transition range
                  "v_transition_min" and "v_transition_max"

        Returns
        -------
        s_dot : numpy array
            The time deriviative of the state, i.e., "s dot", with the same
            ordering convention as "s", i.e.,:
            [px_dot, py_dot, theta_dot, vx_dot, vy_dot, omega_dot, delta_dot]
        """
        # Compute the drive force
        F = a[0] * mp["Cm"] - mp["Cd"] * s[3] * s[3]

        # Get the transition range into local variables
        vt_min = mp["v_transition_min"]
        vt_max = mp["v_transition_max"]

        # Compute the kinematic model "state dot" (only if necessary)
        if (s[3] >= vt_max):
            s_dot_kinematic = np.array([0,0,0,0,0,0,0], dtype=np.float32())
        else:
            # Compute the state dot
            F_on_m = F / mp["m"]
            s_dot_kinematic = np.array([
              s[3] * np.cos(s[2]) - s[4] * np.sin(s[2]) ,
              s[3] * np.sin(s[2]) + s[4] * np.cos(s[2]) ,
              s[5] ,
              F_on_m ,
              ( a[1]*s[3]/(np.cos(s[6])**2) + np.tan(s[6])*F_on_m ) * mp["Lr"] / (mp["Lf"]+mp["Lr"]),
              ( a[1]*s[3]/(np.cos(s[6])**2) + np.tan(s[6])*F_on_m ) * 1.0      / (mp["Lf"]+mp["Lr"]),
              a[1] ,
              ], dtype=np.float32)
            #print(np.tan(s[6]))

        # Compute the dynamic model "state dot" (only if necessary)
        if (s[3] <= vt_min):
            s_dot_dynamic = np.array([0,0,0,0,0,0,0], dtype=np.float32())
        else:
            # Compute the front wheel slip angle
            alpha_f = -np.arctan( (s[4]+mp["Lf"]*s[5])/s[3]) + s[6]
            # Compute the rear wheel slip angle
            alpha_r = -np.arctan( (s[4]-mp["Lr"]*s[5])/s[3])
            # Compute the front wheel normal force
            Fzf = 9.81 * mp["m"] * mp["Lr"] / (mp["Lf"]+mp["Lr"])
            # Compute the rear wheel normal force
            Fzr = 9.81 * mp["m"] * mp["Lf"] / (mp["Lf"]+mp["Lr"])
            # Compute the front wheel lateral force
            Fyf = Fzf * mp["Dp"] * np.sin( mp["Cp"] * np.arctan( mp["Bp"] * alpha_f - mp["Ep"] * ( mp["Bp"] * alpha_f - np.arctan(mp["Bp"] * alpha_f))))
            # Compute the rear wheel lateral force
            Fyr = Fzr * mp["Dp"] * np.sin( mp["Cp"] * np.arctan( mp["Bp"] * alpha_r - mp["Ep"] * ( mp["Bp"] * alpha_r - np.arctan(mp["Bp"] * alpha_r))))

            # The Pacejka Tire Formula used above is:
            # > F_lateral = Fz · D · sin(C · arctan(B·slip – E · (B·slip – arctan(B·slip))))


            # Compute the state dot
            s_dot_dynamic = np.array([
              s[3] * np.cos(s[2]) - s[4] * np.sin(s[2]) ,
              s[3] * np.sin(s[2]) + s[4] * np.cos(s[2]) ,
              s[5] ,
              (1/mp["m"])  * ( F  - Fyf * np.sin(s[6]) ) + s[4] * s[5],
              (1/mp["m"])  * (Fyr + Fyf * np.cos(s[6]) ) - s[3] * s[5],
              (1/mp["Iz"]) * (-Fyr * mp["Lr"] + Fyf * mp["Lf"] * np.cos(s[6])),
              a[1] ,
              ], dtype=np.float32)

        # Compute the "state dot" that blends the kinemaitc and dynamic models
        if (s[3] >= vt_max):
            s_dot = np.array(s_dot_dynamic, dtype=np.float32)
        elif (s[3] <= vt_min):
            s_dot = np.array(s_dot_kinematic, dtype=np.float32)
        else:
            # Compute the blending factor
            blend_factor = max( 0.0, min( (s[3]-vt_min)/(vt_max-vt_min) , 1.0))
            # Compute the blended "state dot"
            s_dot = blend_factor * s_dot_dynamic + (1.0-blend_factor) * s_dot_kinematic

        # Return "state dot":
        return s_dot



    def perform_integration_step(
            self,
            Ts,
            method,
            num_steps = 1,
            should_update_state = False,
            road_condition = None,
            Dp=None, Cp=None, Bp=None, Ep=None
        ):
        """
        Performs numerical integration, using multiple time-steps if specified, to approximate the state after Ts seconds.

        Parameters
        ----------
            Ts : float
                duration for each integration time step (units: seconds).
            method : string
                the numerical integration technique to use.
            num_steps : integer (default = 0)
                the number of integration time steps to perform.
            should_update_state : bool (default = False)
                Boolean flag for whether the class variable for the state should be updated with the result to be returned
            road_condition : string
                Specifies which parameter values to use for determining the
                tire forces that arise due to slippage at high speeds.
                Options: { "dry" , "wet" , "snow" , "ice" , "other" }
                If "other" is specified, then must specify all four of the
                following parameters.
            Dp, Cp, Bp, Ep : float
                The coefficients of Pacejka's Tire Formula, which are used when
                the "road_condition" key is set to "other"

        Returns
        -------
        px, py, theta, vx, vy, omega, delta : float, float, float, float, float, float, float
            The state approximation after the system evolves for Ts seconds, returned as each element of the state separately.
        """

        # Convert the requested actions into the dynamic model actions
        # > For steering angle:
        delta_target = 1.0 * self._delta_request + self.delta_offset
        Ddelta_target = (delta_target - self.delta) / (Ts*num_steps)
        self._Ddelta = max( self.Ddelta_lower_limit , min( Ddelta_target , self.Ddelta_upper_limit ))

        # > For velocity:
        #v_target = self._v_request
        #Dv_target = (v_target - self.v) / (Ts*num_steps)
        #self._Dv = max( self.Dv_lower_limit , min( Dv_target , self.Dv_upper_limit ))

        # > For drive force command
        self._Fcmd = self._drive_command_request

        # Construct the state vector
        s = np.array([self.px, self.py, self.theta, self.vx, self.vy, self.omega, self.delta], dtype=np.float32)

        # Construct the action vector
        a = np.array([self._Fcmd, self._Ddelta], dtype=np.float32)

        # Get the (pointer to the) model parameters into a local variable
        model_params = self.model_params_for_eom

        # If necessary, update the tire mode parameters baesd on the given
        # "road_condition" parameter
        if (road_condition is not None):
            if (road_condition == "dry"):
                model_params["Dp"] = self.Dp_dry
                model_params["Cp"] = self.Cp_dry
                model_params["Bp"] = self.Bp_dry
                model_params["Ep"] = self.Ep_dry
            if (road_condition == "wet"):
                model_params["Dp"] = self.Dp_wet
                model_params["Cp"] = self.Cp_wet
                model_params["Bp"] = self.Bp_wet
                model_params["Ep"] = self.Ep_wet
            if (road_condition == "snow"):
                model_params["Dp"] = self.Dp_snow
                model_params["Cp"] = self.Cp_snow
                model_params["Bp"] = self.Bp_snow
                model_params["Ep"] = self.Ep_snow
            if (road_condition == "ice"):
                model_params["Dp"] = self.Dp_ice
                model_params["Cp"] = self.Cp_ice
                model_params["Bp"] = self.Bp_ice
                model_params["Ep"] = self.Ep_ice
            if (road_condition == "other"):
                model_params["Dp"] = Dp
                model_params["Cp"] = Cp
                model_params["Bp"] = Bp
                model_params["Ep"] = Ep

        # Iterate over the requested number of time step
        for i_step in np.arange(0,num_steps):

            # Switch for the requested integration method
            if (method == "euler"):
                # Compute in one line
                s_plus = s + Ts * self.eom_kinematic_bicycle(0, s, a, model_params)

            elif (method == "midpoint"):
                # Compute Euler’s method for half a time step
                s_half = s + 0.5 * Ts * self.eom_kinematic_bicycle(0, s, a, model_params)
                # Compute full step based on gradient at the half step state (i.e., at the midpoint)
                s_plus = s + Ts * self.eom_kinematic_bicycle(0, s_half, a, model_params)

            elif (method == "huen"):
                # Compute Euler’s method for a full time step
                s_euler = s + Ts * self.eom_kinematic_bicycle(0, s, a, model_params)
                # Compute update based on half gradient at the start and the end
                s_plus = s + 0.5* Ts * (self.eom_kinematic_bicycle(0, s, a, model_params) + self.eom_kinematic_bicycle(0, s_euler, a, model_params))

            elif (method == "rk4"):
                # Compute the eom for the RK4 sequence of states
                m1 = self.eom_kinematic_bicycle(0, s, a, model_params)
                m2 = self.eom_kinematic_bicycle(0, s+0.5*Ts*m1, a, model_params)
                m3 = self.eom_kinematic_bicycle(0, s+0.5*Ts*m2, a, model_params)
                m4 = self.eom_kinematic_bicycle(0, s+Ts*m3, a, model_params)
                # Compute the state update
                s_plus = s + (1/6) * Ts * (m1 + 2*m2 + 2*m3 + m4)

            elif (method == "rk45"):
                # Python function links:
                # > scipy.integrate.RK45: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html#scipy.integrate.RK45
                # > scipy.integrate.solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

                # Call the RK45 method
                sol = integrate.solve_ivp( self.eom_kinematic_bicycle, [0,Ts], s, method="RK45", t_eval=None, args=[a, model_params], vectorized=False)

                # Copy across the solution
                s_plus = np.copy(sol.y[:,-1])

            else:
                # Log this as an error
                print("ERROR: integration method \"" + str(method) + "\" not recognized")
                s_plus = np.zeros(s.shape(), dtype=np.float32)

            # Update the state variable that is local to this function
            s = np.copy(s_plus)

        # Numerical integration is now complete

        # Update the class variable (if requested)
        if (should_update_state):
            self.px    = s[0]
            self.py    = s[1]
            self.theta = s[2]
            self.vx    = s[3]
            self.vy    = s[4]
            self.omega = s[5]
            self.delta = s[6]

        # Return the state components
        return s[0], s[1], s[2] , s[3] , s[4] , s[5] , s[6]



    def get_actions(self):
        """
        Get the value of the current actions

        Parameters
        ----------
            None

        Returns
        -------
        Fcmd, Ddelta : float, float
            Drive force command and the rate-of-change of steering actions.
        """
        # Return the actions
        return self._Fcmd, self._Ddelta



    def render_car(self, axis_handle, px, py, theta, delta, scale=1.0):
        """
        Plot the road.

        Parameters
        ----------
            axis_handle : matplotlib.axes
                A handle for where the road is plotted.
            px : float
            py : float
            theta : float
            delta : float
            scale : float

        Returns
        -------
            plot_handles : [matplotlib.lines.Line2D]
                The handles to the lines that are plotted.
        """
        plot_handles = []
        # Get the length parameters
        Lf = self.Lf * scale
        Lr = self.Lr * scale
        w_radius = 0.125 * np.sqrt(Lf**2 + Lr**2)
        # Compute the front and rear wheel locations
        pxf = px + Lf * np.cos(theta)
        pyf = py + Lf * np.sin(theta)

        pxr = px - Lr * np.cos(theta)
        pyr = py - Lr * np.sin(theta)

        # Plot the body
        this_handle = axis_handle.plot( [pxr,pxf], [pyr,pyf], linewidth=1, color="black" )
        plot_handles.append( this_handle )

        # Compute the front wheel
        wfxf = pxf + w_radius * np.cos(theta+delta)
        wfyf = pyf + w_radius * np.sin(theta+delta)
        wfxr = pxf - w_radius * np.cos(theta+delta)
        wfyr = pyf - w_radius * np.sin(theta+delta)
        # Plot the front wheel
        this_handle = axis_handle.plot( [wfxr,wfxf], [wfyr,wfyf], linewidth=2, color="black" )
        plot_handles.append( this_handle )

        # Compute the rear wheel
        wrxf = pxr + w_radius * np.cos(theta)
        wryf = pyr + w_radius * np.sin(theta)
        wrxr = pxr - w_radius * np.cos(theta)
        wryr = pyr - w_radius * np.sin(theta)
        # Plot the rear wheel
        this_handle = axis_handle.plot( [wrxr,wrxf], [wryr,wryf], linewidth=2, color="black" )
        plot_handles.append( this_handle )

        # # Iterate through the road elements
        # for element_idx, this_isStraight  in enumerate(self.__isStraight):
        #     if (this_isStraight):
        #         # Directly plot the straight line
        #         this_handles = axis_handle.plot( [self.__start_points[element_idx,0],self.__end_points[element_idx,0]], [self.__start_points[element_idx,1],self.__end_points[element_idx,1]] )
        #     else:
        #         # Plot circle by gridding the angle range
        #         this_angles = -np.sign(self.__c[element_idx]) * 0.5 * np.pi + np.linspace( start=self.__start_angles[element_idx], stop=self.__end_angles[element_idx], num=max(2,round(0.5*abs(self.__phi[element_idx])*(180/np.pi))) )
        #         this_x = self.__arc_centers[element_idx,0] + (1/abs(self.__c[element_idx])) * np.cos(this_angles)
        #         this_y = self.__arc_centers[element_idx,1] + (1/abs(self.__c[element_idx])) * np.sin(this_angles)
        #         this_handles = axis_handle.plot(this_x, this_y)

        #     for handle in this_handles: plot_handles.append( handle )

        return plot_handles