import numpy as np
from policies.mpc_solver_for_lin_sys_quad_obj_time_varying import MPCSolverForLinSysQuadObjTimeVarying

# Define the MPC class to implement an MPC policy specific for cruise-control plus lane-keeping
class MPCPolicyForCruiseControlAndLaneKeeping:

    def __init__(self, Ts, N, bicycle_model_parameters, objective_function_parameters, constraint_parameters, osqp_settings = {}):
        """
        Initialize the MPC policy with given parameters.

        Parameters:
            Ts  :  Sampling time for the policy (i.e., the discrete-time step). (Units: seconds)
            ...
        """
        # Size of state and action space
        # > These are fixed for this policy class
        self.n_s = 4
        self.n_a = 2

        # Hard-code in a scaling for the Drive Force action to convert
        # it from [-100,100] percentage to a unit range [-1,1] for improved
        # numerical scaling of the optimization program
        self.Fcmd_scaling = 0.01

        # Extract the sampling time (time step) provided
        self.Ts = Ts

        # Extract the prediction horizon provided
        self.N = N

        # Extract the necessary parameters from the bicycle model
        self.Cm = bicycle_model_parameters["Cm"] / self.Fcmd_scaling
        self.m = bicycle_model_parameters["m"]
        self.Lf = bicycle_model_parameters["Lf"]
        self.Lr = bicycle_model_parameters["Lr"]
        self.delta_request_max = bicycle_model_parameters["delta_request_max"]
        self.Ddelta_lower_limit = bicycle_model_parameters["Ddelta_lower_limit"]
        self.Ddelta_upper_limit = bicycle_model_parameters["Ddelta_upper_limit"]

        # Extract the objective function coefficients provided
        self.Qv      =  objective_function_parameters.get("Qv",       0.0)
        self.Qd      =  objective_function_parameters.get("Qd",       0.0)
        self.Qmu     =  objective_function_parameters.get("Qmu",      0.0)
        self.Qdelta  =  objective_function_parameters.get("Qdelta",   0.0)
        self.RFcmd   =  objective_function_parameters.get("RFcmd",    0.0)
        self.RDdelta =  objective_function_parameters.get("RDdelta",  0.0)
        self.qv      =  objective_function_parameters.get("qv",       0.0)
        self.qd      =  objective_function_parameters.get("qd",       0.0)
        self.qmu     =  objective_function_parameters.get("qmu",      0.0)
        self.qdelta  =  objective_function_parameters.get("qdelta",   0.0)
        self.rFcmd   =  objective_function_parameters.get("rFcmd",    0.0)
        self.rDdelta =  objective_function_parameters.get("rDdelta",  0.0)

        # Extract the Drive Action (Fcmd) constraint bounds
        self.Fcmd_lower = constraint_parameters.get("Fcmd_lower", -100.0) * self.Fcmd_scaling
        self.Fcmd_upper = constraint_parameters.get("Fcmd_upper",  100.0) * self.Fcmd_scaling

        # Extract the Drive Action (Fcmd) rate-of-change bounds
        self.Fcmd_roc_lower = constraint_parameters.get("Fcmd_roc_lower", self.Fcmd_lower) * self.Fcmd_scaling
        self.Fcmd_roc_upper = constraint_parameters.get("Fcmd_roc_upper", self.Fcmd_upper) * self.Fcmd_scaling
        
        # Extract the soft constraint coefficients for soft upper bound on speed
        self.s_soft_con_lin_coeff  = constraint_parameters.get("s_soft_con_lin_coeff",  1.0)
        self.s_soft_con_quad_coeff = constraint_parameters.get("s_soft_con_quad_coeff", 1.0)

        # Call the function to construct the dynamics model
        # > This is just here to exemplify
        # > The model is re-constructed based on the current speed observation
        #   every time that compute action is called.
        temp_v = 55.0/3.6
        self.construct_model_for_given_v(temp_v)

        # Construct the objective function coefficients for MPC
        self.construct_objective_function_coefficients_for_mpc()

        # Construct the action constraint vector
        self.a_lower = np.array([[self.Fcmd_lower],[self.Ddelta_lower_limit]], dtype=np.float64)
        self.a_upper = np.array([[self.Fcmd_upper],[self.Ddelta_upper_limit]], dtype=np.float64)

        # Construct the action rate-of-change constraint vector
        self.a_roc_lower = np.array([[self.Fcmd_roc_lower],[-np.inf]], dtype=np.float64)
        self.a_roc_upper = np.array([[self.Fcmd_roc_upper],[ np.inf]], dtype=np.float64)

        # Construct the state constraint vector
        self.s_lower = np.array([[-np.inf],[-np.inf],[-np.inf],[-self.delta_request_max]], dtype=np.float64)
        self.s_upper = np.array([[ np.inf],[ np.inf],[ np.inf],[ self.delta_request_max]], dtype=np.float64)

        # Initialize the generic MPC solver class
        self.mpc_solver = MPCSolverForLinSysQuadObjTimeVarying(self.n_s, self.n_a, dtype=np.float64)

        # Apply the OSQP settings
        self.mpc_solver.set_osqp_settings(osqp_settings)

        # Set the parameters of the MPC solver class
        # > Set the prediction horizon
        self.mpc_solver.set_horizon(self.N, should_check_inputs=True)

        # > Set the linear system model matrices "A" and "B"
        self.mpc_solver.set_model(self.A, self.B, g=None, should_check_inputs=True)

        # > Set the objective function coefficients
        self.mpc_solver.set_objective_function_coefficients(self.Q, self.R, self.q, self.r, self.QN, self.qN, should_check_inputs=True)

        # > Set the state reference for the objective function
        #   - This is just here to exemplify
        #   - The state reference is set every time that compute action is called.
        s_ref = np.array([[temp_v], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.mpc_solver.set_state_reference_for_objective_function(s_ref, should_check_inputs=True)

        # > Set the action reference for the objective function
        a_ref = np.array([[0.0], [0.0]], dtype=np.float64)
        self.mpc_solver.set_action_reference_for_objective_function(a_ref, should_check_inputs=True)

        # > Set the action box constraints
        self.mpc_solver.set_action_box_constraints(self.a_lower, self.a_upper, should_check_inputs=True)

        # > Set the action rate-of-change constraints
        self.mpc_solver.set_action_rate_of_change_constraints(self.a_roc_lower, self.a_roc_upper, should_check_inputs=True)

        # > Set the state box constraints
        self.mpc_solver.set_state_box_constraints(self.s_lower, self.s_upper, should_check_inputs=True)

        # > Set the state soft constraint
        #   - This is just here to exemplify
        #   - The state soft constraint is set every time that compute action is called.
        s_soft_lower = None
        s_soft_upper = np.array([[temp_v],[np.inf],[np.inf],[np.inf]])
        self.mpc_solver.set_state_soft_box_constraint_bounds(s_soft_lower, s_soft_upper, should_check_inputs=True)

        # > Set the state soft constraint objective coefficients
        s_soft_lin_coeff  = np.full((self.n_s, 1), self.s_soft_con_lin_coeff)
        s_soft_quad_coeff = np.full((self.n_s, 1), self.s_soft_con_quad_coeff)
        self.mpc_solver.set_state_soft_box_constraint_coefficients(s_soft_lin_coeff, s_soft_quad_coeff, should_check_inputs=True)

        # > Set the initial condition
        #   - This is just here to exemplify
        #   - The initial state condition is set every time that compute action is called.
        s0 = np.array([[temp_v], [0.0], [0.0], [0.0]])
        self.mpc_solver.set_initial_condition(s0, should_check_inputs=True)

        # Call the reset function
        self.reset()



    def reset(self):
        """
        Reset the previous optimization solution
        """
        # Reset the linearization speed for dynamics
        self.a_prediction_previous = None
        self.s_prediction_previous = None

        # Set the previous action as zero, which is needed for the rate-of-change constraint
        a_prev = np.array([[0.0], [0.0]], dtype=np.float64)
        self.mpc_solver.set_previous_action(a_prev, should_check_inputs=True)



    def construct_model_for_given_v(self, v):
        self.A = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, self.Ts * v, 0.5 * self.Ts**2 * v**2 /(self.Lf + self.Lr)],
            [0.0, 0.0,         1.0, self.Ts * v /(self.Lf + self.Lr)],
            [0.0, 0.0,         0.0, 1.0],
        ], dtype=np.float64)
        self.B = np.array([
            [self.Ts*self.Cm/self.m, 0.0],
            [0.0, (1.0/6.0) * self.Ts**3 * v**2 /(self.Lf + self.Lr)],
            [0.0, 0.5 * self.Ts**2 * v /(self.Lf + self.Lr)],
            [0.0, self.Ts],
        ], dtype=np.float64)



    def construct_objective_function_coefficients_for_mpc(self):
        self.Q  = np.diag(np.array([self.Qv, self.Qd, self.Qmu, self.Qdelta], dtype=np.float64))
        self.R  = np.diag(np.array([self.RFcmd, self.RDdelta], dtype=np.float64))
        self.q  = np.array([[self.qv], [self.qd], [self.qmu], [self.qdelta]], dtype=np.float64)
        self.r  = np.array([[self.rFcmd], [self.rDdelta]], dtype=np.float64)
        self.QN = self.Q.copy()
        self.qN = self.q.copy()



    def predict_progress(self, a_pred, s_pred):
        """
        Predict the progress of the vehicle based on the predicted state and action trajectory.

        The equation for progress is:
        progress_{k+1} = progress_k + T_s * v_k + 0.5 T_s^2 1/m C_m/100.0 Fcmd_k
        """
        # Initialise the array to fill in
        progress_predicted = np.zeros(self.N+1, dtype=np.float64)

        # Leave progress_predicted[0] = 0.0 because this function is computing
        # predicted progress relative to the current location

        # The input predictions are from the previous time step, hence we skip
        # the first element of those predictions
        N = self.N
        n_s = self.n_s
        n_a = self.n_a
        Ts2_Cm_m = 0.5 * self.Ts**2 * self.Cm / self.m
        # > For k=1,...,N-1 of the prediction (hence iterating over k=0,...,N-2)
        for k in range(N-1):
            progress_predicted[k+1] = progress_predicted[k] + self.Ts * s_pred[(k+1)*n_s] + Ts2_Cm_m * a_pred[(k+1)*n_a]
        # > For k=N-1 of the prediction (use the same last action)
        progress_predicted[N] = progress_predicted[self.N-1] + self.Ts * s_pred[(N)*n_s] + Ts2_Cm_m * a_pred[(N-1)*n_a]

        return progress_predicted



    def compute_action(self, observation, info_dict, terminated, truncated):
        """
        Execute the policy using the given observation.

        Parameters:
            observation: A dictionary with sensor readings:
                - 'vx_sensor': The current speed from the vehicle's speed sensor.
                - 'steering_angle_sensor': The current steering angle of the vehicle's front wheels.
                - 'distance_to_closest_point': The current distance to the lane center.
                - 'heading_angle_relative_to_line': The current heading angle relative to the line.
                - 'recommended_speed': The recommeded speed of the closest road segment
                - 'next_recommended_speed': The recommeded speed of the next road segment
                - 'distance_to_next_recommended_speed': The distance to the next road segment

        Returns:
            A numpy array [cc_action, lk_action] with the actions for
            cruise control and lane keeping respectively.
        """
        # ------------------------
        # EXTRACT THE OBSERVATIONS
        vx_sensor = observation["vx_sensor"][0]
        steering_angle_sensor = observation["steering_angle_sensor"][0]
        distance_to_closest_point = observation["distance_to_closest_point"][0]
        heading_angle_relative_to_line = -observation["heading_angle_relative_to_line"][0]
        recommended_speed = observation["recommended_speed"][0]
        next_recommended_speed = observation["next_recommended_speed"][0]
        distance_to_next_recommended_speed = observation["distance_to_next_recommended_speed"][0]

        # ------------------------------------
        # UPDATE THE INITIAL STATE ESTIMATE
        s0 = np.array([[vx_sensor],[distance_to_closest_point],[heading_angle_relative_to_line],[steering_angle_sensor]], dtype=np.float64)
        self.mpc_solver.set_initial_condition(s0, should_check_inputs=False)

        # -----------------------------------
        # PREDICT THE PROGRESS OF THE VEHICLE
        # > If not previous solve, then add a steady state prediction
        if (self.a_prediction_previous is None) or (self.s_prediction_previous is None):
            # Repeat the initial state N+1 times
            self.s_prediction_previous = np.tile(s0.ravel(), self.N+1)
            # Repeat a zero action
            self.a_prediction_previous = np.tile(np.array([0.0,0.0], dtype=np.float64), self.N)
        # > Call the function to predict progress
        predicted_progress = self.predict_progress(self.a_prediction_previous, self.s_prediction_previous)

        # -----------------------------------------------------------------------
        # CONSTRUCT AND SET THE STATE REFERENCE AND SOFT CONSTRAINTS AND DYNAMICS
        # > Get the index where the predicted progress crosses the next recommended speed
        progress_mask = predicted_progress >= distance_to_next_recommended_speed
        if np.any(progress_mask):
            # Get the time index for when the vehicle is predicted to reach the speed change
            idx = np.argmax(progress_mask)
            # Define the two references
            s_ref_current = np.array([[recommended_speed],      [0.0], [0.0], [0.0]], dtype=np.float64)
            s_ref_next    = np.array([[next_recommended_speed], [0.0], [0.0], [0.0]], dtype=np.float64)
            # Set the time-varying state reference
            s_ref_list = [s_ref_current.copy() for _ in range(idx)] + [s_ref_next.copy() for _ in range(self.N-idx)]
            self.mpc_solver.set_time_varying_state_references_for_objective_function(
                sk_ref_list=s_ref_list,
                sN_ref=s_ref_next,
                should_check_inputs=False,
            )
            
            # Set the time-varying state soft constraints
            s_soft_upper_current = np.array([[recommended_speed],      [np.inf],[np.inf],[np.inf]], dtype=np.float64)
            s_soft_upper_next    = np.array([[next_recommended_speed], [np.inf],[np.inf],[np.inf]], dtype=np.float64)
            s_soft_lower_list = None
            if idx == 0:
                s_soft_upper_list = [s_soft_upper_next.copy() for _ in range(self.N)]
            else:
                s_soft_upper_list = [s_soft_upper_current.copy() for _ in range(idx-1)] + [s_soft_upper_next.copy() for _ in range(self.N-idx+1)]
            self.mpc_solver.set_time_varying_state_soft_box_constraint_bounds(
                s_soft_lower_list=s_soft_lower_list,
                s_soft_upper_list=s_soft_upper_list,
                should_check_inputs=False,
            )

            # Set the time-varyind model
            # > Construct the model for the current speed
            self.construct_model_for_given_v(recommended_speed)
            A_current = self.A.copy()
            B_current = self.B.copy()
            # > Construct the model for the next speed
            self.construct_model_for_given_v(next_recommended_speed)
            A_next = self.A.copy()
            B_next = self.B.copy()
            # > Construct the list of time-varying models
            A_list = [A_current for _ in range(idx)] + [A_next for _ in range(self.N-idx)]
            B_list = [B_current for _ in range(idx)] + [B_next for _ in range(self.N-idx)]
            # > Set the list
            self.mpc_solver.set_time_varying_model(Ak_list=A_list, Bk_list=B_list, should_check_inputs=False)

        else:
            # Set the time-invariant state reference
            s_ref = np.array([[recommended_speed], [0.0], [0.0], [0.0]], dtype=np.float64)
            self.mpc_solver.set_state_reference_for_objective_function(s_ref, should_check_inputs=False)
            # Set the time-invariant state soft constraints
            s_soft_lower = None
            s_soft_upper = np.array([[recommended_speed],[np.inf],[np.inf],[np.inf]])
            self.mpc_solver.set_state_soft_box_constraint_bounds(s_soft_lower, s_soft_upper, should_check_inputs=True)
            # Set the time-invariant model
            self.construct_model_for_given_v(recommended_speed)
            self.mpc_solver.set_model(self.A, self.B, g=None, should_check_inputs=True)



        # ----------------------------------
        # SOLVE THE MPC OPTIMIZATION PROGRAM
        a_0, a_pred, s_pred, osqp_status = self.mpc_solver.solve()

        if (osqp_status != "solved"):
            print("[MPCPolicy] ERROR: OSQP returned solve status = " + osqp_status)
            cc_action = 0.0
            lk_action = 0.0
            self.a_prediction_previous = None
            self.s_prediction_previous = None
        else:
            cc_action = a_0[0] / self.Fcmd_scaling
            lk_action = steering_angle_sensor + a_0[1] * self.Ts
            self.a_prediction_previous = a_pred
            self.s_prediction_previous = s_pred

            # NOTE: the lk_action computed above is identical to s_pred[7]

        # -----------------------------------
        # Return the actions as a numpy array
        return np.array([cc_action, lk_action], dtype=np.float32)
