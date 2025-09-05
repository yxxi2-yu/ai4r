import numpy as np
from scipy import sparse
from scipy import linalg
import osqp

class MPCSolverForLinSysQuadObjTimeVarying:
    """
    This class implements a general-purpose MPC solver with:
        - Most parameters able to be diferent at each time step, i.e.: time-varying.
        - A linear (affine) system for the state-evolution constraint, i.e.:
            s_{k+1} = A_k*s_k + B_k*a_k + g_k
        - Quadratic cost, with the possibility to set a reference
            objective =
                sum_{k=0}^{N}(
                      (s_k-s_ref,k)^T Q_k (s_k-s_ref,k)
                    + (a_k-a_ref,k)^T R_k (a_k-a_ref,k)
                    + q_k^T s_k
                    + r_k^T a_k
                )
                + (s_N-s_ref,N)^T Q_N (s_N-s_ref,N)
                + q_N^T s_N
        - Action space constraint: lower and upper bounds
            l_a,k <= a_k <= u_a,k
        - State space constraints: lower and upper bound soft constraints
            l_a,k <= s_k + lambda_k,    0 <= lambda_k
            u_a,k >= s_k - mu_k,        0 <= mu_k

    Class variables
    ---------------
        - To be added.
    """

    def __init__(self, n_s, n_a, dtype=np.float64):
        """
        Initialization function for the class.
        All parameters initialize to None, except for:
        * The dimension of the state and action spaces.
        * The default data type.

        Parameters
        ----------
            n_s : integer
                Number of states (must be > 0)
            n_a : integer
                Number of actions (must be > 0)
            dtype : np.dtype-like
                Floating dtype to use internally (np.float32 or np.float64)

        Returns
        -------
            Nothing
        """
        # Set the state and action dimensions
        assert isinstance(n_s, (int, np.integer)), f"[MPC SOLVER] ASSERTION: n_s must be an integer, got {type(n_s)}"
        assert isinstance(n_a, (int, np.integer)), f"[MPC SOLVER] ASSERTION: n_a must be an integer, got {type(n_a)}"
        assert n_s > 0, f"[MPC SOLVER] ASSERTION: n_s must be > 0, got {n_s}"
        assert n_a > 0, f"[MPC SOLVER] ASSERTION: n_a must be > 0, got {n_a}"
        self.n_s = int(n_s)
        self.n_a = int(n_a)

        # Set the data type to be applied to everything
        self.allowed_dtypes = (np.dtype(np.float32), np.dtype(np.float64))
        dtype_norm = np.dtype(dtype)
        assert dtype_norm in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype must be np.float32 or np.float64, got {dtype_norm}"
        self.dtype = dtype_norm

        # Initialize all the other class variables to be "None"

        # Verbosity control
        self.silent_mode = False

        # OSQP Solver object and setting
        self.osqp_solver_object = None
        self.osqp_settings = {}

        # Prediction horizon
        self.N = None

        # The model, in a Linear time-varying (LTV) form:
        #     s_{k+1} = A_k*s_k + B_k*a_k + g_k
        self.Ak = None
        self.Bk = None
        self.gk = None
        self.is_time_invariant_lin_sys_model = False

        # Derived details for the model
        self.Ak_sparse = None
        self.Bk_sparse = None

        # Objective function, in a quadratic form:
        #     sum_{k=0}^{N}(
        #           (s_k-s_ref,k)^T Q_k (s_k-s_ref,k)
        #         + (a_k-a_ref,k)^T R_k (a_k-a_ref,k)
        #         + q_k^T s_k
        #         + r_k^T a_k
        #     )
        #     + (s_N-s_ref,N)^T Q_N (s_N-s_ref,N)
        #     + q_N^T s_N
        self.Qk = None
        self.Rk = None
        self.qk = None
        self.rk = None
        self.QN = None
        self.qN = None
        self.is_time_invariant_objective_function = False

        # Derived details for the objective
        self.Qk_sparse = None
        self.Rk_sparse = None
        self.QN_sparse = None


        # Initial condition, i.e., the state at the start of the prediction horizon.
        self.s_0 = np.zeros((n_s,1), dtype=self.dtype)
        self._s_0_was_updated = False

        # Previous action for action rate-of-change constraint at k=0
        self.a_previous = np.zeros((n_a,1), dtype=self.dtype)
        self._a_previous_was_updated = False
        self.a0_from_previous_solve = None

        # Objective function reference for states, i.e., "s_ref" in the above.
        self.sk_ref = None
        self.sN_ref = None
        self.is_time_invariant_state_reference  = False

        # Objective function reference for actions, i.e., "a_ref" in the above.
        self.ak_ref = None
        self.is_time_invariant_action_reference = False

        # Action constraints: box constraints.
        # > I.e., lower and upper bounds on each element:
        #     l_a,k <= a_k <= u_a,k    for k=0..N-1
        self.ak_lower = None
        self.ak_upper = None
        self.is_time_invariant_action_box_constraint = False

        # Action constraints: rate of change constraints.
        # > I.e.. as a lower and upper bound on each element:
        #     l_a,roc,0 <= a_0 - a_{"previous"} <= u_a,roc,0
        #     l_a,roc,k <= a_k - a_{k-1} <= u_a,roc,k    for k=1..N-1
        self.ak_roc_lower = None
        self.ak_roc_upper = None
        self.is_time_invariant_action_roc_constraint = False

        # Derived details for state soft box constraint
        self.n_ak_roc_constraints = None
        self.ak_roc_constraint_selector_matrix = None
        # > A slice for where to update the full "l" and "u" vectors
        #   in the constraints for the k=0 rate-of-chance constraint,
        #   which needs the a_{"previous"} value as an input parameter.
        self._row_slice_a0roc = slice(0, 0)

        # State constraints: box constraints.
        # > I.e., lower and upper bounds on each element:
        #     l_s,k <= s_k <= u_s,k   for k=1..N
        self.sk_lower = None
        self.sk_upper = None
        self.is_time_invariant_state_box_constraint = False

        # Derived details for state soft box constraint
        self.n_sk_box_constraints = None
        self.sk_box_constraint_selector_matrix = None

        # State constraints: soft box constraints.
        # > I.e., soft constraint for lower and upper bound on each element:
        #     l_s,soft,k <= s_k + lambda_k,    0 <= lambda_k
        #     u_s,soft,k >= s_k - mu_k,        0 <= mu_k
        self.sk_soft_lower = None
        self.sk_soft_upper = None
        self.is_time_invariant_state_soft_box_constraint_bounds = False

        # Derived details for state soft box constraint
        self.n_sk_soft_lower = None
        self.n_sk_soft_upper = None
        self.sk_soft_lower_selector_matrix = None
        self.sk_soft_upper_selector_matrix = None

        # Quadratic and linear coefficients for soft state constraint
        #     objective terms: 
        #     lambda_k^T diag(sk_soft_quad_coeff) lambda_k + sk_soft_lin_coeff^T lambda_k
        self.sk_soft_lin_coeff  = None
        self.sk_soft_quad_coeff = None
        self.is_time_invariant_state_soft_box_constraint_coefficients = False



        # Polygon constraints for state and actions, i.e.:
        #     H_ineq_s*s <= h_ineq_s
        #     H_ineq_s*a <= h_ineq_a
        #self.H_ineq_s = None
        #self.h_ineq_s = None
        #self.H_ineq_a = None
        #self.h_ineq_a = None

        # Polygon constraints combining state and actions, i.e.:
        #     H_ineq_sa*[s;a] <= h_ineq_sa
        #self.H_ineq_sa = None
        #self.h_ineq_sa = None

        # The number of each type of polygon constraint
        #self.n_ineq_s = 0
        #self.n_ineq_a = 0
        #self.n_ineq_sa = 0

        # A flag that the MPC optmization formulation needs to be re-built
        self._rebuild_needed = True


    def set_osqp_settings(self, osqp_settings: dict = {}):
        self.osqp_settings = osqp_settings



    def _log(self, level: str, message: str):
        """
        Internal logger with simple level handling:
        - "error": always prints
        - "warn"/"warning" and "info": print only if `silent_mode` is False
        - anything else: print (conservative default)
        """
        lvl = (level or "").lower()
        if lvl == "error":
            print(message)
            return
        if lvl in ("warn", "warning", "info", "information"):
            if not getattr(self, "silent_mode", False):
                print(message)
            return
        # default: print
        print(f"Log with level = {level}, and message = {message}")



    def set_horizon(self, N: int, should_check_inputs: bool = True):
        """
        Set the prediction horizon.

        Parameters
        ----------
            N : integrer
                The length on the MPC prediction horizon in number of discrete-time steps.
                Must be > 0
        """
        # Assert that the horion length must be positive
        assert N > 0, f"[MPC SOVLER] ASSERTION: Horizon length N must be positive, got {N}"
        # Assert that the horion length must be a positive integer
        if should_check_inputs:
            assert isinstance(N, (int, np.integer)), f"[MPC SOLVER] ASSERTION: N must be an integer, got {type(N)}"
            assert N > 0, f"[MPC SOLVER] ASSERTION: Horizon length N must be positive, got {N}"

            # If time-varying data already exists, warn (don't assert) on mismatches
            pairs = [
                ("Ak", self.Ak),
                ("Bk", self.Bk),
                ("gk", self.gk),
                ("Qk", self.Qk),
                ("Rk", self.Rk),
                ("Sk", getattr(self, "Sk", None) if hasattr(self, "Sk") else None),
                ("qk", self.qk),
                ("rk", self.rk),
                ("sk_ref", self.sk_ref),
                ("ak_ref", self.ak_ref),
                ("ak_lower", self.ak_lower),
                ("ak_upper", self.ak_upper),
                ("sk_soft_quad_coeff", self.sk_soft_quad_coeff),
                ("sk_soft_lin_coeff", self.sk_soft_lin_coeff),
            ]
            for name, lst in pairs:
                if lst is not None and len(lst) != int(N):
                    self._log("warn", f"[MPC SOLVER] WARNING: {name} length {len(lst)} != N {int(N)}")

        # Set the class variable
        self.N = int(N)
        # Update the rebuild flag:
        self._rebuild_needed = True


    def set_model(self, A: np.ndarray, B: np.ndarray, g: np.ndarray | None = None, should_check_inputs: bool = True):
        """
        Sets the linear-system model matrices to be time-invariant (i..e, LTI).
        Stores the LTI model internally as time-indexed lists, i.e.:
            [A_0, ..., A_{N-1}],
            [B_0, ..., B_{N-1}],
            [g_0, ..., g_{N-1}] (optional),
        because this make the "build MPC" functions simpler.

        Parameters
        ----------
            A : (n_s, n_s) ndarray
                State transition matrix (time-invariant).
            B : (n_s, n_a) ndarray
                Action influence matrix (time-invariant).
            g : (n_s, 1) ndarray, optional
                Constant vector (time-invariant).
                If omitted, sets to None

        Effects
        -------
        - self.Ak, self.Bk, self.gk become lists of length N with copies of A, B, g.
        - self.is_time_invariant_lin_sys_model = True
        - self._rebuild_needed = True
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOVLER] ERROR: Horizon length self.N is not set.")

        # Assert the type and size of the input arguments
        if (should_check_inputs):
            assert isinstance(A, np.ndarray), f"[MPC SOVLER] ASSERTION: Matrix A must be a numpy array, got {type(A)}"
            assert isinstance(B, np.ndarray), f"[MPC SOVLER] ASSERTION: Matrix B must be a numpy array, got {type(B)}"
            assert A.shape == (self.n_s, self.n_s), f"[MPC SOVLER] ASSERTION: Matrix A must be ({self.n_s}, {self.n_s}), got {A.shape}"
            assert B.shape == (self.n_s, self.n_a), f"[MPC SOVLER] ASSERTION: Matrix B must be ({self.n_s}, {self.n_a}), got {B.shape}"
            assert A.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of matrix A must be one of {self.allowed_dtypes}, got {A.dtype}"
            assert B.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of matrix B must be one of {self.allowed_dtypes}, got {B.dtype}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        A = A.astype(self.dtype, copy=False)
        B = B.astype(self.dtype, copy=False)

        # Use copies to avoid aliasing across time indices
        self.Ak = [A.copy() for _ in range(self.N)]
        self.Bk = [B.copy() for _ in range(self.N)]

        # Process the constant vector in the same fashion
        if (g is not None):
            if (should_check_inputs):
                assert isinstance(g, np.ndarray), f"[MPC SOVLER] ASSERTION: Vector g must be a numpy array, got {type(g)}"
                assert g.shape == (self.n_s, 1), f"[MPC SOVLER] ASSERTION: Vector g must be ({self.n_s}, 1), got {g.shape}"
                assert g.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of vector g must be one of {self.allowed_dtypes}, got {g.dtype}"
            g = g.astype(self.dtype, copy=False)
            self.gk = [g.copy() for _ in range(self.N)]
        else:
            self.gk = None

        # Update the flags
        self.is_time_invariant_lin_sys_model = True
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_model(should_check_inputs=should_check_inputs)



    def set_time_varying_model(self, Ak_list, Bk_list, gk_list: list[np.ndarray] | None = None, should_check_inputs: bool = True):
        """
        Sets the linear-system model matrices to be time-varying (i.e., LTV).
        Stores the model as time-indexed lists:
            [A_0, ..., A_{N-1}],
            [B_0, ..., B_{N-1}],
            [g_0, ..., g_{N-1}] (optional).

        Parameters
        ----------
            Ak_list : list of (n_s, n_s) ndarrays
                State transition matrices per time step.
            Bk_list : list of (n_s, n_a) ndarrays
                Action influence matrices per time step.
            gk_list : list of (n_s, 1) ndarrays, optional
                Constant vectors per time step. If omitted, sets to None.

        Effects
        -------
        - self.Ak, self.Bk, self.gk become lists of length N with copies of inputs.
        - self.N is set to len(Ak_list), only if self.N is None
        - self.is_time_invariant_lin_sys_model = False
        - self._rebuild_needed = True
        """
        # Assert the type and length of the input arguments
        if (should_check_inputs):
            assert isinstance(Ak_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: Ak_list must be a list/tuple, got {type(Ak_list)}"
            assert isinstance(Bk_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: Bk_list must be a list/tuple, got {type(Bk_list)}"
            assert len(Ak_list) == len(Bk_list), f"[MPC SOLVER] ASSERTION: Ak_list and Bk_list must have the same length, got {len(Ak_list)} and {len(Bk_list)}"
        
        # Update the horizon length if it is None
        if (self.N is None):
            if should_check_inputs:
                assert len(Ak_list) > 0, "[MPC SOLVER] ASSERTION: Ak_list must have a positive length"
            self._log("info", f"[MPC SOLVER] INFO: the horizon length (self.N) is None, hence setting it to {len(Ak_list)}, which is the length of AK_list")
            self.N = len(Ak_list)
        else:
            # Otherwise, we do NOT update self.N because multiple functions can add time-varying data
            # Assert the type and horizon length agrees:
            if (should_check_inputs):
                assert len(Ak_list) == self.N, f"[MPC SOLVER] ASSERTION: The length of Ak_list must match the horizon length self.N, got {len(Ak_list)} versus {self.N}"
        
        # Assert the type and shape of the input arguments
        if (should_check_inputs):
            for k, (Ak, Bk) in enumerate(zip(Ak_list, Bk_list)):
                assert isinstance(Ak, np.ndarray), f"[MPC SOLVER] ASSERTION: A_{k} must be numpy array, got {type(Ak)}"
                assert isinstance(Bk, np.ndarray), f"[MPC SOLVER] ASSERTION: B_{k} must be numpy array, got {type(Bk)}"
                assert Ak.shape == (self.n_s, self.n_s), f"[MPC SOLVER] ASSERTION: A_{k} must be ({self.n_s},{self.n_s}), got {Ak.shape}"
                assert Bk.shape == (self.n_s, self.n_a), f"[MPC SOLVER] ASSERTION: B_{k} must be ({self.n_s},{self.n_a}), got {Bk.shape}"
                assert Ak.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of A_{k} must be one of {self.allowed_dtypes}, got {Ak.dtype}"
                assert Bk.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of B_{k} must be one of {self.allowed_dtypes}, got {Bk.dtype}"
        

        # Store A_k and B_k matrices (auto-cast + defensive copies)
        self.Ak = [Ak.astype(self.dtype, copy=False).copy() for Ak in Ak_list]
        self.Bk = [Bk.astype(self.dtype, copy=False).copy() for Bk in Bk_list]

        # Process the constant vector in the same fashion
        if gk_list is not None:
            if (should_check_inputs):
                assert isinstance(gk_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: gk_list must be a list/tuple, got {type(gk_list)}"
                assert len(gk_list) == self.N, f"[MPC SOLVER] ASSERTION: gk_list length must match horizon length self.N, got {len(gk_list)} versus {self.N}"
                for k, gk in enumerate(gk_list):
                    assert isinstance(gk, np.ndarray), f"[MPC SOLVER] ASSERTION: g_{k} must be numpy array, got {type(gk)}"
                    assert gk.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: g_{k} must be ({self.n_s},1), got {gk.shape}"
                    assert gk.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of g_{k} must be one of {self.allowed_dtypes}, got {gk.dtype}"
            # Store gk vectors (auto-cast + defensive copies)
            self.gk = [gk.astype(self.dtype, copy=False).copy() for gk in gk_list]
        else:
            self.gk = None

        # Update the flags
        self.is_time_invariant_lin_sys_model = False
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_model(should_check_inputs=should_check_inputs)



    def _compute_derived_details_for_model(self, should_check_inputs: bool = True):
        """
        Precompute sparse CSC blocks for the time-varying model:
            Ak_sparse[k] := csc(Ak[k])  shape (n_s, n_s), dtype=self.dtype
            Bk_sparse[k] := csc(Bk[k])  shape (n_s, n_a), dtype=self.dtype

        Assumes Ak/Bk have already been validated and cast in the setters.
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")
        if should_check_inputs:
            assert self.Ak is not None and self.Bk is not None, "[MPC SOLVER] ASSERTION: Ak and Bk must be set."
            assert len(self.Ak) == self.N and len(self.Bk) == self.N, "[MPC SOLVER] ASSERTION: Ak and Bk must have length N."

        n_s, n_a = self.n_s, self.n_a
        dt = self.dtype

        # Build CSC once; sparse.csc_matrix will wrap if already CSC
        self.Ak_sparse = [sparse.csc_matrix(Ak, shape=(n_s, n_s), dtype=dt) for Ak in self.Ak]
        self.Bk_sparse = [sparse.csc_matrix(Bk, shape=(n_s, n_a), dtype=dt) for Bk in self.Bk]



    def set_objective_function_coefficients(
        self,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        qN: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set the quadratic and linear coefficients of the objective function.

        The whole objective function takes the form:
            sum_{k=0}^{N}(
                  (s_k-s_ref,k)^T Q_k (s_k-s_ref,k)
                + (a_k-a_ref,k)^T R_k (a_k-a_ref,k)
                + q_k^T s_k
                + r_k^T a_k
            )
            + (s_N-s_ref,N)^T Q_N (s_N-s_ref,N)
            + q_N^T s_N

        All the input parameters of this function are optional.
        For any paremeter not provided (or provided as None), that coefficient
        is set to zero during the building of the MPC optimization formulation.
        
        Parameters
        ----------
            Q : numpy array of size n_s -by- n_s
                The quadratic coefficient of the state
            R : numpy array of size n_a -by- n_a
                The quadratic coefficient of the action
            q : numpy array of length n_s
                The linear coefficient of the state
            r : numpy array of legnth n_a
                The linear coefficient of the action
            QN : numpy array of size n_s -by- n_s
                The quadratic coefficient of the terminal state
            qN : numpy array of length n_s
                The linear coefficient of the terminal state
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOVLER] ERROR: Horizon length self.N is not set.")

        # Assert the type and size of the input arguments
        if (should_check_inputs):
            input_checks = [
                ("Q",  Q,  (self.n_s, self.n_s)),
                ("R",  R,  (self.n_a, self.n_a)),
                ("q",  q,  (self.n_s, 1)),
                ("r",  r,  (self.n_a, 1)),
                ("QN", QN, (self.n_s, self.n_s)),
                ("qN", qN, (self.n_s, 1)),
            ]
            for name, arr, shape in input_checks:
                if arr is None:
                    continue
                assert isinstance(arr, np.ndarray), f"[MPC SOLVER] ASSERTION: {name} must be a numpy array, got {type(arr)}"
                assert arr.shape == shape, f"[MPC SOLVER] ASSERTION: {name} must be {shape}, got {arr.shape}"
                assert arr.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: dtype of {name} must be one of {self.allowed_dtypes}, got {arr.dtype}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        Q  = Q.astype(self.dtype,  copy=False)  if Q  is not None else None
        R  = R.astype(self.dtype,  copy=False)  if R  is not None else None
        q  = q.astype(self.dtype,  copy=False)  if q  is not None else None
        r  = r.astype(self.dtype,  copy=False)  if r  is not None else None
        QN = QN.astype(self.dtype, copy=False)  if QN is not None else None
        qN = qN.astype(self.dtype, copy=False)  if qN is not None else None
        
        # Use copies to avoid aliasing across time indices
        self.Qk = [Q.copy() for _ in range(self.N)] if Q is not None else None
        self.Rk = [R.copy() for _ in range(self.N)] if R is not None else None
        self.qk = [q.copy() for _ in range(self.N)] if q is not None else None
        self.rk = [r.copy() for _ in range(self.N)] if r is not None else None

        # Terminal terms (defensive copies)
        self.QN = QN.copy() if QN is not None else None
        self.qN = qN.copy() if qN is not None else None

        # Update the flags
        self.is_time_invariant_objective_function = True
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_objective_mats(should_check_inputs=should_check_inputs)



    def set_time_varying_objective_function_coefficients(
        self,
        Qk_list: list[np.ndarray] | None = None,
        Rk_list: list[np.ndarray] | None = None,
        qk_list: list[np.ndarray] | None = None,
        rk_list: list[np.ndarray] | None = None,
        QN: np.ndarray | None = None,
        qN: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set time-varying quadratic and linear coefficients for the objective:

            sum_{k=0}^{N-1}(
                (s_k - s_ref,k)^T Q_k (s_k - s_ref,k)
                + (a_k - a_ref,k)^T R_k (a_k - a_ref,k)
                + q_k^T s_k
                + r_k^T a_k
            )
            + (s_N - s_ref,N)^T QN (s_N - s_ref,N) + qN^T s_N

        Each *_list, if provided, must have length N.
        All the input parameters of this function are optional.
        For any paremeter not provided (or provided as None), that coefficient
        is set to zero during the building of the MPC optimization formulation.

        Parameters
        ----------
            Qk_list : list of (n_s, n_s) ndarrays
                The quadratic coefficient of the state, per time step
            Rk_list : list of (n_a, n_a) ndarrays
                The quadratic coefficient of the action, per time step
            qk_list : list of (n_s, 1) ndarrays, optional
                The linear coefficient of the state, per time step
            rk_list : list of (n_a, 1) ndarrays, optional
                The linear coefficient of the action, per time step
            QN : numpy array of size (n_s, n_s)
                The quadratic coefficient of the terminal state
            qN : numpy array of size (n_s, 1)
                The linear coefficient of the terminal state
        """

        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            # choose the first provided list to infer N
            candidates = [lst for lst in (Qk_list, Rk_list, qk_list, rk_list) if lst is not None]
            if not candidates:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide at least one *_list.")
            inferred_N = len(candidates[0])
            if should_check_inputs:
                for name, lst in (("Qk_list", Qk_list), ("Rk_list", Rk_list), ("qk_list", qk_list), ("rk_list", rk_list)):
                    if lst is not None:
                        assert len(lst) == inferred_N, f"[MPC SOLVER] ASSERTION: {name} length {len(lst)} != inferred N {inferred_N}"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {inferred_N}.")
            self.N = inferred_N
        else:
            if should_check_inputs:
                for name, lst in (("Qk_list", Qk_list), ("Rk_list", Rk_list), ("qk_list", qk_list), ("rk_list", rk_list)):
                    if lst is not None:
                        assert len(lst) == self.N, f"[MPC SOLVER] ASSERTION: {name} length {len(lst)} != self.N {self.N}"

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            def _check_list(name, lst, shape):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, M in enumerate(lst):
                    assert isinstance(M, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(M)}"
                    assert M.shape == shape, f"[MPC SOLVER] ASSERTION: {name}[{k}] must be {shape}, got {M.shape}"
                    assert M.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {M.dtype} not in {self.allowed_dtypes}"

            _check_list("Qk_list", Qk_list, (self.n_s, self.n_s))
            _check_list("Rk_list", Rk_list, (self.n_a, self.n_a))
            _check_list("qk_list", qk_list, (self.n_s, 1))
            _check_list("rk_list", rk_list, (self.n_a, 1))

            if QN is not None:
                assert isinstance(QN, np.ndarray), f"[MPC SOLVER] ASSERTION: QN must be ndarray, got {type(QN)}"
                assert QN.shape == (self.n_s, self.n_s), f"[MPC SOLVER] ASSERTION: QN must be ({self.n_s},{self.n_s}), got {QN.shape}"
                assert QN.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: QN dtype {QN.dtype} not in {self.allowed_dtypes}"
            if qN is not None:
                assert isinstance(qN, np.ndarray), f"[MPC SOLVER] ASSERTION: qN must be ndarray, got {type(qN)}"
                assert qN.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: qN must be ({self.n_s},1), got {qN.shape}"
                assert qN.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: qN dtype {qN.dtype} not in {self.allowed_dtypes}"

        # Store the coefficients (auto-cast + defensive copies)
        self.Qk = [Q.astype(self.dtype, copy=False).copy() for Q in Qk_list] if Qk_list is not None else None
        self.Rk = [R.astype(self.dtype, copy=False).copy() for R in Rk_list] if Rk_list is not None else None
        self.qk = [q.astype(self.dtype, copy=False).copy() for q in qk_list] if qk_list is not None else None
        self.rk = [r.astype(self.dtype, copy=False).copy() for r in rk_list] if rk_list is not None else None

        self.QN = QN.astype(self.dtype, copy=False).copy() if QN is not None else None
        self.qN = qN.astype(self.dtype, copy=False).copy() if qN is not None else None

        # Update the flags
        self.is_time_invariant_objective_function = False
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_objective_mats(should_check_inputs=should_check_inputs)



    def _compute_derived_details_for_objective_mats(self, should_check_inputs: bool = True):
        """
        Precompute sparse CSC blocks for objective matrices:
        - Qk_sparse[k] := csc(Qk[k])  shape (n_s, n_s)
        - Rk_sparse[k] := csc(Rk[k])  shape (n_a, n_a)
        - QN_sparse    := csc(QN)     shape (n_s, n_s)

        Assumes arrays were validated/cast at set-time.
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")
        N, n_s, n_a, dt = self.N, self.n_s, self.n_a, self.dtype

        # Per time-step state cost
        if self.Qk is None:
            self.Qk_sparse = [None] * N
        else:
            if should_check_inputs:
                assert len(self.Qk) == N, f"[MPC SOLVER] ASSERTION: len(Qk) {len(self.Qk)} != N {N}"
            self.Qk_sparse = [
                (sparse.csc_matrix(Qk, shape=(n_s, n_s), dtype=dt) if Qk is not None else None)
                for Qk in self.Qk
            ]

        # Per time-step action cost
        if self.Rk is None:
            self.Rk_sparse = [None] * N
        else:
            if should_check_inputs:
                assert len(self.Rk) == N, f"[MPC SOLVER] ASSERTION: len(Rk) {len(self.Rk)} != N {N}"
            self.Rk_sparse = [
                (sparse.csc_matrix(Rk, shape=(n_a, n_a), dtype=dt) if Rk is not None else None)
                for Rk in self.Rk
            ]

        # Terminal cost
        self.QN_sparse = (
            sparse.csc_matrix(self.QN, shape=(n_s, n_s), dtype=dt) if self.QN is not None else None
        )



    def set_state_reference_for_objective_function(
        self,
        s_ref: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set the state reference that is used in the objective function:
            sum_{k=0}^{N-1}(
                (s_k - s_ref,k)^T Q_k (s_k - s_ref,k)
                + (a_k - a_ref,k)^T R_k (a_k - a_ref,k)
                + q_k^T s_k
                + r_k^T a_k
            )
            + (s_N - s_ref,N)^T QN (s_N - s_ref,N) + qN^T s_N

        Input s_ref may be None; missing terms treated as zeros at build time.
        Expects column vector s_ref to have shape (n_s -by- 1)

        Parameters
        ----------
            s_ref : numpy array of size (n_s,1)
                The state reference
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            if s_ref is not None:
                assert isinstance(s_ref, np.ndarray), f"[MPC SOLVER] ASSERTION: s_ref must be ndarray, got {type(s_ref)}"
                assert s_ref.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_ref must be ({self.n_s},1), got {s_ref.shape}"
                assert s_ref.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_ref dtype {s_ref.dtype} not in {self.allowed_dtypes}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        s_ref  = s_ref.astype(self.dtype,  copy=False) if s_ref  is not None else None

        # Use copies to avoid aliasing across time indices
        self.sk_ref = [s_ref.copy() for _ in range(self.N)] if s_ref is not None else None
        self.sN_ref = s_ref.copy() if s_ref is not None else None

        # Update the flags
        self.is_time_invariant_state_reference = (s_ref is not None)
        self._rebuild_needed = True



    def set_time_varying_state_references_for_objective_function(
        self,
        sk_ref_list: list[np.ndarray] | None = None,
        sN_ref: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set the state and action reference that is used in the objective function:
            sum_{k=0}^{N-1}(
                (s_k - s_ref,k)^T Q_k (s_k - s_ref,k)
                + (a_k - a_ref,k)^T R_k (a_k - a_ref,k)
                + q_k^T s_k
                + r_k^T a_k
            )
            + (s_N - s_ref,N)^T QN (s_N - s_ref,N) + qN^T s_N

        sk_ref_list elements and sN_ref, if provided, must be column vectors (n_s,1).

        Parameters
        ----------
            sk_ref_list : list of (n_s, 1) ndarrays
                The state reference, per time step
            sN_ref : numpy array of size (n_s, 1)
                The terminal state reference
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            if sk_ref_list is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide sk_ref_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(sk_ref_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: sk_ref_list must be list/tuple, got {type(sk_ref_list)}"
                assert len(sk_ref_list) > 0, "[MPC SOLVER] ASSERTION: sk_ref_list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(sk_ref_list)}.")
            self.N = len(sk_ref_list)
        else:
            if should_check_inputs and sk_ref_list is not None:
                assert len(sk_ref_list) == self.N, f"[MPC SOLVER] ASSERTION: sk_ref_list length {len(sk_ref_list)} != self.N {self.N}"

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            if sk_ref_list is not None:
                assert isinstance(sk_ref_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: sk_ref_list must be list/tuple, got {type(sk_ref_list)}"
                for k, sk in enumerate(sk_ref_list):
                    assert isinstance(sk, np.ndarray), f"[MPC SOLVER] ASSERTION: s_ref[{k}] must be ndarray, got {type(sk)}"
                    assert sk.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_ref[{k}] must be ({self.n_s},1), got {sk.shape}"
                    assert sk.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_ref[{k}] dtype {sk.dtype} not in {self.allowed_dtypes}"
            if sN_ref is not None:
                assert isinstance(sN_ref, np.ndarray), f"[MPC SOLVER] ASSERTION: sN_ref must be ndarray, got {type(sN_ref)}"
                assert sN_ref.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: sN_ref must be ({self.n_s},1), got {sN_ref.shape}"
                assert sN_ref.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: sN_ref dtype {sN_ref.dtype} not in {self.allowed_dtypes}"

        # Store the coefficients (auto-cast + defensive copies)
        self.sk_ref = (
            [sk.astype(self.dtype, copy=False).copy() for sk in sk_ref_list]
            if sk_ref_list is not None else None
        )
        self.sN_ref = sN_ref.astype(self.dtype, copy=False).copy() if sN_ref is not None else None

        # Update the flags
        self.is_time_invariant_state_reference = False if sk_ref_list is not None else True
        self._rebuild_needed = True



    def set_action_reference_for_objective_function(
        self,
        a_ref: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set a *time-invariant* action reference a_ref used in:
            sum_{k=0}^{N-1} (a_k - a_ref)^T R_k (a_k - a_ref)

        a_ref may be None; missing terms are treated as zeros at build time.
        Expects column vector a_ref to have shape (n_s -by- 1)

        Parameters
        ----------
            a_ref : numpy array of size (n_a,1)
                The action reference
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Assert the type and shape of the input arguments
        if should_check_inputs and a_ref is not None:
            assert isinstance(a_ref, np.ndarray), f"[MPC SOLVER] ASSERTION: a_ref must be ndarray, got {type(a_ref)}"
            assert a_ref.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_ref must be ({self.n_a},1), got {a_ref.shape}"
            assert a_ref.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_ref dtype {a_ref.dtype} not in {self.allowed_dtypes}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        a_ref = a_ref.astype(self.dtype, copy=False) if a_ref is not None else None

        # Use copies to avoid aliasing across time indices
        self.ak_ref = [a_ref.copy() for _ in range(self.N)] if a_ref is not None else None

        # Update the flags
        self.is_time_invariant_action_reference = (a_ref is not None)
        self._rebuild_needed = True



    def set_time_varying_action_references_for_objective_function(
        self,
        ak_ref_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* action references a_k used in:
            sum_{k=0}^{N-1} (a_k - a_k_ref)^T R_k (a_k - a_k_ref)

        ak_ref_list, if provided, must have length N with each element (n_a,1).
        There is no terminal action reference (no R_N term by convention).

        Parameters
        ----------
            ak_ref_list : list of (n_a, 1) ndarrays
                The action reference, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            if ak_ref_list is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide ak_ref_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(ak_ref_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: ak_ref_list must be list/tuple, got {type(ak_ref_list)}"
                assert len(ak_ref_list) > 0, "[MPC SOLVER] ASSERTION: ak_ref_list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(ak_ref_list)}.")
            self.N = len(ak_ref_list)
        else:
            if should_check_inputs and ak_ref_list is not None:
                assert len(ak_ref_list) == self.N, f"[MPC SOLVER] ASSERTION: ak_ref_list length {len(ak_ref_list)} != self.N {self.N}"

        # Assert the type and shape of the input argument
        if should_check_inputs and ak_ref_list is not None:
            assert isinstance(ak_ref_list, (list, tuple)), f"[MPC SOLVER] ASSERTION: ak_ref_list must be list/tuple, got {type(ak_ref_list)}"
            for k, ar in enumerate(ak_ref_list):
                assert isinstance(ar, np.ndarray), f"[MPC SOLVER] ASSERTION: a_ref[{k}] must be ndarray, got {type(ar)}"
                assert ar.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_ref[{k}] must be ({self.n_a},1), got {ar.shape}"
                assert ar.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_ref[{k}] dtype {ar.dtype} not in {self.allowed_dtypes}"

        # Store the coefficients (auto-cast + defensive copies)
        self.ak_ref = (
            [ar.astype(self.dtype, copy=False).copy() for ar in ak_ref_list]
            if ak_ref_list is not None else None
        )

        # Update the flags
        self.is_time_invariant_action_reference = False if ak_ref_list is not None else True
        self._rebuild_needed = True



    def set_action_box_constraints(
        self,
        a_lower: np.ndarray | None = None,
        a_upper: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-invariant* action bounds, broadcast over k = 0..N-1:
            a_lower <= a_k <= a_upper

        Each of a_lower / a_upper may be None; missing side(s) treated as unbounded at build time.
        Expects column vectors (n_a, 1).

        Parameters
        ----------
            a_lower : numpy array of size (n_a,1)
                The lower bound constraint on the action
            a_upper : numpy array of size (n_a,1)
                The upper bound constraint on the action
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            if a_lower is not None:
                assert isinstance(a_lower, np.ndarray), f"[MPC SOLVER] ASSERTION: a_lower must be ndarray, got {type(a_lower)}"
                assert a_lower.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_lower must be ({self.n_a},1), got {a_lower.shape}"
                assert a_lower.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_lower dtype {a_lower.dtype} not in {self.allowed_dtypes}"
            if a_upper is not None:
                assert isinstance(a_upper, np.ndarray), f"[MPC SOLVER] ASSERTION: a_upper must be ndarray, got {type(a_upper)}"
                assert a_upper.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_upper must be ({self.n_a},1), got {a_upper.shape}"
                assert a_upper.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_upper dtype {a_upper.dtype} not in {self.allowed_dtypes}"
            if a_lower is not None and a_upper is not None:
                assert np.all(a_lower <= a_upper), "[MPC SOLVER] ASSERTION: a_lower must be elementwise <= a_upper"

        # Cast to the dtype of the class (no copy if dtype already matches)
        a_lower = a_lower.astype(self.dtype, copy=False) if a_lower is not None else None
        a_upper = a_upper.astype(self.dtype, copy=False) if a_upper is not None else None

        # Use copies to avoid aliasing across time indices
        self.ak_lower = [a_lower.copy() for _ in range(self.N)] if a_lower is not None else None
        self.ak_upper = [a_upper.copy() for _ in range(self.N)] if a_upper is not None else None

        # Update the flags
        self.is_time_invariant_action_box_constraint = (a_lower is not None or a_upper is not None)
        self._rebuild_needed = True



    def set_time_varying_action_box_constraints(
        self,
        a_lower_list: list[np.ndarray] | None = None,
        a_upper_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* action bounds per k = 0..N-1:
            a_lower_k <= a_k <= a_upper_k

        Each of a_lower_list / a_upper_list may be None; missing side(s) treated as unbounded at build time.
        Lists, if provided, must have length N with elements shaped (n_a, 1).

        Parameters
        ----------
            a_lower_list : list of (n_a, 1) ndarrays
                The lower bound constraint on the action, per time step
            a_upper_list : list of (n_a, 1) ndarrays
                The upper bound constraint on the action, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            first = a_lower_list if a_lower_list is not None else a_upper_list
            if first is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide a_lower_list or a_upper_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(first, (list, tuple)), f"[MPC SOLVER] ASSERTION: bounds list must be list/tuple, got {type(first)}"
                assert len(first) > 0, "[MPC SOLVER] ASSERTION: bounds list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(first)}.")
            self.N = len(first)
        else:
            if should_check_inputs:
                if a_lower_list is not None:
                    assert len(a_lower_list) == self.N, f"[MPC SOLVER] ASSERTION: a_lower_list length {len(a_lower_list)} != self.N {self.N}"
                if a_upper_list is not None:
                    assert len(a_upper_list) == self.N, f"[MPC SOLVER] ASSERTION: a_upper_list length {len(a_upper_list)} != self.N {self.N}"

        # Assert the type and shape of the input argument
        if should_check_inputs:
            def _check_vec_list(name, lst):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, v in enumerate(lst):
                    assert isinstance(v, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(v)}"
                    assert v.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ({self.n_a},1), got {v.shape}"
                    assert v.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {v.dtype} not in {self.allowed_dtypes}"
            _check_vec_list("a_lower_list", a_lower_list)
            _check_vec_list("a_upper_list", a_upper_list)

            if a_lower_list is not None and a_upper_list is not None:
                for k, (lo, hi) in enumerate(zip(a_lower_list, a_upper_list)):
                    assert np.all(lo <= hi), f"[MPC SOLVER] ASSERTION: a_lower_list[{k}] must be elementwise <= a_upper_list[{k}]"

        # Store the coefficients (auto-cast + defensive copies)
        self.ak_lower = (
            [lo.astype(self.dtype, copy=False).copy() for lo in a_lower_list]
            if a_lower_list is not None else None
        )
        self.ak_upper = (
            [hi.astype(self.dtype, copy=False).copy() for hi in a_upper_list]
            if a_upper_list is not None else None
        )

        # Update the flags
        self.is_time_invariant_action_box_constraint = False if (a_lower_list is not None or a_upper_list is not None) else True
        self._rebuild_needed = True



    def set_action_rate_of_change_constraints(
        self,
        a_roc_lower: np.ndarray | None = None,
        a_roc_upper: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-invariant* action rate-of-change bounds, broadcast over k = 0..N-1:
            l_a,roc,0 <= a_0 - a_{"previous"} <= u_a,roc,0
            l_a,roc,k <= a_k - a_{k-1} <= u_a,roc,k    for k=1..N-1

        Each of a_roc_lower / a_roc_upper may be None; missing side(s) treated as unbounded at build time.
        Expects column vectors (n_a, 1).

        Parameters
        ----------
            a_roc_lower : numpy array of size (n_a,1)
                The lower rate-of-change bound constraint on the action
            a_roc_upper : numpy array of size (n_a,1)
                The upper rate-of-change bound constraint on the action
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Validate inputs
        if should_check_inputs:
            if a_roc_lower is not None:
                assert isinstance(a_roc_lower, np.ndarray), f"[MPC SOLVER] ASSERTION: a_roc_lower must be ndarray, got {type(a_roc_lower)}"
                assert a_roc_lower.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_roc_lower must be ({self.n_a},1), got {a_roc_lower.shape}"
                assert a_roc_lower.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_roc_lower dtype {a_roc_lower.dtype} not in {self.allowed_dtypes}"
            if a_roc_upper is not None:
                assert isinstance(a_roc_upper, np.ndarray), f"[MPC SOLVER] ASSERTION: a_roc_upper must be ndarray, got {type(a_roc_upper)}"
                assert a_roc_upper.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_roc_upper must be ({self.n_a},1), got {a_roc_upper.shape}"
                assert a_roc_upper.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_roc_upper dtype {a_roc_upper.dtype} not in {self.allowed_dtypes}"
            if (a_roc_lower is not None) and (a_roc_upper is not None):
                assert np.all(a_roc_lower <= a_roc_upper), "[MPC SOLVER] ASSERTION: a_roc_lower must be elementwise <= a_roc_upper"

        # Cast to dtype
        a_roc_lower = a_roc_lower.astype(self.dtype, copy=False) if a_roc_lower is not None else None
        a_roc_upper = a_roc_upper.astype(self.dtype, copy=False) if a_roc_upper is not None else None

        # Broadcast across horizon with defensive copies
        self.ak_roc_lower = [a_roc_lower.copy() for _ in range(self.N)] if a_roc_lower is not None else None
        self.ak_roc_upper = [a_roc_upper.copy() for _ in range(self.N)] if a_roc_upper is not None else None

        # Update the flags
        self.is_time_invariant_action_roc_constraint = (a_roc_lower is not None or a_roc_upper is not None)
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_action_rate_of_change_constraints()



    def set_time_varying_action_rate_of_change_constraints(
        self,
        a_roc_lower_list: list[np.ndarray] | None = None,
        a_roc_upper_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* action rate-of-change bounds per k = 0..N-1:
            l_a,roc,0 <= a_0 - a_{"previous"} <= u_a,roc,0
            l_a,roc,k <= a_k - a_{k-1} <= u_a,roc,k    for k=1..N-1

        Each of a_roc_lower_list / a_roc_upper_list may be None; missing side(s) treated as unbounded at build time.
        Lists, if provided, must have length N with elements shaped (n_a, 1).

        Parameters
        ----------
            a_roc_lower_list : list of (n_a, 1) ndarrays
                The lower rate-of-change bound constraint on the action, per time step
            a_roc_upper_list : list of (n_a, 1) ndarrays
                The upper rate-of-change bound constraint on the action, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            first = a_roc_lower_list if a_roc_lower_list is not None else a_roc_upper_list
            if first is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide a_roc_lower_list or a_roc_upper_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(first, (list, tuple)), f"[MPC SOLVER] ASSERTION: coefficients list must be list/tuple, got {type(first)}"
                assert len(first) > 0, "[MPC SOLVER] ASSERTION: coefficients list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(first)}.")
            self.N = len(first)
        else:
            if should_check_inputs:
                if a_roc_lower_list is not None:
                    assert len(a_roc_lower_list) == self.N, f"[MPC SOLVER] ASSERTION: a_roc_lower_list length {len(a_roc_lower_list)} != self.N {self.N}"
                if a_roc_upper_list is not None:
                    assert len(a_roc_upper_list) == self.N, f"[MPC SOLVER] ASSERTION: a_roc_upper_list length {len(a_roc_upper_list)} != self.N {self.N}"

        # Validate types and shapes
        if should_check_inputs:
            def _check_vec_list(name, lst):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, v in enumerate(lst):
                    assert isinstance(v, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(v)}"
                    assert v.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ({self.n_a},1), got {v.shape}"
                    assert v.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {v.dtype} not in {self.allowed_dtypes}"
            _check_vec_list("a_roc_lower_list", a_roc_lower_list)
            _check_vec_list("a_roc_upper_list", a_roc_upper_list)

            if a_roc_lower_list is not None and a_roc_upper_list is not None:
                for k, (lo, hi) in enumerate(zip(a_roc_lower_list, a_roc_upper_list)):
                    assert np.all(lo <= hi), f"[MPC SOLVER] ASSERTION: a_roc_lower_list[{k}] must be elementwise <= a_roc_upper_list[{k}]"

        # Store the coefficients (auto-cast + defensive copies)
        self.ak_roc_lower = (
            [lo.astype(self.dtype, copy=False).copy() for lo in a_roc_lower_list]
            if a_roc_lower_list is not None else None
        )
        self.ak_roc_upper = (
            [hi.astype(self.dtype, copy=False).copy() for hi in a_roc_upper_list]
            if a_roc_upper_list is not None else None
        )

        # Update the flags
        self.is_time_invariant_action_roc_constraint = False if (a_roc_lower_list is not None or a_roc_upper_list is not None) else True
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_action_rate_of_change_constraints()



    def _compute_derived_details_for_action_rate_of_change_constraints(self):
        """
        Precompute, per k = 0..N-1, which action elements participate in
        rate-of-change constraints, and build a selector matrix that picks
        exactly those action components. A component participates if it has
        EITHER a finite lower bound OR a finite upper bound at that step.

        Produces lists of length N:
        - n_ak_roc_constraints[k] : number of participating action components
        - ak_roc_constraint_selector_matrix[k] : (n_ak_roc_constraints[k], n_a) CSC selector

        If no components participate at step k, the selector is an empty
        CSC matrix of shape (0, n_a) with dtype=self.dtype.
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")

        n_a = self.n_a

        # Ensure lists exist for iteration; treat missing side as all-infinite per-k
        lower_list = self.ak_roc_lower if self.ak_roc_lower is not None else [None] * self.N
        upper_list = self.ak_roc_upper if self.ak_roc_upper is not None else [None] * self.N

        self.n_ak_roc_constraints = []
        self.ak_roc_constraint_selector_matrix = []

        for k in range(self.N):
            if isinstance(lower_list[k], np.ndarray):
                mask_lo = np.isfinite(lower_list[k][:, 0])
            else:
                mask_lo = np.zeros(n_a, dtype=bool)

            if isinstance(upper_list[k], np.ndarray):
                mask_hi = np.isfinite(upper_list[k][:, 0])
            else:
                mask_hi = np.zeros(n_a, dtype=bool)

            mask_any = np.logical_or(mask_lo, mask_hi)
            idx = np.nonzero(mask_any)[0]
            n_rows = int(idx.size)
            self.n_ak_roc_constraints.append(n_rows)

            if n_rows > 0:
                data = np.ones(n_rows, dtype=self.dtype)
                rows = np.arange(n_rows, dtype=np.int64)
                cols = idx.astype(np.int64, copy=False)
                S = sparse.csc_matrix((data, (rows, cols)), shape=(n_rows, n_a), dtype=self.dtype)
            else:
                S = sparse.csc_matrix((0, n_a), dtype=self.dtype)

            self.ak_roc_constraint_selector_matrix.append(S)


    def set_state_box_constraints(
        self,
        s_lower: np.ndarray | None = None,
        s_upper: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-invariant* state bounds, broadcast over k = 0..N-1:
            s_lower <= s_k <= s_upper    for k=1..N

        Each of s_lower / s_upper may be None; missing side(s) treated as unbounded at build time.
        Expects column vectors (n_s, 1).

        Parameters
        ----------
            s_lower : numpy array of size (n_s,1)
                The lower bound constraint on the state
            s_upper : numpy array of size (n_s,1)
                The upper bound constraint on the state
        """
        # Assert the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Validate inputs
        if should_check_inputs:
            if s_lower is not None:
                assert isinstance(s_lower, np.ndarray), f"[MPC SOLVER] ASSERTION: s_lower must be ndarray, got {type(s_lower)}"
                assert s_lower.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_lower must be ({self.n_s},1), got {s_lower.shape}"
                assert s_lower.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_lower dtype {s_lower.dtype} not in {self.allowed_dtypes}"
            if s_upper is not None:
                assert isinstance(s_upper, np.ndarray), f"[MPC SOLVER] ASSERTION: s_upper must be ndarray, got {type(s_upper)}"
                assert s_upper.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_upper must be ({self.n_s},1), got {s_upper.shape}"
                assert s_upper.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_upper dtype {s_upper.dtype} not in {self.allowed_dtypes}"
            if (s_lower is not None) and (s_upper is not None):
                assert np.all(s_lower <= s_upper), "[MPC SOLVER] ASSERTION: s_lower must be elementwise <= s_upper"

        # Cast to dtype
        s_lower = s_lower.astype(self.dtype, copy=False) if s_lower is not None else None
        s_upper = s_upper.astype(self.dtype, copy=False) if s_upper is not None else None

        # Broadcast across horizon with defensive copies (N entries correspond to k=1..N)
        self.sk_lower = [s_lower.copy() for _ in range(self.N)] if s_lower is not None else None
        self.sk_upper = [s_upper.copy() for _ in range(self.N)] if s_upper is not None else None

        # Update flags and rebuild
        self.is_time_invariant_state_box_constraint = (s_lower is not None or s_upper is not None)
        self._rebuild_needed = True

        # Derived details
        self._compute_derived_details_for_state_box_constraints()



    def set_time_varying_state_box_constraints(
        self,
        s_lower_list: list[np.ndarray] | None = None,
        s_upper_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* state bounds per k = 0..N-1:
            s_lower_k <= s_k <= s_upper_k    for k=1..N

        Each of s_lower_list / s_upper_list may be None; missing side(s) treated as unbounded at build time.
        Lists, if provided, must have length N with elements shaped (n_s, 1).

        Parameters
        ----------
            s_lower_list : list of (n_s, 1) ndarrays
                The lower bound constraint on the state, per time step
            s_upper_list : list of (n_s, 1) ndarrays
                The upper bound constraint on the state, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            first = s_lower_list if s_lower_list is not None else s_upper_list
            if first is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide s_lower_list or s_upper_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(first, (list, tuple)), f"[MPC SOLVER] ASSERTION: coefficients list must be list/tuple, got {type(first)}"
                assert len(first) > 0, "[MPC SOLVER] ASSERTION: coefficients list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(first)}.")
            self.N = len(first)
        else:
            if should_check_inputs:
                if s_lower_list is not None:
                    assert len(s_lower_list) == self.N, f"[MPC SOLVER] ASSERTION: s_lower_list length {len(s_lower_list)} != self.N {self.N}"
                if s_upper_list is not None:
                    assert len(s_upper_list) == self.N, f"[MPC SOLVER] ASSERTION: s_upper_list length {len(s_upper_list)} != self.N {self.N}"

        # Validate types and shapes
        if should_check_inputs:
            def _check_vec_list(name, lst):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, v in enumerate(lst):
                    assert isinstance(v, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(v)}"
                    assert v.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ({self.n_s},1), got {v.shape}"
                    assert v.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {v.dtype} not in {self.allowed_dtypes}"
            _check_vec_list("s_lower_list", s_lower_list)
            _check_vec_list("s_upper_list", s_upper_list)

            if s_lower_list is not None and s_upper_list is not None:
                for k, (lo, hi) in enumerate(zip(s_lower_list, s_upper_list)):
                    assert np.all(lo <= hi), f"[MPC SOLVER] ASSERTION: s_lower_list[{k}] must be elementwise <= s_upper_list[{k}]"

        # Store (auto-cast + defensive copies)
        self.sk_lower = (
            [lo.astype(self.dtype, copy=False).copy() for lo in s_lower_list]
            if s_lower_list is not None else None
        )
        self.sk_upper = (
            [hi.astype(self.dtype, copy=False).copy() for hi in s_upper_list]
            if s_upper_list is not None else None
        )

        # Update flags and rebuild
        self.is_time_invariant_state_box_constraint = False if (s_lower_list is not None or s_upper_list is not None) else True
        self._rebuild_needed = True

        # Derived details
        self._compute_derived_details_for_state_box_constraints()


    def _compute_derived_details_for_state_box_constraints(self):
        """
        Precompute, per k = 1..N, which state elements participate in
        state box constraints, and build a selector matrix that picks
        exactly those state components. A component participates if it has
        EITHER a finite lower bound OR a finite upper bound at that step.

        Produces lists of length N:
        - n_sk_box_constraints[k] : number of participating state components
        - sk_box_constraint_selector_matrix[k] : (n_sk_box_constraints[k], n_s) CSC selector

        If no components participate at step k, the selector is an empty
        CSC matrix of shape (0, n_s) with dtype=self.dtype.
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")

        n_s = self.n_s

        # Ensure lists exist for iteration; treat missing side as all-infinite per-k
        lower_list = self.sk_lower if self.sk_lower is not None else [None] * self.N
        upper_list = self.sk_upper if self.sk_upper is not None else [None] * self.N

        self.n_sk_box_constraints = []
        self.sk_box_constraint_selector_matrix = []

        for k in range(self.N):
            if isinstance(lower_list[k], np.ndarray):
                mask_lo = np.isfinite(lower_list[k][:, 0])
            else:
                mask_lo = np.zeros(n_s, dtype=bool)

            if isinstance(upper_list[k], np.ndarray):
                mask_hi = np.isfinite(upper_list[k][:, 0])
            else:
                mask_hi = np.zeros(n_s, dtype=bool)

            mask_any = np.logical_or(mask_lo, mask_hi)
            idx = np.nonzero(mask_any)[0]
            n_rows = int(idx.size)
            self.n_sk_box_constraints.append(n_rows)

            if n_rows > 0:
                data = np.ones(n_rows, dtype=self.dtype)
                rows = np.arange(n_rows, dtype=np.int64)
                cols = idx.astype(np.int64, copy=False)
                S = sparse.csc_matrix((data, (rows, cols)), shape=(n_rows, n_s), dtype=self.dtype)
            else:
                S = sparse.csc_matrix((0, n_s), dtype=self.dtype)

            self.sk_box_constraint_selector_matrix.append(S)


    def set_state_soft_box_constraint_bounds(
        self,
        s_soft_lower: np.ndarray | None = None,
        s_soft_upper: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-invariant* state bounds, broadcast over k = 0..N-1:
            s_soft_lower <= s_k + lambda_k
                            s_k - mu_k      <= s_soft_upper
                       0 <= lambda_k
                       0 <= mu_k

        Each of s_soft_lower / s_soft_upper may be None; missing side(s) treated as unbounded at build time.
        Expects column vectors (n_s, 1).

        Parameters
        ----------
            s_soft_lower : numpy array of size (n_s,1)
                The soft lower bound constraint on the state
            s_soft_upper : numpy array of size (n_s,1)
                The soft upper bound constraint on the state
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            if s_soft_lower is not None:
                assert isinstance(s_soft_lower, np.ndarray), f"[MPC SOLVER] ASSERTION: s_soft_lower must be ndarray, got {type(s_soft_lower)}"
                assert s_soft_lower.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_soft_lower must be ({self.n_s},1), got {s_soft_lower.shape}"
                assert s_soft_lower.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_soft_lower dtype {s_soft_lower.dtype} not in {self.allowed_dtypes}"
            if s_soft_upper is not None:
                assert isinstance(s_soft_upper, np.ndarray), f"[MPC SOLVER] ASSERTION: s_soft_upper must be ndarray, got {type(s_soft_upper)}"
                assert s_soft_upper.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_soft_upper must be ({self.n_s},1), got {s_soft_upper.shape}"
                assert s_soft_upper.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_soft_upper dtype {s_soft_upper.dtype} not in {self.allowed_dtypes}"
            if s_soft_lower is not None and s_soft_upper is not None:
                assert np.all(s_soft_lower <= s_soft_upper), "[MPC SOLVER] ASSERTION: s_soft_lower must be elementwise <= s_soft_upper"

        # Cast to the dtype of the class (no copy if dtype already matches)
        s_soft_lower = s_soft_lower.astype(self.dtype, copy=False) if s_soft_lower is not None else None
        s_soft_upper = s_soft_upper.astype(self.dtype, copy=False) if s_soft_upper is not None else None

        # Use copies to avoid aliasing across time indices
        self.sk_soft_lower = [s_soft_lower.copy() for _ in range(self.N)] if s_soft_lower is not None else None
        self.sk_soft_upper = [s_soft_upper.copy() for _ in range(self.N)] if s_soft_upper is not None else None

        # Update the flags
        self.is_time_invariant_state_soft_box_constraint_bounds = (s_soft_lower is not None or s_soft_upper is not None)
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_state_soft_constraint_bounds()



    def set_time_varying_state_soft_box_constraint_bounds(
        self,
        s_soft_lower_list: list[np.ndarray] | None = None,
        s_soft_upper_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* state soft bounds per k = 1..N:
            s_soft_lower <= s_k + lambda_k
                            s_k - mu_k      <= s_soft_upper
                       0 <= lambda_k
                       0 <= mu_k

        Each of s_soft_lower_list / s_soft_upper_list may be None; missing side(s) treated as unbounded at build time.
        Lists, if provided, must have length N with elements shaped (n_s, 1).

        Parameters
        ----------
            s_soft_lower_list : list of (n_s, 1) ndarrays
                The soft lower bound constraint on the state, per time step
            s_soft_upper_list : list of (n_s, 1) ndarrays
                The soft upper bound constraint on the state, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            first = s_soft_lower_list if s_soft_lower_list is not None else s_soft_upper_list
            if first is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide s_soft_lower_list or s_soft_upper_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(first, (list, tuple)), f"[MPC SOLVER] ASSERTION: bounds list must be list/tuple, got {type(first)}"
                assert len(first) > 0, "[MPC SOLVER] ASSERTION: bounds list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(first)}.")
            self.N = len(first)
        else:
            if should_check_inputs:
                if s_soft_lower_list is not None:
                    assert len(s_soft_lower_list) == self.N, f"[MPC SOLVER] ASSERTION: s_soft_lower_list length {len(s_soft_lower_list)} != self.N {self.N}"
                if s_soft_upper_list is not None:
                    assert len(s_soft_upper_list) == self.N, f"[MPC SOLVER] ASSERTION: s_soft_upper_list length {len(s_soft_upper_list)} != self.N {self.N}"

        # Assert the type and shape of the input argument
        if should_check_inputs:
            def _check_vec_list(name, lst):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, v in enumerate(lst):
                    assert isinstance(v, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(v)}"
                    assert v.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ({self.n_s},1), got {v.shape}"
                    assert v.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {v.dtype} not in {self.allowed_dtypes}"
            _check_vec_list("s_soft_lower_list", s_soft_lower_list)
            _check_vec_list("s_soft_upper_list", s_soft_upper_list)

            if s_soft_lower_list is not None and s_soft_upper_list is not None:
                for k, (lo, hi) in enumerate(zip(s_soft_lower_list, s_soft_upper_list)):
                    assert np.all(lo <= hi), f"[MPC SOLVER] ASSERTION: s_soft_lower_list[{k}] must be elementwise <= s_soft_upper_list[{k}]"

        # Store the coefficients (auto-cast + defensive copies)
        self.sk_soft_lower = (
            [lo.astype(self.dtype, copy=False).copy() for lo in s_soft_lower_list]
            if s_soft_lower_list is not None else None
        )
        self.sk_soft_upper = (
            [hi.astype(self.dtype, copy=False).copy() for hi in s_soft_upper_list]
            if s_soft_upper_list is not None else None
        )

        # Update the flags
        self.is_time_invariant_state_soft_box_constraint_bounds = False if (s_soft_lower_list is not None or s_soft_upper_list is not None) else True
        self._rebuild_needed = True

        # Compute the derived details
        self._compute_derived_details_for_state_soft_constraint_bounds()



    def _compute_derived_details_for_state_soft_constraint_bounds(self):
        """
        Per time step derived details for soft state box constraints.

        Produces lists of length N:
        - n_sk_soft_lower[k] := number of finite lower bounds at step k
        - n_sk_soft_upper[k] := number of finite upper bounds at step k
        - sk_soft_lower_selector_matrix[k] : (n_sk_soft_lower[k], n_s) CSC selector
        - sk_soft_upper_selector_matrix[k] : (n_sk_soft_upper[k], n_s) CSC selector

        If a side is absent or has no finite entries at step k, the selector is an empty
        CSC matrix of shape (0, n_s) with dtype=self.dtype.
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")

        n_s = self.n_s

        # Ensure lists exist for iteration; treat missing side as all-infinite per-k
        lower_list = self.sk_soft_lower if self.sk_soft_lower is not None else [None] * self.N
        upper_list = self.sk_soft_upper if self.sk_soft_upper is not None else [None] * self.N

        self.n_sk_soft_lower = []
        self.n_sk_soft_upper = []
        self.sk_soft_lower_selector_matrix = []
        self.sk_soft_upper_selector_matrix = []

        for k in range(self.N):
            # Lower bound
            if isinstance(lower_list[k], np.ndarray):
                mask_lower = np.isfinite(lower_list[k][:, 0])
            else:
                mask_lower = np.zeros(n_s, dtype=bool)

            idx_lower = np.nonzero(mask_lower)[0]
            n_lower = int(idx_lower.size)
            self.n_sk_soft_lower.append(n_lower)
            if n_lower > 0:
                data = np.ones(n_lower, dtype=self.dtype)
                rows = np.arange(n_lower, dtype=np.int64)
                cols = idx_lower.astype(np.int64, copy=False)
                S_low = sparse.csc_matrix((data, (rows, cols)), shape=(n_lower, n_s), dtype=self.dtype)
            else:
                S_low = sparse.csc_matrix((0, n_s), dtype=self.dtype)
            self.sk_soft_lower_selector_matrix.append(S_low)

            # Upper bound
            if isinstance(upper_list[k], np.ndarray):
                mask_upper = np.isfinite(upper_list[k][:, 0])
            else:
                mask_upper = np.zeros(n_s, dtype=bool)

            idx_upper = np.nonzero(mask_upper)[0]
            n_upper = int(idx_upper.size)
            self.n_sk_soft_upper.append(n_upper)
            if n_upper > 0:
                data = np.ones(n_upper, dtype=self.dtype)
                rows = np.arange(n_upper, dtype=np.int64)
                cols = idx_upper.astype(np.int64, copy=False)
                S_up = sparse.csc_matrix((data, (rows, cols)), shape=(n_upper, n_s), dtype=self.dtype)
            else:
                S_up = sparse.csc_matrix((0, n_s), dtype=self.dtype)
            self.sk_soft_upper_selector_matrix.append(S_up)



    def set_state_soft_box_constraint_coefficients(
        self,
        s_soft_lin_coeff: np.ndarray | None = None,
        s_soft_quad_coeff: np.ndarray | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-invariant* objective function coefficients for the slack variables
        in the soft state constraint bound, broadcast over k = 0..N-1:
            lambda_k^T diag(sk_soft_quad_coeff) lambda_k + sk_soft_lin_coeff^T lambda_k
            
        Each of s_soft_lin_coeff / s_soft_quad_coeff may be None; missing side(s) treated as zero at build time.
        Expects column vectors (n_s, 1).

        Parameters
        ----------
            s_soft_lin_coeff : numpy array of size (n_s,1)
                The linear coefficient for soft lower bound constraint on the state
            s_soft_quad_coeff : numpy array of size (n_s,1)
                The quadratic coefficient for soft upper bound constraint on the state
        """
        # Assert the the horizon length is set
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Assert the type and shape of the input arguments
        if should_check_inputs:
            if s_soft_lin_coeff is not None:
                assert isinstance(s_soft_lin_coeff, np.ndarray), f"[MPC SOLVER] ASSERTION: s_soft_lin_coeff must be ndarray, got {type(s_soft_lin_coeff)}"
                assert s_soft_lin_coeff.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_soft_lin_coeff must be ({self.n_s},1), got {s_soft_lin_coeff.shape}"
                assert s_soft_lin_coeff.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_soft_lin_coeff dtype {s_soft_lin_coeff.dtype} not in {self.allowed_dtypes}"
            if s_soft_quad_coeff is not None:
                assert isinstance(s_soft_quad_coeff, np.ndarray), f"[MPC SOLVER] ASSERTION: s_soft_quad_coeff must be ndarray, got {type(s_soft_quad_coeff)}"
                assert s_soft_quad_coeff.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_soft_quad_coeff must be ({self.n_s},1), got {s_soft_quad_coeff.shape}"
                assert s_soft_quad_coeff.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_soft_quad_coeff dtype {s_soft_quad_coeff.dtype} not in {self.allowed_dtypes}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        s_soft_lin_coeff = s_soft_lin_coeff.astype(self.dtype, copy=False) if s_soft_lin_coeff is not None else None
        s_soft_quad_coeff = s_soft_quad_coeff.astype(self.dtype, copy=False) if s_soft_quad_coeff is not None else None

        # Use copies to avoid aliasing across time indices
        self.sk_soft_lin_coeff = [s_soft_lin_coeff.copy() for _ in range(self.N)] if s_soft_lin_coeff is not None else None
        self.sk_soft_quad_coeff = [s_soft_quad_coeff.copy() for _ in range(self.N)] if s_soft_quad_coeff is not None else None

        # Update the flags
        self.is_time_invariant_state_soft_box_constraint_coefficients = (s_soft_lin_coeff is not None or s_soft_quad_coeff is not None)
        self._rebuild_needed = True



    def set_time_varying_state_soft_box_constraint_coefficients(
        self,
        s_soft_lin_coeff_list: list[np.ndarray] | None = None,
        s_soft_quad_coeff_list: list[np.ndarray] | None = None,
        should_check_inputs: bool = True
    ):
        """
        Set *time-varying* objective function coefficients for the slack variables
        in the soft state constraint bound, broadcast over k = 0..N-1:
            lambda_k^T diag(sk_soft_quad_coeff) lambda_k + sk_soft_lin_coeff^T lambda_k

        Each of s_soft_lin_coeff / s_soft_quad_coeff may be None; missing side(s) treated as zero at build time.
        Lists, if provided, must have length N with elements shaped (n_s, 1).

        Parameters
        ----------
            s_soft_lin_coeff : list of (n_s, 1) ndarrays
                The linear coefficient for soft lower bound constraint on the state, per time step
            s_soft_quad_coeff : list of (n_s, 1) ndarrays
                The quadratic coefficient for soft upper bound constraint on the state, per time step
        """
        # Determine or validate horizon (depending on whether self.N is already set)
        if self.N is None:
            first = s_soft_lin_coeff_list if s_soft_lin_coeff_list is not None else s_soft_quad_coeff_list
            if first is None:
                raise ValueError("[MPC SOLVER] ERROR: Cannot infer self.N; provide s_soft_lin_coeff_list or s_soft_quad_coeff_list when self.N is None.")
            if should_check_inputs:
                assert isinstance(first, (list, tuple)), f"[MPC SOLVER] ASSERTION: coefficients list must be list/tuple, got {type(first)}"
                assert len(first) > 0, "[MPC SOLVER] ASSERTION: coefficients list must be non-empty"
            self._log("info", f"[MPC SOLVER] INFO: Horizon self.N is None; setting to {len(first)}.")
            self.N = len(first)
        else:
            if should_check_inputs:
                if s_soft_lin_coeff_list is not None:
                    assert len(s_soft_lin_coeff_list) == self.N, f"[MPC SOLVER] ASSERTION: s_soft_lin_coeff_list length {len(s_soft_lin_coeff_list)} != self.N {self.N}"
                if s_soft_quad_coeff_list is not None:
                    assert len(s_soft_quad_coeff_list) == self.N, f"[MPC SOLVER] ASSERTION: s_soft_quad_coeff_list length {len(s_soft_quad_coeff_list)} != self.N {self.N}"

        # Assert the type and shape of the input argument
        if should_check_inputs:
            def _check_vec_list(name, lst):
                if lst is None: return
                assert isinstance(lst, (list, tuple)), f"[MPC SOLVER] ASSERTION: {name} must be list/tuple, got {type(lst)}"
                for k, v in enumerate(lst):
                    assert isinstance(v, np.ndarray), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ndarray, got {type(v)}"
                    assert v.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: {name}[{k}] must be ({self.n_s},1), got {v.shape}"
                    assert v.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: {name}[{k}] dtype {v.dtype} not in {self.allowed_dtypes}"
            _check_vec_list("s_soft_lin_coeff_list", s_soft_lin_coeff_list)
            _check_vec_list("s_soft_quad_coeff_list", s_soft_quad_coeff_list)

        # Store the coefficients (auto-cast + defensive copies)
        self.sk_soft_lin_coeff = (
            [lo.astype(self.dtype, copy=False).copy() for lo in s_soft_lin_coeff_list]
            if s_soft_lin_coeff_list is not None else None
        )
        self.sk_soft_quad_coeff = (
            [hi.astype(self.dtype, copy=False).copy() for hi in s_soft_quad_coeff_list]
            if s_soft_quad_coeff_list is not None else None
        )

        # Update the flags
        self.is_time_invariant_state_soft_box_constraint_coefficients = False if (s_soft_lin_coeff_list is not None or s_soft_quad_coeff_list is not None) else True
        self._rebuild_needed = True



    def set_initial_condition(self, s_0: np.ndarray, should_check_inputs: bool = True):
        """
        Update the initial condition for the state
        
         Parameters
        ----------
            s_0 : numpy array of size (n_s,1)
                The initial condition for the state at the start of the prediction horizon
        """
        # Assert the type and shape of the input argument
        if should_check_inputs:
            assert isinstance(s_0, np.ndarray), f"[MPC SOLVER] ASSERTION: s_0 must be ndarray, got {type(s_0)}"
            assert s_0.shape == (self.n_s, 1), f"[MPC SOLVER] ASSERTION: s_0 must be ({self.n_s},1), got {s_0.shape}"
            assert s_0.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: s_0 dtype {s_0.dtype} not in {self.allowed_dtypes}"

        # Cast to the dtype of the class (no copy if dtype already matches)
        s_0 = s_0.astype(self.dtype, copy=False)

        # Update the class variable (defensive copy)
        self.s_0 = s_0.copy()

        # Update the s_0 flag:
        self._s_0_was_updated = True


    def set_previous_action(self, a_prev: np.ndarray, should_check_inputs: bool = True):
        """
        Update the previous action a_{"previous"} used in the action
        rate-of-change constraint at k = 0.

        Parameters
        ----------
            a_prev : numpy array of size (n_a,1)
                The action applied before the current horizon starts.
        """
        if should_check_inputs:
            assert isinstance(a_prev, np.ndarray), f"[MPC SOLVER] ASSERTION: a_prev must be ndarray, got {type(a_prev)}"
            assert a_prev.shape == (self.n_a, 1), f"[MPC SOLVER] ASSERTION: a_prev must be ({self.n_a},1), got {a_prev.shape}"
            assert a_prev.dtype in self.allowed_dtypes, f"[MPC SOLVER] ASSERTION: a_prev dtype {a_prev.dtype} not in {self.allowed_dtypes}"

        a_prev = a_prev.astype(self.dtype, copy=False)
        self.a_previous = a_prev.copy()
        self._a_previous_was_updated = True

    def set_previous_action_from_last_solve(self):
        """
        Convenience: set previous action from the last MPC solve's a0.

        If no prior solve exists, emits a warning (unless in silent_mode) and
        leaves the previous action unchanged.
        """
        if self.a0_from_previous_solve is None:
            self._log("warn", "[MPC SOLVER] WARNING: Cannot set previous action from last solve; no prior solve recorded.")
            return
        self.set_previous_action(self.a0_from_previous_solve, should_check_inputs=False)



    # def set_polygon_constraints(self, H_ineq_s=None, h_ineq_s=None, H_ineq_a=None, h_ineq_a=None, 
    #                              H_ineq_sa=None, h_ineq_sa=None):
    #     """
    #     Set the polygon constraints for states, actions, and state-action combined.

    #     The polygon constraints take the form:
    #         H_ineq_s*s  <= h_ineq_s
    #         H_ineq_s*a  <= h_ineq_a
    #         H_ineq_s*sa <= h_ineq_sa

    #     All of the input parameters of this function are optional. For any paremeter
    #     not provided (or provided as None), that constraint is not included during
    #     the building of the MPC optimization formulation.

    #     Parameters
    #     ----------
    #         H_ineq_s : numpy array of size ">=0" -by- n_s
    #             The matrix of the state-only polygon constraint
    #         h_ineq_s : numpy array of length H_ineq_s.shape[0] (i.e., 1-dimensional array)
    #             The vector of the state-only polygon constraint
    #         H_ineq_a : numpy array of size ">=0" -by- n_a
    #             The matrix of the action-only polygon constraint
    #         h_ineq_a : numpy array of length H_ineq_a.shape[0] (i.e., 1-dimensional array)
    #             The vector of the action-only polygon constraint
    #         H_ineq_sa : numpy array of size ">=0" -by- (n_s+n_a)
    #             The matrix of the state-action polygon constraint
    #         h_ineq_sa : numpy array of length H_ineq_sa.shape[0] (i.e., 1-dimensional array)
    #             The vector of the state-action polygon constraint
    #     """
    #     assert False, "ERROR in MPC class: polygon constraints are no yet implemented"
    #     # Process the state-only polygon constraints
    #     if H_ineq_s is not None:
    #         assert H_ineq_s.shape[1] == (self.n_a,), "ERROR in MPC class: H_ineq_s must be of size (...,n_s)"
    #         if h_ineq_s is not None:
    #             n_ineq_s = H_ineq_s.shape[0]
    #             assert h_ineq_s.size == n_ineq_s, "ERROR in MPC class: h_ineq_s must be of size (H_ineq_s.shape[0],)"
    #         else:
    #             print("WARNING in MPC class: H_ineq_s is is not None while h_ineq_s is None, hence setting H_ineq_s to be None.")
    #             # Set H_ineq_s to None
    #             H_ineq_s = None
    #             n_ineq_s = 0
    #     else:
    #         if h_ineq_s is not None:
    #             print("WARNING in MPC class: h_ineq_s is is not None while H_ineq_s is None, hence setting h_ineq_s to be None.")
    #             # Set H_ineq_s to None
    #             h_ineq_s = None
    #             n_ineq_s = 0
    #         else:
    #             n_ineq_s = 0
        
    #     # Process the action-only polygon constraints
    #     if H_ineq_a is not None:
    #         assert H_ineq_a.shape[1] == (self.n_a,), "ERROR in MPC class: H_ineq_a must be of size (...,n_a)"
    #         if h_ineq_a is not None:
    #             n_ineq_a = H_ineq_a.shape[0]
    #             assert h_ineq_a.size == n_ineq_a, "ERROR in MPC class: h_ineq_a must be of size (H_ineq_a.shape[0],)"
    #         else:
    #             print("WARNING in MPC class: H_ineq_a is is not None while h_ineq_a is None, hence setting H_ineq_a to be None.")
    #             # Set H_ineq_a to None
    #             H_ineq_a = None
    #             n_ineq_a = 0
    #     else:
    #         if h_ineq_a is not None:
    #             print("WARNING in MPC class: h_ineq_a is is not None while H_ineq_a is None, hence setting h_ineq_a to be None.")
    #             # Set H_ineq_a to None
    #             h_ineq_a = None
    #             n_ineq_a = 0
    #         else:
    #             n_ineq_a = 0

    #     # Process the state-only polygon constraints
    #     if H_ineq_sa is not None:
    #         assert H_ineq_sa.shape[1] == (self.n_a,), "ERROR in MPC class: H_ineq_sa must be of size (...,n_sa)"
    #         if h_ineq_sa is not None:
    #             n_ineq_sa = H_ineq_sa.shape[0]
    #             assert h_ineq_sa.size == n_ineq_sa, "ERROR in MPC class: h_ineq_sa must be of size (H_ineq_sa.shape[0],)"
    #         else:
    #             print("WARNING in MPC class: H_ineq_sa is is not None while h_ineq_sa is None, hence setting H_ineq_sa to be None.")
    #             # Set H_ineq_sa to None
    #             H_ineq_sa = None
    #             n_ineq_sa = 0
    #     else:
    #         if h_ineq_sa is not None:
    #             print("WARNING in MPC class: h_ineq_sa is is not None while H_ineq_sa is None, hence setting h_ineq_sa to be None.")
    #             # Set H_ineq_sa to None
    #             h_ineq_sa = None
    #             n_ineq_sa = 0
    #         else:
    #             n_ineq_sa = 0

    #     # Set the class variables
    #     # > For state-only
    #     self.H_ineq_s = H_ineq_s
    #     self.h_ineq_s = h_ineq_s
    #     self.n_ineq_s = n_ineq_s
    #     # > For action-only
    #     self.H_ineq_a = H_ineq_a
    #     self.h_ineq_a = h_ineq_a
    #     self.n_ineq_a = n_ineq_a
    #     # > For state-action
    #     self.H_ineq_sa = H_ineq_sa
    #     self.h_ineq_sa = h_ineq_sa
    #     self.n_ineq_sa = n_ineq_sa

    #     # Update the rebuild flag:
    #     self._rebuild_needed = True



    def _build_all_constraints(self, should_check_inputs: bool = True):
        """
        Construct the linear equality constraints that encode:
        - The state evolution
        - The action constraints
        - Any soft constraints on the state

        Note that we need to build everything together in one function in
        order of maintain that the "A" matrix of OSQP is block diagonal.

        As a reminder, the optimization variable ordering used for this build:
            [ s_0, a_0, s_1, a_1, …, s_{N-1}, a_{N-1}, s_N ]
        
        If there are soft constraint on the state, then the slack variables are
        inserted next to the respective state to ensure the build of "P" and "A"
        (in the OSQP format) remain block diagonal.
            [ s_0, a_0,
              s_1, lambda_1, mu_1, a_1,
              s_2, lambda_2, mu_2, a_2,
              …,
              s_{N-1}, lambda_{N-1}, mu_{N-1}, a_{N-1},
              s_N, lambda_N, mu_N
            ]

        The state evoluation constraints are of the form:
            s_{k+1} = A_k*s_k + B_k*a_k + g_k
        And hence in the format of the optimization variable, this becomes:
            [I ,0, ...] [s_0; ...] = "initial condition"
            [A, B, -I] [s_0; a_0; s_1] = -g_0
            [A, 0, B, -I] [s_k; lambda_k; mu_k; a_k; s_{k+1}] = -g_k    for k = 1,...,N-1
        
        The action box constraints are of the form:
            ak_lower <= a_k <= ak_upper
        And hence in the format of the optimization variable, this becomes:
            a0_lower <= [0, I] [s_0; a_0] <= a0_upper
            ak_lower <= [0, 0, 0, I] [s_k; lambda_k; mu_k; a_k] <= ak_upper    for k = 1,...,N-1

        The action rate-of-change constraints are of the form:
            ak_roc_lower <= a_k - a_{k-1} <= ak_roc_upper   for k=1..N-1
        And hence in the format of the optimization variable, this becomes:
            a0_roc_lower + a_{"previous"} <= [0, I] [s_0; a_0] <= a0_roc_upper + a_{"previous"}
            ak_roc_lower <= [..., -I, 0, 0, 0, I] [...; a_k-1; s_k; lambda_k; mu_k; a_k] <= ak_roc_upper    for k = 1,...,N-1
        Important to note that the identity matrices here are actually the
        "ak_roc_constraint_selector_matrix" to allow that only certain elements
        of the action vector have a rate-of-change constraint.

        The state box constraints are of the form:
            sk_lower <= s_k <= sk_upper
        And hence in the format of the optimization variable, this becomes:
            sk_lower <= [I, 0, 0, 0] [s_k; lambda_k; mu_k; a_k] <= sk_upper    for k = 1,...,N
        Important to note that the identity matrices here are actually the
        "sk_box_constraint_selector_matrix" to allow that only certain elements
        of the state vector have a rate-of-change constraint.

        The soft state box constraints are of the form:
            s_soft_lower <= s_k + lambda_k
                            s_k - mu_k      <= s_soft_upper
                       0 <= lambda_k
                       0 <= mu_k
        And hence in the format of the optimization variable, this becomes:
            sk_soft_lower <= [I, I,  0, 0] [s_k; lambda_k; mu_k; a_k] <= inf              for k = 1,...,N
            -inf          <= [I, 0, -I, 0] [s_k; lambda_k; mu_k; a_k] <= sk_soft_upper    for k = 1,...,N
            0             <= [0, I,  0, 0] [s_k; lambda_k; mu_k; a_k] <= inf              for k = 1,...,N
            0             <= [0, 0,  I, 0] [s_k; lambda_k; mu_k; a_k] <= inf              for k = 1,...,N
        Important to note that the identity matrices here are actually the
        "sk_soft_lower_selector_matrix" and "sk_soft_upper_selector_matrix"
        to allow that only certain elements of the state vector have a soft
        constraint.
        
        Putting this all together in a block row ordering that keeps "A" block diagonal
            >> Initial state constraint
                             [I ,0, ...] [s_0; a_0, ...]          = "initial condition"
            >> Initial action rate-of-change constaint
            a0_roc_lower + a_{"previous"}  <= [0, I, ...] [s_0; a_0, ...]  <= a0_roc_upper + a_{"previous"}
            >> Initial action box constraint
            a0_lower      <= [0, I, ...] [s_0; a_0, ...]          <= a0_upper
            >> Initial model evolution
                             [A, B, -I, ...] [s_0; a_0; s_1, ...] = -g_0
            >> Remaining for for k = 1,...,N
            sk_lower      <= [...,  0 , I, 0,  0, 0,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= sk_upper
            sk_soft_lower <= [...,  0, I, I,  0, 0,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= inf
            -inf          <= [...,  0, I, 0, -I, 0,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= sk_soft_upper
            0             <= [...,  0, 0, I,  0, 0,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= inf
            0             <= [...,  0, 0, 0,  I, 0,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= inf
            ak_lower      <= [...,  0, 0, 0,  0, I,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= ak_upper
            ak_roc_lower  <= [...,  0, 0, 0,  0, I,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] <= ak_roc_upper
                             [..., -I, 0, 0,  0, I,  0, ...] [..., a_k-1; s_k; lambda_k; mu_k; a_k, s_k+1, ...] = -g_0
            >> The final soft constraint
            sk_soft_lower <= [..., I, I,  0] [..., s_N; lambda_N; mu_N] <= inf
            -inf          <= [..., I, 0, -I] [..., s_N; lambda_N; mu_N] <= sk_soft_upper
            0             <= [..., 0, I,  0] [..., s_N; lambda_N; mu_N] <= inf
            0             <= [..., 0, 0,  I] [..., s_N; lambda_N; mu_N] <= inf

        Important to note that the following constraints only have a row for each element of
        the respective vector (i.e., state or action vector) where the constrain is finite:
        - Action rate-of-change constraints
        - State box constraints
        - State soft constraints

        Summary Notes
        -------------
        - lambda_k size = number of finite LOWER bounds for s_k (using the per-k selectors)
        - mu_k size     = number of finite UPPER bounds for s_k (using the per-k selectors)
        - lambda_k, mu_k exist only when counts > 0 at that k (columns omitted otherwise)
        - Soft box constaints are applied for k = 1..N (hence i = k-1 in the lists sk_soft_lower and similar)
        - Soft state bounds are applied for k = 1..N (hence i = k-1 in the lists sk_soft_lower and similar)
        - Action box constraints are applied for k = 0..N-1
        - Action rate-of-change constraint are applied:
            - As "normal" for k=1..N-1
            - For k=0, needs the previous action as an input parameter to this optimization, i.e., a_{"previous"}
        - State evolution equalities:
            s_{k+1} - A_k s_k - B_k a_k = g_k   (hence l = u = g_k)
        Initial condition equality:
            s_0 = s0_meas                       (hence l = u = s_0)
        
        Returns
        -------
        col_layout : dict[(str,int) -> slice]
            Maps ('s',k), ('a',k), ('lam',k), ('mu',k) to column slices in the stacked variable.
            Note: ('lam',k) / ('mu',k) are present only if those slacks exist at step k.
        """
        
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")
        N, n_s, n_a, dt = self.N, self.n_s, self.n_a, self.dtype

        # Ensure derived details exist
        if self.Ak_sparse is None or self.Bk_sparse is None:
            self._compute_derived_details_for_model(should_check_inputs=should_check_inputs)
        if (self.sk_soft_lower_selector_matrix is None) or (self.sk_soft_upper_selector_matrix is None):
            self._compute_derived_details_for_state_soft_constraint_bounds()
        # Ensure state hard box and action RoC derived details exist
        if self.sk_box_constraint_selector_matrix is None:
            self._compute_derived_details_for_state_box_constraints()
        if self.ak_roc_constraint_selector_matrix is None:
            self._compute_derived_details_for_action_rate_of_change_constraints()

        if should_check_inputs:
            assert len(self.Ak_sparse) == N and len(self.Bk_sparse) == N, \
                "[MPC SOLVER] ASSERTION: Ak_sparse/Bk_sparse must have length N"
            assert len(self.sk_soft_lower_selector_matrix) == N, \
                "[MPC SOLVER] ASSERTION: sk_soft_lower_selector_matrix must have length N"
            assert len(self.sk_soft_upper_selector_matrix) == N, \
                "[MPC SOLVER] ASSERTION: sk_soft_upper_selector_matrix must have length N"

        # ---------- Column schema ----------
        col_blocks: list[tuple[str,int,int]] = []
        col_blocks.append(("s", 0, n_s))
        for i in range(N):
            col_blocks.append(("a", i,   n_a))
            col_blocks.append(("s", i+1, n_s))
            nL = self.n_sk_soft_lower[i] if self.n_sk_soft_lower else 0
            nU = self.n_sk_soft_upper[i] if self.n_sk_soft_upper else 0
            if nL > 0: col_blocks.append(("lam", i+1, nL))
            if nU > 0: col_blocks.append(("mu",  i+1, nU))

        col_layout: dict[tuple[str,int], slice] = {}
        off = 0
        for name, k, w in col_blocks:
            col_layout[(name, k)] = slice(off, off + w)
            off += w
        n_block_cols = len(col_blocks)

        def _col_index(kind: str, k: int) -> int:
            for j, (nm, kk, _) in enumerate(col_blocks):
                if nm == kind and kk == k:
                    return j
            return -1

        # ---------- Build rows ----------
        rows: list[list[sparse.spmatrix | None]] = []
        l_list: list[np.ndarray] = []
        u_list: list[np.ndarray] = []

        I_s = sparse.eye(n_s, format="csc", dtype=dt)
        I_a = sparse.eye(n_a, format="csc", dtype=dt)

        def _bounds_vec_action(k, is_lo: bool):
            arr_list = self.ak_lower if is_lo else self.ak_upper
            if arr_list is None:
                val = -np.inf if is_lo else np.inf
                return np.full((n_a, 1), val, dtype=dt)
            return arr_list[k].astype(dt, copy=False)

        # --- Initial group ---
        # (1) s0 equality
        row = [None] * n_block_cols
        row[_col_index("s", 0)] = I_s
        rows.append(row)
        row_labels: list[str] = []
        row_labels.extend(["init state"] * n_s)
        s0_rhs = np.zeros((n_s, 1), dtype=dt)
        if self.s_0 is not None:
            s0_rhs[:] = self.s_0
        l_list.append(s0_rhs); u_list.append(s0_rhs)

        # (2) a0 rate-of-change using a_previous (if any rows)
        self._row_slice_a0roc = slice(0, 0)  # default empty
        if self.ak_roc_constraint_selector_matrix is not None and len(self.ak_roc_constraint_selector_matrix) > 0:
            S_roc0 = self.ak_roc_constraint_selector_matrix[0]
            nR0 = int(S_roc0.shape[0])
            if nR0 > 0:
                row = [None] * n_block_cols
                row[_col_index("a", 0)] = S_roc0
                rows.append(row)
                row_labels.extend(["action roc k=0"] * nR0)
                # Base bounds selected
                if self.ak_roc_lower is not None and self.ak_roc_lower[0] is not None:
                    lo0_sel = (S_roc0 @ self.ak_roc_lower[0]).astype(dt, copy=False)
                else:
                    lo0_sel = np.full((nR0, 1), -np.inf, dtype=dt)
                if self.ak_roc_upper is not None and self.ak_roc_upper[0] is not None:
                    up0_sel = (S_roc0 @ self.ak_roc_upper[0]).astype(dt, copy=False)
                else:
                    up0_sel = np.full((nR0, 1),  np.inf, dtype=dt)
                # Shift by S * a_previous
                a_prev = self.a_previous
                a_prev = np.zeros((n_a,1), dtype=dt) if a_prev is None else a_prev.astype(dt, copy=False)
                shift0 = S_roc0 @ a_prev
                l_list.append(lo0_sel + shift0)
                u_list.append(up0_sel + shift0)
                # Compute slice for these rows
                n_rows_before = sum(vec.shape[0] for vec in l_list[:-1])
                self._row_slice_a0roc = slice(n_rows_before, n_rows_before + nR0)

        # (3) a0 box
        row = [None] * n_block_cols
        row[_col_index("a", 0)] = I_a
        rows.append(row)
        row_labels.extend(["init action"] * n_a)
        l_list.append(_bounds_vec_action(0, True))
        u_list.append(_bounds_vec_action(0, False))

        # (4) dyn k=0: s1 - A0 s0 - B0 a0 = g0
        row = [None] * n_block_cols
        row[_col_index("s", 0)]   = -self.Ak_sparse[0]
        row[_col_index("a", 0)]   = -self.Bk_sparse[0]
        row[_col_index("s", 1)]   =  I_s
        rows.append(row)
        row_labels.extend([f"model[{i}] k=0" for i in range(n_s)])
        g0 = self.gk[0] if (self.gk is not None and self.gk[0] is not None) else np.zeros((n_s,1), dtype=dt)
        l_list.append(g0); u_list.append(g0)

        # --- Per-step groups for k = 1..N-1 ---
        for k in range(1, N):
            i = k - 1  # selector index for s_k

            # State box constraints at s_k (i.e. hard bound constraints)
            if self.sk_box_constraint_selector_matrix is not None:
                S_box = self.sk_box_constraint_selector_matrix[i]
                nB = int(S_box.shape[0])
                if nB > 0:
                    row = [None] * n_block_cols
                    row[_col_index("s", k)] = S_box
                    rows.append(row)
                    if self.sk_lower is not None and self.sk_lower[i] is not None:
                        lo_box_sel = (S_box @ self.sk_lower[i]).astype(dt, copy=False)
                    else:
                        lo_box_sel = np.full((nB, 1), -np.inf, dtype=dt)
                    if self.sk_upper is not None and self.sk_upper[i] is not None:
                        up_box_sel = (S_box @ self.sk_upper[i]).astype(dt, copy=False)
                    else:
                        up_box_sel = np.full((nB, 1),  np.inf, dtype=dt)
                    l_list.append(lo_box_sel)
                    u_list.append(up_box_sel)

            # State soft constraints LOWER: S_low s_k + I λ_k >= s_low_sel_k
            S_low = self.sk_soft_lower_selector_matrix[i]; nL = S_low.shape[0]
            if nL > 0:
                row = [None] * n_block_cols
                row[_col_index("s", k)] = S_low
                j_l = _col_index("lam", k)
                if j_l >= 0: row[j_l] = sparse.eye(nL, format="csc", dtype=dt)
                rows.append(row)

                s_low_k = self.sk_soft_lower[i] if (self.sk_soft_lower is not None and self.sk_soft_lower[i] is not None) \
                        else np.full((n_s,1), -np.inf, dtype=dt)
                lo_sel = S_low @ s_low_k
                l_list.append(lo_sel.astype(dt, copy=False))
                u_list.append(np.full((nL,1), np.inf, dtype=dt))

                # λ_k ≥ 0
                if j_l >= 0:
                    row = [None] * n_block_cols
                    row[j_l] = sparse.eye(nL, format="csc", dtype=dt)
                    rows.append(row)
                    l_list.append(np.zeros((nL,1), dtype=dt))
                    u_list.append(np.full((nL,1), np.inf, dtype=dt))

            # State soft constraints UPPER: S_up s_k - I μ_k <= s_up_sel_k
            S_up = self.sk_soft_upper_selector_matrix[i]; nU = S_up.shape[0]
            if nU > 0:
                row = [None] * n_block_cols
                row[_col_index("s", k)] = S_up
                j_m = _col_index("mu", k)
                if j_m >= 0: row[j_m] = -sparse.eye(nU, format="csc", dtype=dt)
                rows.append(row)

                s_up_k = self.sk_soft_upper[i] if (self.sk_soft_upper is not None and self.sk_soft_upper[i] is not None) \
                        else np.full((n_s,1), np.inf, dtype=dt)
                up_sel = S_up @ s_up_k
                l_list.append(np.full((nU,1), -np.inf, dtype=dt))
                u_list.append(up_sel.astype(dt, copy=False))

                # μ_k ≥ 0
                if j_m >= 0:
                    row = [None] * n_block_cols
                    row[j_m] = sparse.eye(nU, format="csc", dtype=dt)
                    rows.append(row)
                    l_list.append(np.zeros((nU,1), dtype=dt))
                    u_list.append(np.full((nU,1), np.inf, dtype=dt))

            # Action box constraints a_k (i.e. hard bound constraints)
            row = [None] * n_block_cols
            row[_col_index("a", k)] = I_a
            rows.append(row)
            l_list.append(_bounds_vec_action(k, True))
            u_list.append(_bounds_vec_action(k, False))

            # Action rate-of-change constraints a_k - a_{k-1}
            if self.ak_roc_constraint_selector_matrix is not None:
                S_rok = self.ak_roc_constraint_selector_matrix[k]
                nRk = int(S_rok.shape[0])
                if nRk > 0:
                    row = [None] * n_block_cols
                    row[_col_index("a", k-1)] = -S_rok
                    row[_col_index("a", k)]   =  S_rok
                    rows.append(row)
                    if self.ak_roc_lower is not None and self.ak_roc_lower[k] is not None:
                        lo_sel_k = (S_rok @ self.ak_roc_lower[k]).astype(dt, copy=False)
                    else:
                        lo_sel_k = np.full((nRk, 1), -np.inf, dtype=dt)
                    if self.ak_roc_upper is not None and self.ak_roc_upper[k] is not None:
                        up_sel_k = (S_rok @ self.ak_roc_upper[k]).astype(dt, copy=False)
                    else:
                        up_sel_k = np.full((nRk, 1),  np.inf, dtype=dt)
                    l_list.append(lo_sel_k)
                    u_list.append(up_sel_k)

            # Dynamics k: s_{k+1} - A_k s_k - B_k a_k = g_k
            row = [None] * n_block_cols
            row[_col_index("s", k)]   = -self.Ak_sparse[k]
            row[_col_index("a", k)]   = -self.Bk_sparse[k]
            row[_col_index("s", k+1)] =  I_s
            rows.append(row)
            gk = self.gk[k] if (self.gk is not None and self.gk[k] is not None) else np.zeros((n_s,1), dtype=dt)
            l_list.append(gk); u_list.append(gk)

        # Terminal state box constraints at s_N
        i = N - 1
        if self.sk_box_constraint_selector_matrix is not None:
            S_boxN = self.sk_box_constraint_selector_matrix[i]
            nBN = int(S_boxN.shape[0])
            if nBN > 0:
                row = [None] * n_block_cols
                row[_col_index("s", N)] = S_boxN
                rows.append(row)
                if self.sk_lower is not None and self.sk_lower[i] is not None:
                    lo_box_selN = (S_boxN @ self.sk_lower[i]).astype(dt, copy=False)
                else:
                    lo_box_selN = np.full((nBN, 1), -np.inf, dtype=dt)
                if self.sk_upper is not None and self.sk_upper[i] is not None:
                    up_box_selN = (S_boxN @ self.sk_upper[i]).astype(dt, copy=False)
                else:
                    up_box_selN = np.full((nBN, 1),  np.inf, dtype=dt)
                l_list.append(lo_box_selN)
                u_list.append(up_box_selN)

        # Terminal state soft constraints on s_N (i = N-1)
        # >> Soft LOWER at s_N
        S_low = self.sk_soft_lower_selector_matrix[i]; nL = S_low.shape[0]
        if nL > 0:
            row = [None] * n_block_cols
            row[_col_index("s", N)] = S_low
            j_lN = _col_index("lam", N)
            if j_lN >= 0: row[j_lN] = sparse.eye(nL, format="csc", dtype=dt)
            rows.append(row)

            s_low_N = self.sk_soft_lower[i] if (self.sk_soft_lower is not None and self.sk_soft_lower[i] is not None) \
                    else np.full((n_s,1), -np.inf, dtype=dt)
            lo_selN = S_low @ s_low_N
            l_list.append(lo_selN.astype(dt, copy=False))
            u_list.append(np.full((nL,1), np.inf, dtype=dt))

            # λ_N ≥ 0
            if j_lN >= 0:
                row = [None] * n_block_cols
                row[j_lN] = sparse.eye(nL, format="csc", dtype=dt)
                rows.append(row)
                l_list.append(np.zeros((nL,1), dtype=dt))
                u_list.append(np.full((nL,1), np.inf, dtype=dt))

        # >> Soft UPPER at s_N
        S_up = self.sk_soft_upper_selector_matrix[i]; nU = S_up.shape[0]
        if nU > 0:
            row = [None] * n_block_cols
            row[_col_index("s", N)] = S_up
            j_mN = _col_index("mu", N)
            if j_mN >= 0: row[j_mN] = -sparse.eye(nU, format="csc", dtype=dt)
            rows.append(row)

            s_up_N = self.sk_soft_upper[i] if (self.sk_soft_upper is not None and self.sk_soft_upper[i] is not None) \
                    else np.full((n_s,1), np.inf, dtype=dt)
            up_selN = S_up @ s_up_N
            l_list.append(np.full((nU,1), -np.inf, dtype=dt))
            u_list.append(up_selN.astype(dt, copy=False))

            # μ_N ≥ 0
            if j_mN >= 0:
                row = [None] * n_block_cols
                row[j_mN] = sparse.eye(nU, format="csc", dtype=dt)
                rows.append(row)
                l_list.append(np.zeros((nU,1), dtype=dt))
                u_list.append(np.full((nU,1), np.inf, dtype=dt))

        # Assemble
        A = sparse.bmat(rows, format="csc", dtype=dt)
        l = np.vstack(l_list).astype(dt, copy=False)
        u = np.vstack(u_list).astype(dt, copy=False)

        # Store
        self._A_for_constraints = A
        self._l_for_constraints = l
        self._u_for_constraints = u
        self._col_layout = col_layout

        return col_layout




    def _build_objective(self, col_layout: dict | None = None, should_check_inputs: bool = True):
        """
        Construct the quadratic and linear terms of the objective function,
        to match the OSQP formulation of:
            (1/2) x^T P x + q^T x

        The objective function over the time step takes the form:
            sum_{k=0}^{N-1} ( (sk-sk_ref)^T Qk (sk-sk_ref) + (ak-ak_ref)^T Rk (ak-ak_ref) + qk^T sk + rk^T ak )

        The inside of the sum is a per-time step function, hence we expand it out as:
              sk^T Qk sk - 2 sk_ref^T Qk sk + sk_ref^T Qk sk_ref
            + ak^T Rk ak - 2 ak_ref^T Rk ak + ak_ref^T Rk ak_ref
            + qk^T sk + rk^T ak

        Collecting together the quadratic, linear, and constant terms:
            Quadratic in sk:    + sk^T Qk sk
            Linear in sk:       + (-2 sk_ref^T Qk + qk^T) sk
            Quadratic in ak:    + ak^T Rk ak
            Linear in ak:       + (-2 ak_ref^T Rk + rk^T) ak
            Constant:           + sk_ref^T Qk sk_ref + ak_ref^T Rk ak_ref

        The terminal objective takes the form:
            (sN-sN_ref)^T QN (s-sN_ref) + qN^T sN

        Expanding the terminal objective and collectiong together the
        quadratic, linear, and constant terms:
            Quadratic in sN:    + sN^T QN sN
            Linear in sN:       + (-2 sN_ref^T QN + qN^T) sN
            Constant:            + sN_ref^T QN sN_ref
        
        The soft constraints objective terms take the form:
            lambda_k^T diag(sk_soft_quad_coeff) lambda_k + sk_soft_lin_coeff^T lambda_k
            + mu_k^T diag(sk_soft_quad_coeff) mu_k + sk_soft_lin_coeff^T mu_k
        For k=1..N
        
        As a reminder, the optimization variable ordering used for this build:
            [ s_0, a_0, s_1, a_1, …, s_{N-1}, a_{N-1}, s_N ]
        
        If there are soft constraint on the state, then the slack variables are
        inserted next to the respective state to ensure the build of "P" and "A"
        (in the OSQP format) remain block diagonal.
            [ s_0, a_0, s_1, lambda_1, mu_1, a_1, …, s_{N-1}, lambda_{N-1}, mu_{N-1}, a_{N-1}, s_N, lambda_N, mu_N]

        Uses
        ----
            - Stage costs:  Qk, Rk, qk, rk (lists length N or None)
            - Terminal:     QN, qN (or None)
            - References:   sk_ref (list len N or None), ak_ref (list len N or None), sN_ref (or None)
            - Soft slack costs (per-k): sk_soft_quad_coeff, sk_soft_lin_coeff (lists len N or None)
                applied only for active lower/upper selector rows at each k (λ_k / μ_k)
        
        Produces
        --------
            self._P_for_objective    : csc_matrix  (block-diagonal)
            self._q_for_objective    : ndarray     ((n_vars, 1))
            self._objective_constant : scalar      (sum of terms independent of x)
        """
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon self.N is not set.")
        N, n_s, n_a, dt = self.N, self.n_s, self.n_a, self.dtype

        # Ensure precomputed CSC for objective matrices
        if self.Qk_sparse is None or self.Rk_sparse is None or (self.QN is not None and self.QN_sparse is None):
            self._compute_derived_details_for_objective_mats(should_check_inputs=should_check_inputs)

        # Ensure per-k selector details exist for soft slacks (even if empty)
        if (self.sk_soft_lower_selector_matrix is None) or (self.sk_soft_upper_selector_matrix is None):
            self._compute_derived_details_for_state_soft_constraint_bounds()

        # Column layout: use provided, else reuse one built earlier
        if col_layout is None:
            if not hasattr(self, "_col_layout") or self._col_layout is None:
                # Recreate the exact layout logic used in _build_all_constraints:
                col_blocks = [("s", 0, n_s)]
                for i in range(N):
                    col_blocks += [("a", i, n_a), ("s", i+1, n_s)]
                    nL = self.n_sk_soft_lower[i] if self.n_sk_soft_lower else 0
                    nU = self.n_sk_soft_upper[i] if self.n_sk_soft_upper else 0
                    if nL > 0: col_blocks.append(("lam", i+1, nL))
                    if nU > 0: col_blocks.append(("mu",  i+1, nU))
                # Build slices
                layout = {}
                off = 0
                for name, k, w in col_blocks:
                    layout[(name, k)] = slice(off, off + w)
                    off += w
                self._col_layout = layout
            col_layout = self._col_layout

        # Convenience accessor for slices (empty slice if absent)
        def sl(kind: str, k: int) -> slice:
            return col_layout.get((kind, k), slice(0, 0))

        # ---- Initialize diagonal blocks and linear vector
        n_vars = max((s.stop for s in col_layout.values()), default=0)
        q = np.zeros((n_vars, 1), dtype=dt)

        # Build diag blocks EXACTLY in the actual column order
        # (sort by slice start, which encodes the variable position in x)
        ordered_items = sorted(col_layout.items(), key=lambda kv: kv[1].start)
        diag_blocks = []
        key_to_blockidx = {}
        for idx, (key, slc) in enumerate(ordered_items):
            w = slc.stop - slc.start
            key_to_blockidx[key] = idx
            diag_blocks.append(sparse.csc_matrix((w, w), dtype=dt))

        def add_block(kind: str, k: int, M: sparse.spmatrix | None):
            if M is None: 
                return
            key = (kind, k)
            i = key_to_blockidx.get(key, None)
            if i is None:  # block absent (e.g., no slacks at that k)
                return
            diag_blocks[i] = diag_blocks[i] + M


        # ---- Stage terms for k = 0..N-1
        objective_const = 0.0
        for k in range(N):
            # Qk, Rk as sparse (None -> zeros)
            #Qk_sp = sparse.csc_matrix(self.Qk[k], shape=(n_s, n_s), dtype=dt) if self.Qk is not None and self.Qk[k] is not None else None
            #Rk_sp = sparse.csc_matrix(self.Rk[k], shape=(n_a, n_a), dtype=dt) if self.Rk is not None and self.Rk[k] is not None else None
            Qk_sp = self.Qk_sparse[k] if self.Qk_sparse is not None else None
            Rk_sp = self.Rk_sparse[k] if self.Rk_sparse is not None else None

            # References, linear terms (None -> zeros)
            sref_k = self.sk_ref[k] if (self.sk_ref is not None and self.sk_ref[k] is not None) else np.zeros((n_s,1), dtype=dt)
            aref_k = self.ak_ref[k] if (self.ak_ref is not None and self.ak_ref[k] is not None) else np.zeros((n_a,1), dtype=dt)
            qk     = self.qk[k]     if (self.qk is not None     and self.qk[k]     is not None) else np.zeros((n_s,1), dtype=dt)
            rk     = self.rk[k]     if (self.rk is not None     and self.rk[k]     is not None) else np.zeros((n_a,1), dtype=dt)

            # Quadratic blocks (factor 2 for OSQP)
            if Qk_sp is not None: add_block("s", k, 2.0 * Qk_sp)
            if Rk_sp is not None: add_block("a", k, 2.0 * Rk_sp)

            # Linear pieces: (-2 Qk sref_k + qk) and (-2 Rk aref_k + rk)
            if Qk_sp is not None:
                q[ sl("s", k) ] += (-2.0 * (Qk_sp @ sref_k) + qk)
                objective_const += float((sref_k.T @ (Qk_sp @ sref_k)).ravel())
            else:
                q[ sl("s", k) ] += qk
            if Rk_sp is not None:
                q[ sl("a", k) ] += (-2.0 * (Rk_sp @ aref_k) + rk)
                objective_const += float((aref_k.T @ (Rk_sp @ aref_k)).ravel())
            else:
                q[ sl("a", k) ] += rk

        # ---- Terminal (s_N)
        #QN_sp = sparse.csc_matrix(self.QN, shape=(n_s, n_s), dtype=dt) if self.QN is not None else None
        QN_sp = self.QN_sparse if self.QN_sparse is not None else None

        qN    = self.qN    if self.qN is not None else np.zeros((n_s,1), dtype=dt)
        sNref = self.sN_ref if self.sN_ref is not None else np.zeros((n_s,1), dtype=dt)

        if QN_sp is not None: add_block("s", N, 2.0 * QN_sp)
        if QN_sp is not None:
            q[ sl("s", N) ] += (-2.0 * (QN_sp @ sNref) + qN)
            objective_const += float((sNref.T @ (QN_sp @ sNref)).ravel())
        else:
            q[ sl("s", N) ] += qN

        # ---- Soft slack costs per-k on λ_{k} and μ_{k} (k = 1..N)
        # Coeff vectors per step (None -> zeros)
        for i in range(N):         # i corresponds to state s_{i+1}
            # Project coefficients to active indices using selectors
            S_low = self.sk_soft_lower_selector_matrix[i]   # (nL, n_s), maybe (0,n_s)
            S_up  = self.sk_soft_upper_selector_matrix[i]   # (nU, n_s), maybe (0,n_s)

            nL, nU = S_low.shape[0], S_up.shape[0]

            # Selected quadratic weights (vectors) and linear weights
            if self.sk_soft_quad_coeff is not None and self.sk_soft_quad_coeff[i] is not None:
                wL = (S_low @ self.sk_soft_quad_coeff[i]) if nL > 0 else None  # (nL,1)
                wU = (S_up  @ self.sk_soft_quad_coeff[i]) if nU > 0 else None
            else:
                wL = None; wU = None

            if self.sk_soft_lin_coeff is not None and self.sk_soft_lin_coeff[i] is not None:
                vL = (S_low @ self.sk_soft_lin_coeff[i]) if nL > 0 else None   # (nL,1)
                vU = (S_up  @ self.sk_soft_lin_coeff[i]) if nU > 0 else None
            else:
                vL = None; vU = None

            # λ_{i+1} block
            if nL > 0 and ('lam', i+1) in col_layout:
                lam_slice = sl('lam', i+1)
                if wL is not None:
                    # Quadratic diag: 2 * diag(wL)
                    add_block('lam', i+1, sparse.diags((2.0 * wL.ravel()), format='csc', dtype=dt))
                if vL is not None:
                    q[lam_slice] += vL

            # μ_{i+1} block
            if nU > 0 and ('mu', i+1) in col_layout:
                mu_slice = sl('mu', i+1)
                if wU is not None:
                    add_block('mu', i+1, sparse.diags((2.0 * wU.ravel()), format='csc', dtype=dt))
                if vU is not None:
                    q[mu_slice] += vU

        # ---- Assemble P (block diagonal) in the *same order* as layout
        P = sparse.block_diag(diag_blocks, format="csc", dtype=dt)

        # Store
        self._P_for_objective = P
        self._q_for_objective = q
        self._objective_constant = float(objective_const)



    def var_slice(
        self,
        kind: str,
        k: int,
        *,
        require_layout: bool = True,
        with_width: bool = False,
    ):
        """
        Return the column slice (and optionally its width) for a block in the stacked OSQP
        decision vector built by `_build_all_constraints`.

        Blocks are identified by:
            kind ∈ {"s", "a", "lam", "mu"} and stage index k:
            - ("s",  k): state s_k
            - ("a",  k): action a_k
            - ("lam", k): lower-bound slack λ_k  (exists only if there are active soft lower bounds at k)
            - ("mu",  k): upper-bound slack μ_k  (exists only if there are active soft upper bounds at k)

        Parameters
        ----------
        kind : str
            One of {"s","a","lam","mu"}.
        k : int
            Stage index.
        require_layout : bool, default True
            If True, raises RuntimeError when the column layout is not available.
            Set to False to return an empty slice when layout hasn’t been built yet.
        with_width : bool, default False
            If False  → return just the `slice`.
            If True   → return `(slice, width)` where `width = slice.stop - slice.start`
                        (0 when the block is absent).

        Returns
        -------
        slice
            When `with_width=False`.
        (slice, int)
            When `with_width=True`.

        Raises
        ------
        RuntimeError
            If `require_layout=True` and the layout has not been built (call `_build_all_constraints()` first).

        Notes
        -----
        - Absent blocks (e.g., no slacks at step k) return `slice(0, 0)` (and width 0 when `with_width=True`).
        - This function uses `self._col_layout` produced by `_build_all_constraints()` and kept in sync with your
        current constraint/variable structure.

        Examples
        --------
        Reading solution slices:
            # After calling _build_all_constraints()
            s3 = x[self.var_slice("s", 3)]
            a0 = x[self.var_slice("a", 0)]
            lamN = x[self.var_slice("lam", N)]     # empty slice if no λ_N

        Warm-starting (writing into x0):
            sl, w = self.var_slice("lam", 5, with_width=True)
            if w:  # only if λ_5 exists
                x0[sl] = np.zeros((w, 1), dtype=self.dtype)

        Safe usage before layout (will not raise, just empty):
            sl = self.var_slice("mu", 2, require_layout=False)
            # sl is slice(0,0) if layout not built yet
        """
        # Ensure layout
        layout = getattr(self, "_col_layout", None)
        if layout is None:
            if require_layout:
                raise RuntimeError("[MPC SOLVER] ERROR: Column layout is not available. Build constraints first.")
            # graceful empty result
            slc = slice(0, 0)
            return (slc, 0) if with_width else slc

        # Look up slice (empty if absent)
        slc = layout.get((kind, k), slice(0, 0))
        if with_width:
            return slc, (slc.stop - slc.start)
        return slc



    def solve(self):
        """
        Solve the MPC QP with OSQP and return the first action and predicted trajectories.

        Pipeline
        --------
        1) Validates that the initial condition `s_0` is set.
        2) If the problem needs rebuilding (or no solver exists yet), it:
            - builds constraint matrix A and bounds l,u via `_build_all_constraints()`,
            - builds the objective P,q via `_build_objective()`,
            - sets up the OSQP instance with (P, q, A, l, u).
        Otherwise, it only refreshes the **initial-condition rows** of l,u and updates the solver.
        3) Solves with OSQP.
        4) Returns:
            - a_0         : 1D ndarray of length n_a (the first control)
            - a_pred      : 1D ndarray of length N*n_a (stacked a_0..a_{N-1})
            - s_pred      : 1D ndarray of length (N+1)*n_s (stacked s_0..s_N)
            - osqp_status : OSQP status string

        Notes
        -----
        - Variable ordering and sizes (including per-k slacks) are determined by
        `_build_all_constraints()` and stored in `self._col_layout`.
        - If only the initial state changes between calls, this function avoids a rebuild and
        updates just the IC equality rows in (l,u).
        - All vectors passed to OSQP are 1D (raveled); internal storage uses column vectors
        where convenient.

        Raises
        ------
        ValueError
            If `s_0` has not been set prior to solving.
        """
        # Basic checks
        if self.s_0 is None:
            raise ValueError("[MPC SOLVER] ERROR: Initial condition s_0 must be set before solving.")
        if self.N is None:
            raise ValueError("[MPC SOLVER] ERROR: Horizon length self.N is not set.")

        # Optional heads-up if s_0 hasn't been refreshed (your existing style)
        if (not self._s_0_was_updated):
            self._log("warn", "[MPC SOLVER] WARNING: The initial condition s_0 was not updated since the previous solve.")

        # (Re)build if needed
        need_rebuild = self._rebuild_needed or (self.osqp_solver_object is None)
        if need_rebuild:
            # Build all constraints (A, l, u) and get column layout
            col_layout = self._build_all_constraints()
            # Build objective (P, q) using the same layout
            self._build_objective(col_layout=col_layout)

            # Prepare OSQP inputs
            P = self._P_for_objective
            q = self._q_for_objective.ravel()
            A = self._A_for_constraints
            l = self._l_for_constraints.copy().ravel()
            u = self._u_for_constraints.copy().ravel()

            # Ensure the initial condition equality rows (first n_s) reflect current s_0
            l[:self.n_s] = self.s_0.ravel()
            u[:self.n_s] = self.s_0.ravel()

            # Ensure the initial action RoC rows reflect current a_previous
            sl_roc = self._row_slice_a0roc
            if (sl_roc.stop - sl_roc.start) > 0:
                S0 = self.ak_roc_constraint_selector_matrix[0]
                nR0 = int(S0.shape[0])
                # Base bounds
                if self.ak_roc_lower is not None and self.ak_roc_lower[0] is not None:
                    lo0_sel = (S0 @ self.ak_roc_lower[0]).astype(self.dtype, copy=False)
                else:
                    lo0_sel = np.full((nR0, 1), -np.inf, dtype=self.dtype)
                if self.ak_roc_upper is not None and self.ak_roc_upper[0] is not None:
                    up0_sel = (S0 @ self.ak_roc_upper[0]).astype(self.dtype, copy=False)
                else:
                    up0_sel = np.full((nR0, 1),  np.inf, dtype=self.dtype)
                # Choose effective previous action: user-updated a_previous, else a0 from previous solve, else zeros
                if self._a_previous_was_updated:
                    eff_prev = self.a_previous.astype(self.dtype, copy=False)
                elif self.a0_from_previous_solve is not None:
                    eff_prev = self.a0_from_previous_solve.astype(self.dtype, copy=False)
                    #self._log("info", "[MPC SOLVER] INFO: Using a0 from previous solve as a_previous for RoC constraint.")
                else:
                    eff_prev = np.zeros((self.n_a,1), dtype=self.dtype)
                    self._log("warn", "[MPC SOLVER] WARNING: No previous action provided; using zeros for a_previous in RoC constraint.")
                shift0 = S0 @ eff_prev
                l[sl_roc] = (lo0_sel + shift0).ravel()
                u[sl_roc] = (up0_sel + shift0).ravel()

            # Setup OSQP
            self.osqp_solver_object = osqp.OSQP()
            # Optional solver settings bag (dict). If you keep one in __init__, it’ll be used here.
            settings = self.osqp_settings
            self.osqp_solver_object.setup(P=P, q=q, A=A, l=l, u=u, **settings)

            # Mark clean
            self._rebuild_needed = False

            # Keep copies for fast updates next time
            self._l_for_osqp = l
            self._u_for_osqp = u

        else:
            # Only update the initial-condition equality rows in l,u and push to OSQP
            # (Everything else unchanged)
            self._l_for_osqp[:self.n_s] = self.s_0.ravel()
            self._u_for_osqp[:self.n_s] = self.s_0.ravel()
            # Also update initial action RoC rows if present
            sl_roc = self._row_slice_a0roc
            if (sl_roc.stop - sl_roc.start) > 0:
                S0 = self.ak_roc_constraint_selector_matrix[0]
                nR0 = int(S0.shape[0])
                if self.ak_roc_lower is not None and self.ak_roc_lower[0] is not None:
                    lo0_sel = (S0 @ self.ak_roc_lower[0]).astype(self.dtype, copy=False)
                else:
                    lo0_sel = np.full((nR0, 1), -np.inf, dtype=self.dtype)
                if self.ak_roc_upper is not None and self.ak_roc_upper[0] is not None:
                    up0_sel = (S0 @ self.ak_roc_upper[0]).astype(self.dtype, copy=False)
                else:
                    up0_sel = np.full((nR0, 1),  np.inf, dtype=self.dtype)
                # Choose effective previous action as in the build path
                if self._a_previous_was_updated:
                    eff_prev = self.a_previous.astype(self.dtype, copy=False)
                elif self.a0_from_previous_solve is not None:
                    eff_prev = self.a0_from_previous_solve.astype(self.dtype, copy=False)
                    self._log("info", "[MPC SOLVER] INFO: Using a0 from previous solve as a_previous for RoC constraint.")
                else:
                    eff_prev = np.zeros((self.n_a,1), dtype=self.dtype)
                    self._log("warn", "[MPC SOLVER] WARNING: No previous action provided; using zeros for a_previous in RoC constraint.")
                shift0 = S0 @ eff_prev
                self._l_for_osqp[sl_roc] = (lo0_sel + shift0).ravel()
                self._u_for_osqp[sl_roc] = (up0_sel + shift0).ravel()
            self.osqp_solver_object.update(l=self._l_for_osqp, u=self._u_for_osqp)

        # Solve
        osqp_result = self.osqp_solver_object.solve()
        osqp_status = osqp_result.info.status

        if osqp_status != "solved":
            self._log("warn", f"[MPC SOLVER] WARNING: OSQP returned status = {osqp_status}")
            return None, None, None, osqp_status

        x = osqp_result.x  # 1D solution vector

        # Extract stacked predictions using your layout
        # States: s_0..s_N  → concat in order
        s_slices = [self.var_slice("s", k) for k in range(self.N + 1)]
        s_pred = np.concatenate([x[sl] for sl in s_slices], axis=0)

        # Actions: a_0..a_{N-1}
        a_slices = [self.var_slice("a", k) for k in range(self.N)]
        a_pred = np.concatenate([x[sl] for sl in a_slices], axis=0) if a_slices else np.empty((0,), dtype=self.dtype)

        # First control
        a_0 = x[self.var_slice("a", 0)] if self.N > 0 else np.empty((self.n_a,), dtype=self.dtype)

        # Optionally store last solution (handy for warm-starts outside this function)
        self._last_solution = x
        self._last_status = osqp_status

        # Initial condition has been "consumed" for this solve;
        # hence "reset" the flag accordingly
        self._s_0_was_updated = False

        # Save a0 from this solve to serve as next a_previous if user doesn't override
        if self.N > 0 and a_0.size == self.n_a:
            self.a0_from_previous_solve = a_0.reshape((self.n_a, 1)).astype(self.dtype, copy=False)
        # The previous-action has been "consumed" for this solve;
        # hence "reset" the flag accordingly
        self._a_previous_was_updated = False

        return a_0, a_pred, s_pred, osqp_status



    def print_parameters(self, verbose: bool = False, max_k_preview: int = 2):
        """
        Pretty-print a concise summary of the MPC configuration.

        Parameters
        ----------
        verbose : bool, default False
            If True, print small previews for the first/last few time steps of
            time-varying lists (Ak/Bk/gk, costs, refs, bounds, etc.).
        max_k_preview : int, default 2
            How many steps to preview from the head and tail when verbose=True.
        """
        def _fmt_arr(a):
            if a is None: return "None"
            return f"shape={tuple(getattr(a,'shape',()))}, dtype={getattr(getattr(a,'dtype',None),'name',type(a).__name__)}"

        def _fmt_list(lst, expect_len=None):
            if lst is None: return "None"
            L = len(lst)
            base = f"len={L}"
            if expect_len is not None and L != expect_len:
                base += f" (expected {expect_len})"
            # Try to infer element shape/dtype from first non-None
            for it in lst:
                if it is not None:
                    return base + f", elem[{type(it).__name__}]({_fmt_arr(it)})"
            return base + ", all None"

        def _preview_list(lst, label):
            if not verbose or lst is None or len(lst) == 0:
                return
            L = len(lst)
            head = list(range(min(max_k_preview, L)))
            tail = list(range(max(0, L - max_k_preview), L)) if L > max_k_preview else []
            printed = set()
            for k in head + tail:
                if k in printed: continue
                printed.add(k)
                val = lst[k]
                print(f"    {label}[{k}]: {_fmt_arr(val)}")
            if L > 2 * max_k_preview:
                print(f"    ... ({L - 2*max_k_preview} steps omitted)")

        print("=== MPC Configuration Summary ===")
        print(f"Dimensions: n_s={self.n_s}, n_a={self.n_a}, N={self.N}")
        print(f"dtype: {getattr(self, 'dtype', None)}  | allowed: {getattr(self, 'allowed_dtypes', None)}")
        print(f"Build flags: _rebuild_needed={getattr(self, '_rebuild_needed', None)}, "
            f"osqp={'set' if getattr(self, 'osqp_solver_object', None) is not None else 'None'}")

        # ----- Model -----
        print("\n[Model: s_{k+1} = A_k s_k + B_k a_k + g_k]")
        print(f"  LTI flag: {getattr(self, 'is_time_invariant_lin_sys_model', None)}")
        print(f"  Ak: {_fmt_list(getattr(self, 'Ak', None), expect_len=self.N)}")
        print(f"  Bk: {_fmt_list(getattr(self, 'Bk', None), expect_len=self.N)}")
        print(f"  gk: {_fmt_list(getattr(self, 'gk', None), expect_len=self.N)}")
        _preview_list(getattr(self, 'Ak', None), "A")
        _preview_list(getattr(self, 'Bk', None), "B")
        _preview_list(getattr(self, 'gk', None), "g")
        # Derived (sparse)
        print(f"  Ak_sparse: {_fmt_list(getattr(self, 'Ak_sparse', None), expect_len=self.N)}")
        print(f"  Bk_sparse: {_fmt_list(getattr(self, 'Bk_sparse', None), expect_len=self.N)}")

        # ----- Objective -----
        print("\n[Objective coefficients]")
        print(f"  LTI objective flag: {getattr(self, 'is_time_invariant_objective_function', None)}")
        print(f"  Qk: {_fmt_list(getattr(self, 'Qk', None), expect_len=self.N)}")
        print(f"  Rk: {_fmt_list(getattr(self, 'Rk', None), expect_len=self.N)}")
        print(f"  qk: {_fmt_list(getattr(self, 'qk', None), expect_len=self.N)}")
        print(f"  rk: {_fmt_list(getattr(self, 'rk', None), expect_len=self.N)}")
        print(f"  QN: {_fmt_arr(getattr(self, 'QN', None))}")
        print(f"  qN: {_fmt_arr(getattr(self, 'qN', None))}")
        _preview_list(getattr(self, 'Qk', None), "Q")
        _preview_list(getattr(self, 'Rk', None), "R")
        _preview_list(getattr(self, 'qk', None), "q")
        _preview_list(getattr(self, 'rk', None), "r")
        # Derived (sparse)
        print(f"  Qk_sparse: {_fmt_list(getattr(self, 'Qk_sparse', None), expect_len=self.N)}")
        print(f"  Rk_sparse: {_fmt_list(getattr(self, 'Rk_sparse', None), expect_len=self.N)}")
        print(f"  QN_sparse: {_fmt_arr(getattr(self, 'QN_sparse', None))}")

        # ----- References -----
        print("\n[References]")
        print(f"  State refs LTI flag: {getattr(self, 'is_time_invariant_state_reference', None)}")
        print(f"  Action refs LTI flag: {getattr(self, 'is_time_invariant_action_reference', None)}")
        print(f"  sk_ref: {_fmt_list(getattr(self, 'sk_ref', None), expect_len=self.N)}")
        print(f"  ak_ref: {_fmt_list(getattr(self, 'ak_ref', None), expect_len=self.N)}")
        print(f"  sN_ref: {_fmt_arr(getattr(self, 'sN_ref', None))}")
        _preview_list(getattr(self, 'sk_ref', None), "s_ref")
        _preview_list(getattr(self, 'ak_ref', None), "a_ref")

        # ----- Action box constraints -----
        print("\n[Action box constraints]")
        print(f"  LTI flag: {getattr(self, 'is_time_invariant_action_box_constraint', None)}")
        print(f"  ak_lower: {_fmt_list(getattr(self, 'ak_lower', None), expect_len=self.N)}")
        print(f"  ak_upper: {_fmt_list(getattr(self, 'ak_upper', None), expect_len=self.N)}")
        _preview_list(getattr(self, 'ak_lower', None), "a_lower")
        _preview_list(getattr(self, 'ak_upper', None), "a_upper")

        # ----- State soft bounds -----
        print("\n[State soft box constraints]")
        print(f"  Bounds LTI flag: {getattr(self, 'is_time_invariant_state_soft_box_constraint_bounds', None)}")
        print(f"  Coeffs LTI flag: {getattr(self, 'is_time_invariant_state_soft_box_constraint_coefficients', None)}")
        print(f"  sk_soft_lower: {_fmt_list(getattr(self, 'sk_soft_lower', None), expect_len=self.N)}")
        print(f"  sk_soft_upper: {_fmt_list(getattr(self, 'sk_soft_upper', None), expect_len=self.N)}")
        print(f"  sk_soft_lin_coeff:  {_fmt_list(getattr(self, 'sk_soft_lin_coeff', None), expect_len=self.N)}")
        print(f"  sk_soft_quad_coeff: {_fmt_list(getattr(self, 'sk_soft_quad_coeff', None), expect_len=self.N)}")
        _preview_list(getattr(self, 'sk_soft_lower', None), "s_soft_lower")
        _preview_list(getattr(self, 'sk_soft_upper', None), "s_soft_upper")
        _preview_list(getattr(self, 'sk_soft_lin_coeff', None), "soft_lin")
        _preview_list(getattr(self, 'sk_soft_quad_coeff', None), "soft_quad")

        # Derived selectors (per-k, CSC)
        print(f"  n_sk_soft_lower (per-k): {getattr(self, 'n_sk_soft_lower', None)}")
        print(f"  n_sk_soft_upper (per-k): {getattr(self, 'n_sk_soft_upper', None)}")
        # Just show lengths and one preview of selector shapes
        S_low = getattr(self, 'sk_soft_lower_selector_matrix', None)
        S_up  = getattr(self, 'sk_soft_upper_selector_matrix', None)
        if isinstance(S_low, list) and len(S_low) > 0:
            print(f"  sk_soft_lower_selector_matrix: len={len(S_low)}, ex[0].shape={S_low[0].shape}")
        else:
            print(f"  sk_soft_lower_selector_matrix: {_fmt_arr(S_low)}")
        if isinstance(S_up, list) and len(S_up) > 0:
            print(f"  sk_soft_upper_selector_matrix: len={len(S_up)}, ex[0].shape={S_up[0].shape}")
        else:
            print(f"  sk_soft_upper_selector_matrix: {_fmt_arr(S_up)}")

        # ----- Build artifacts (if present) -----
        print("\n[Build artifacts]")
        print(f"  A (constraints): {_fmt_arr(getattr(self, '_A_for_constraints', None))}")
        print(f"  l: {_fmt_arr(getattr(self, '_l_for_constraints', None))}")
        print(f"  u: {_fmt_arr(getattr(self, '_u_for_constraints', None))}")
        print(f"  P (objective): {_fmt_arr(getattr(self, '_P_for_objective', None))}")
        print(f"  q: {_fmt_arr(getattr(self, '_q_for_objective', None))}")
        print(f"  objective constant: {getattr(self, '_objective_constant', None)}")
        print(f"  column layout keys: {sorted(getattr(self, '_col_layout', {}).keys(), key=lambda t: (t[1], t[0]))}")

        # ----- Initial condition & last solve -----
        print("\n[Initial condition / solve state]")
        print(f"  s_0: {_fmt_arr(getattr(self, 's_0', None))}   (_s_0_was_updated={getattr(self, '_s_0_was_updated', None)})")
        print(f"  last status: {getattr(self, '_last_status', None)}")
        print("=== end ===")

    # ---------- Debug/inspection helpers ----------
    def _matrix_to_string(self, M, decimals: int = 3) -> str:
        arr = None
        try:
            import numpy as _np
            from scipy import sparse as _sps
            if _sps.issparse(M):
                arr = M.toarray()
            else:
                arr = _np.asarray(M)
            fmt = { 'float_kind': (lambda x: f"{x:.{decimals}f}") }
            return _np.array2string(arr, max_line_width=200, formatter=fmt)
        except Exception:
            return str(M)

    def print_decision_variable_layout(self, as_row: bool = False):
        """
        Print the stacked decision variable layout in human-readable order.

        Example (as_row=True):
            [s0, a0, s1, lambda1, mu1, a1, ..., sN, lambdaN, muN]
        Slacks only appear if present at that k.
        """
        # Ensure layout
        layout = getattr(self, "_col_layout", None)
        if layout is None:
            self._build_all_constraints()
            layout = self._col_layout
        # Order by column position
        ordered = sorted(layout.items(), key=lambda kv: kv[1].start)
        labels = []
        for (kind, k), sl in ordered:
            if sl.stop - sl.start == 0:
                continue
            if kind == 's': labels.append(f"s{k}")
            elif kind == 'a': labels.append(f"a{k}")
            elif kind == 'lam': labels.append(f"lambda{k}")
            elif kind == 'mu': labels.append(f"mu{k}")
        if as_row:
            print("[" + ", ".join(labels) + "]")
        else:
            print("[")
            for lab in labels:
                print(f"  {lab},")
            print("]")

    def print_P(self, decimals_matrix: int = 3, decimals_vector: int = 3,
                show_decision_variable: bool = False, show_col_labels: bool = False,
                wrap_cols: int | None = None, head_cols: int | None = None, tail_cols: int | None = None,
                zero_epsilon: float = 1e-9, zero_char: str = '.'):
        """
        Print the full P matrix (objective Hessian) aligned by column, with optional
        right-side q vector and decision-variable labels for each row.

        Parameters
        ----------
        decimals : int
            Number of decimal places to print.
        show_decision_variable : bool
            If True, prints the decision variable label on the right-hand side per row.
        """
        # Ensure layout and objective are built
        layout = getattr(self, "_col_layout", None)
        if layout is None:
            layout = self._build_all_constraints()
        if getattr(self, "_P_for_objective", None) is None:
            self._build_objective(col_layout=layout)

        import numpy as _np
        from scipy import sparse as _sps

        # Convert to dense for pretty printing
        P = self._P_for_objective.toarray() if _sps.issparse(self._P_for_objective) else _np.asarray(self._P_for_objective)
        q = self._q_for_objective.reshape((-1, 1)) if getattr(self, "_q_for_objective", None) is not None else _np.zeros((P.shape[0], 1))

        n = P.shape[0]
        # Build per-index labels based on column layout order
        ordered = sorted(layout.items(), key=lambda kv: kv[1].start)
        labels = []
        for (kind, k), sl in ordered:
            w = sl.stop - sl.start
            if w <= 0:
                continue
            name = "s" if kind == "s" else ("a" if kind == "a" else ("lambda" if kind == "lam" else ("mu" if kind == "mu" else str(kind))))
            for idx in range(w):
                labels.append(f"{name}{k}[{idx}]")
        if len(labels) < n:
            labels.extend([""] * (n - len(labels)))

        # Decide which columns to display (head/tail truncation)
        all_cols = list(range(n))
        ELLIPSIS = -1
        if (head_cols is not None or tail_cols is not None) and n > 0:
            h = int(head_cols or 0)
            t = int(tail_cols or 0)
            if h + t < n:
                display_cols = list(range(h)) + ([ELLIPSIS] if t > 0 else []) + list(range(n - t, n))
            else:
                display_cols = all_cols
        else:
            display_cols = all_cols

        # Precompute column widths (only for displayed columns)
        def fmtM(x):
            return f"{float(x):.{decimals_matrix}f}"
        def fmtV(x):
            return f"{float(x):.{decimals_vector}f}"
        col_widths = {}
        for j in display_cols:
            if j == ELLIPSIS:
                col_widths[j] = 3
                continue
            maxw = 0
            for i in range(n):
                s = fmtM(P[i, j])
                maxw = max(maxw, len(s))
            if show_col_labels and j < len(labels):
                maxw = max(maxw, len(labels[j]))
            col_widths[j] = max(maxw, decimals_matrix + 4)
        # Width for q column
        qw = max(decimals_vector + 4, max(len(fmtV(v)) for v in q.ravel()))

        # Create column blocks for wrapping
        def _chunk_cols(cols: list[int | None]) -> list[list[int | None]]:
            if not wrap_cols or wrap_cols <= 0:
                return [cols]
            blocks = []
            cur = []
            for c in cols:
                cur.append(c)
                if len(cur) >= wrap_cols:
                    blocks.append(cur)
                    cur = []
            if cur:
                blocks.append(cur)
            return blocks

        col_blocks = _chunk_cols(display_cols)

        # Optional header per block
        for blk in col_blocks:
            if show_col_labels:
                header = " ".join(("...".rjust(col_widths[j]) if j == ELLIPSIS else (f"{labels[j]:>{col_widths[j]}}" if j < len(labels) else "".rjust(col_widths[j]))) for j in blk)
                print(f"{header} | {'q'.rjust(qw)}" + (" | var" if show_decision_variable else ""))
            # Print rows
            for i in range(n):
                parts = []
                for j in blk:
                    if j == ELLIPSIS:
                        parts.append("...".rjust(col_widths[j]))
                        continue
                    val = P[i, j]
                    if abs(val) < zero_epsilon:
                        parts.append(f"{zero_char:>{col_widths[j]}}")
                    else:
                        s = fmtM(val)
                        parts.append(f"{s:>{col_widths[j]}}")
                row_str = " ".join(parts)
                q_str = f"{fmtV(q[i,0]):>{qw}}"
                if show_decision_variable:
                    label = labels[i] if i < len(labels) else ""
                    print(f"{row_str} | {q_str} | {label}")
                else:
                    print(f"{row_str} | {q_str}")
            # spacer between blocks
            if blk is not col_blocks[-1]:
                print("")

    def print_A_with_bounds(self, decimals_matrix: int = 3, decimals_bounds: int = 3,
                             show_col_labels: bool = False, show_constraint_labels: bool = False,
                             wrap_cols: int | None = None, head_cols: int | None = None, tail_cols: int | None = None,
                             zero_epsilon: float = 1e-9, zero_char: str = '.'):
        """
        Print l | A | u to ease inspection of constraints.
        """
        if getattr(self, "_A_for_constraints", None) is None:
            self._build_all_constraints()
        import numpy as _np
        A = self._A_for_constraints.toarray()
        l = self._l_for_constraints
        u = self._u_for_constraints
        # Build decision variable labels across columns
        layout = getattr(self, "_col_layout", {})
        ordered = sorted(layout.items(), key=lambda kv: kv[1].start)
        labels = []
        ncols = A.shape[1]
        for (kind, k), sl in ordered:
            w = sl.stop - sl.start
            if w <= 0:
                continue
            name = "s" if kind == "s" else ("a" if kind == "a" else ("lambda" if kind == "lam" else ("mu" if kind == "mu" else str(kind))))
            for idx in range(w):
                labels.append(f"{name}{k}[{idx}]")
        if len(labels) < ncols:
            labels.extend([""] * (ncols - len(labels)))

        def fmtM(x): return f"{float(x):.{decimals_matrix}f}"
        def fmtB(x): return f"{float(x):.{decimals_bounds}f}"

        # Decide display columns (head/tail truncation)
        all_cols = list(range(ncols))
        ELLIPSIS = -1
        if (head_cols is not None or tail_cols is not None) and ncols > 0:
            h = int(head_cols or 0)
            t = int(tail_cols or 0)
            if h + t < ncols:
                display_cols = list(range(h)) + ([ELLIPSIS] if t > 0 else []) + list(range(ncols - t, ncols))
            else:
                display_cols = all_cols
        else:
            display_cols = all_cols

        # Compute per-column widths considering matrix values and header labels
        col_widths = {}
        for j in display_cols:
            if j == ELLIPSIS:
                col_widths[j] = 3
                continue
            maxw = 0
            for i in range(A.shape[0]):
                maxw = max(maxw, len(fmtM(A[i, j])))
            if show_col_labels and j < len(labels):
                maxw = max(maxw, len(labels[j]))
            col_widths[j] = max(maxw, decimals_matrix + 4)
        # Width for bounds columns
        lw = max(decimals_bounds + 4, max(len(fmtB(v)) for v in l.ravel())) if l is not None else 0
        uw = max(decimals_bounds + 4, max(len(fmtB(v)) for v in u.ravel())) if u is not None else 0

        # Create column blocks for wrapping
        def _chunk_cols(cols: list[int | None]) -> list[list[int | None]]:
            if not wrap_cols or wrap_cols <= 0:
                return [cols]
            blocks = []
            cur = []
            for c in cols:
                cur.append(c)
                if len(cur) >= wrap_cols:
                    blocks.append(cur)
                    cur = []
            if cur:
                blocks.append(cur)
            return blocks

        col_blocks = _chunk_cols(display_cols)

        # Row labels (constraints) — compute from current configuration
        row_labels = []
        if show_constraint_labels:
            N = self.N or 0
            n_s = self.n_s; n_a = self.n_a
            # init state
            row_labels.extend(["init state"] * n_s)
            # init action roc
            if getattr(self, "ak_roc_constraint_selector_matrix", None) is not None and len(self.ak_roc_constraint_selector_matrix) > 0:
                S0 = self.ak_roc_constraint_selector_matrix[0]
                nR0 = int(S0.shape[0])
                if nR0 > 0:
                    row_labels.extend(["action roc k=0"] * nR0)
            # init action box (k=0)
            row_labels.extend(["action box k=0"] * n_a)
            # k=0 dynamics
            row_labels.extend([f"model[{i}] k=0" for i in range(n_s)])
            # k=1..N-1
            for k in range(1, N):
                i = k - 1
                # state box
                if getattr(self, "sk_box_constraint_selector_matrix", None) is not None:
                    nB = int(self.sk_box_constraint_selector_matrix[i].shape[0])
                    row_labels.extend([f"state box k={k}"] * nB)
                # soft lower + lambda >= 0
                nL = int(self.sk_soft_lower_selector_matrix[i].shape[0]) if getattr(self, "sk_soft_lower_selector_matrix", None) is not None else 0
                row_labels.extend([f"state soft lower k={k}"] * nL)
                row_labels.extend([f"slack lambda k={k} >= 0"] * (nL if nL > 0 else 0))
                # soft upper + mu >= 0
                nU = int(self.sk_soft_upper_selector_matrix[i].shape[0]) if getattr(self, "sk_soft_upper_selector_matrix", None) is not None else 0
                row_labels.extend([f"state soft upper k={k}"] * nU)
                row_labels.extend([f"slack mu k={k} >= 0"] * (nU if nU > 0 else 0))
                # action box
                row_labels.extend([f"action box k={k}"] * n_a)
                # action roc
                if getattr(self, "ak_roc_constraint_selector_matrix", None) is not None:
                    nRk = int(self.ak_roc_constraint_selector_matrix[k].shape[0])
                    row_labels.extend([f"action roc k={k}"] * nRk)
                # dynamics
                row_labels.extend([f"model[{i}] k={k}" for i in range(n_s)])
            # terminal (k=N)
            i = N - 1
            if N > 0:
                if getattr(self, "sk_box_constraint_selector_matrix", None) is not None:
                    nBN = int(self.sk_box_constraint_selector_matrix[i].shape[0])
                    row_labels.extend([f"state box k={N}"] * nBN)
                nL = int(self.sk_soft_lower_selector_matrix[i].shape[0]) if getattr(self, "sk_soft_lower_selector_matrix", None) is not None else 0
                row_labels.extend([f"state soft lower k={N}"] * nL)
                row_labels.extend([f"slack lambda k={N} >= 0"] * (nL if nL > 0 else 0))
                nU = int(self.sk_soft_upper_selector_matrix[i].shape[0]) if getattr(self, "sk_soft_upper_selector_matrix", None) is not None else 0
                row_labels.extend([f"state soft upper k={N}"] * nU)
                row_labels.extend([f"slack mu k={N} >= 0"] * (nU if nU > 0 else 0))

        for blk in col_blocks:
            # Optional header for this block
            if show_col_labels:
                header = " ".join(("...".rjust(col_widths[j]) if j == ELLIPSIS else (f"{labels[j]:>{col_widths[j]}}" if j < len(labels) else "".rjust(col_widths[j]))) for j in blk)
                left_hdr = "l".rjust(lw) if l is not None else ""
                right_hdr = "u".rjust(uw) if u is not None else ""
                extra_hdr = " | constraint" if show_constraint_labels else ""
                print(f"{left_hdr} | {header} | {right_hdr}{extra_hdr}")

            for i in range(A.shape[0]):
                left = (f"{fmtB(l[i,0]):>{lw}}" if l is not None else "")
                right = (f"{fmtB(u[i,0]):>{uw}}" if u is not None else "")
                row_parts = []
                for j in blk:
                    if j == ELLIPSIS:
                        row_parts.append("...".rjust(col_widths[j]))
                        continue
                    val = A[i, j]
                    if abs(val) < zero_epsilon:
                        row_parts.append(f"{zero_char:>{col_widths[j]}}")
                    else:
                        row_parts.append(f"{fmtM(val):>{col_widths[j]}}")
                row = " ".join(row_parts)
                suffix = (" | " + (row_labels[i] if i < len(row_labels) else "")) if show_constraint_labels else ""
                print(f"{left} | {row} | {right}{suffix}")
            if blk is not col_blocks[-1]:
                print("")


# ---------- Tiny smoke test ----------
def smoke_test_action_roc():
    """
    Construct a tiny MPC with N=3, n_s=1, n_a=1; add action RoC bounds and
    run two solves to validate a_previous update path and printing helpers.
    """
    import numpy as np
    mpc = MPCSolverForLinSysQuadObjTimeVarying(n_s=1, n_a=1, dtype=np.float64)
    mpc.silent_mode = False
    mpc.set_horizon(3)

    # Model: s_{k+1} = s_k + a_k (unit integrator)
    A = np.array([[2.0]])
    B = np.array([[3.0]])
    Ak_list = [A, A, A]
    Bk_list = [B, B, B]
    gk_list = [np.zeros((1,1)) for _ in range(3)]
    mpc.set_lin_sys_model_time_varying(Ak_list, Bk_list, gk_list)

    # Objective: penalize actions and states mildly
    Q = np.array([[0.4]])
    R = np.array([[0.5]])
    mpc.set_objective_function_coefficients(Q=Q, R=R, QN=Q)

    # Bounds: action in [-1, 1]
    mpc.set_action_box_constraints(a_lower=np.array([[-1.0]]), a_upper=np.array([[1.0]]))

    # Action RoC: delta a in [-0.2, 0.2]
    aroc_lo = np.array([[-0.6]])
    aroc_hi = np.array([[ 0.6]])
    mpc.set_action_rate_of_change_constraints(aroc_lo, aroc_hi)

    # Initial state and previous action
    mpc.set_initial_condition(np.array([[0.7]]))
    mpc.set_previous_action(np.array([[0.8]]))

    # Solve 1 (uses user-provided a_previous)
    a0, a_pred, s_pred, status = mpc.solve()
    print("\nSolve 1 status:", status)
    print("Decision variable layout:")
    mpc.print_decision_variable_layout(as_row=True)
    print("\nP matrix:")
    mpc.print_P(decimals_matrix=2, decimals_vector=2, show_decision_variable=True, show_col_labels=True)
    print("\nConstraints l | A | u:")
    mpc.print_A_with_bounds(decimals_matrix=2, decimals_bounds=2, show_col_labels=True, show_constraint_labels=True)

    # Solve 2 (uses a0_from_previous_solve implicitly)
    a0b, a_pred2, s_pred2, status2 = mpc.solve()
    print("\nSolve 2 status:", status2)
    print("Decision variable layout:")
    mpc.print_decision_variable_layout(as_row=True)
    print("\nConstraints l | A | u:")
    mpc.print_A_with_bounds(decimals_matrix=2, decimals_bounds=2, show_col_labels=True, show_constraint_labels=True)
