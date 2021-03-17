# MRAC Adaptive controller

# Import libraries
import numpy as np
from base_controller import BaseController
from lqr_solver import dlqr, lqr
from scipy.linalg import solve_continuous_lyapunov, solve_lyapunov, solve_discrete_lyapunov
from math import cos, sin
import numpy as np
from scipy import signal

class AdaptiveController(BaseController):
    """ The LQR controller class.

    """

    def __init__(self, robot, lossOfThurst):
        """ MRAC adaptive controller __init__ method.

        Initialize parameters here.

        Args:
            robot (webots controller object): Controller for the drone.
            lossOfThrust (float): percent lost of thrust.

        """

        super().__init__(robot, lossOfThurst)

        # define integral error
        self.int_e1 = 0
        self.int_e2 = 0
        self.int_e3 = 0
        self.int_e4 = 0

        # flag for initializing adaptive controller
        self.have_initialized_adaptive = False

        # reference model
        self.x_m = None

        # baseline LQR controller gain
        self.Kbl = None

        # Saved matrix for adaptive law computation
        self.A_d = None
        self.B_d = None
        self.Bc_d = None

        self.B = None
        self.Gamma = None
        self.P = None

        # adaptive gain
        self.K_ad = None

    def initializeGainMatrix(self):
        """ Calculate the LQR gain matrix and matrices for adaptive controller.

        """

        # ---------------|LQR Controller|-------------------------
        # Use the results of linearization to create a state-space model

        n_p = 12 # number of states
        m_i = 4 # number of integral error terms

        # ----------------- Your Code Here ----------------- #
        # Compute the continuous A, B, Bc, C, D and
        # discretized A_d, B_d, Bc_d, C_d, D_d, for the computation of LQR gain
        
        # Continuous-Time State Space
        A1 = np.zeros((6, 6))
        A2 = np.eye(6)
        A3 = np.zeros((6, 6))
        A3[0][4] = self.g
        A3[1][3] = -1*self.g
        A4 = A1
        
        A12 = np.concatenate((A1, A2), 1) # Column-wise concatenation
        A34 = np.concatenate((A3, A4), 1)
        
        Ap = np.concatenate((A12, A34)) # n_p x n_p (12 x 12) Matrix
        
        B1 = np.zeros((8, m_i))
        B2 = np.zeros((m_i, m_i))
        
        B2[0][0] = 1/self.m
        B2[1][1] = 1/self.Ix
        B2[2][2] = 1/self.Iy
        B2[3][3] = 1/self.Iz

        Bp = np.concatenate((B1, B2)) # n_p x m_i (12 x 4) Matrix
        
        Cp = np.zeros((4, 12)) # m_i x n_p (4 x 12) Matrix
        Cp[0][0] = 1
        Cp[1][1] = 1
        Cp[2][2] = 1
        Cp[3][5] = 1
        
        A1 = Ap
        A2 = np.zeros((n_p, m_i))
        A3 = Cp
        A4 = np.zeros((m_i, m_i))

        A12 = np.concatenate((A1, A2), 1)
        A34 = np.concatenate((A3, A4), 1)
        
        At = np.concatenate((A12, A34)) # (16 x 16) Matrix
        A = At
        
        Bt = np.concatenate((Bp, A4)) # (16 x 4) Matrix, I wanted to combine a 4x4 zero matrix, and A4 is just that
        
        B1 = np.zeros((n_p, m_i))
        B2 = -np.eye(m_i)
        
        Bc = np.concatenate((B1, B2)) # (16 x 4) Matrix
        
        B = np.concatenate((Bt, Bc), 1) # (16 x 8) Matrix
        
        Ct = np.concatenate((Cp, A4), 1) # (4 x 16) Matrix
        C = Ct
        
        D = np.zeros((m_i, 1)) 
                
        # Discretization of CT SS
        A_d, B_d, C_d, D_d, delT = signal.cont2discrete((A, B, C, D), 0.01)
        B = B[:,:4]
        Bc_d = B_d[:,4:]
        B_d = B_d[:,:4] 
          

        # ----------------- Your Code Ends Here ----------------- #

        # Record the matrix for later use
        self.A = A # continuous version of A
        self.B = B  # continuous version of B
        self.A_d = A_d  # discrete version of A
        self.B_d = B_d  # discrete version of B
        self.Bc_d = Bc_d  # discrete version of Bc

        # -----------------    Example code     ----------------- #
        max_pos = 15.0
        max_ang = 0.2 * self.pi
        max_vel = 6.0
        max_rate = 0.015 * self.pi
        max_eyI = 3. 

        max_states = np.array([0.1 * max_pos, 0.1 * max_pos, max_pos,
                            max_ang, max_ang, max_ang,
                            0.5 * max_vel, 0.5 * max_vel, max_vel,
                            max_rate, max_rate, max_rate,
                            0.1 * max_eyI, 0.1 * max_eyI, 1 * max_eyI, 0.1 * max_eyI])

        max_inputs = np.array([0.2 * self.U1_max, self.U1_max, self.U1_max, self.U1_max])

        Q = np.diag(1/max_states**2)
        R = np.diag(1/max_inputs**2)
        # -----------------  Example code Ends ----------------- #
        # ----------------- Your Code Here ----------------- #
        # Come up with reasonable values for Q and R (state and control weights)
        # The example code above is a good starting point, feel free to use them or write you own.
        # Tune them to get the better performance

        # ----------------- Your Code Ends Here ----------------- #

        # solve for LQR gains   
        [K, _, _] = dlqr(A_d, B_d, Q, R)

        self.Kbl = -K

        [K_CT, _, _] = lqr(A, B, Q, R)
        Kbl_CT = -K_CT

        # initialize adaptive controller gain to baseline LQR controller gain
        self.K_ad = self.Kbl.T

        # -----------------    Example code     ----------------- #
        self.Gamma = 3e-3 * np.eye(16)

        Q_lyap = np.copy(Q)
        Q_lyap[0:3,0:3] *= 30
        Q_lyap[6:9,6:9] *= 150
        Q_lyap[14,14] *= 2e-3
        # -----------------  Example code Ends ----------------- #
        # ----------------- Your Code Here ----------------- #
        # Come up with reasonable value for Gamma matrix and Q_lyap
        # The example code above is a good starting point, feel free to use them or write you own.
        # Tune them to get the better performance        

        # ----------------- Your Code Ends Here ----------------- #

        A_m = A + self.B @ Kbl_CT
        self.P = solve_continuous_lyapunov(A_m.T, -Q_lyap)

    def update(self, r):
        """ Get current states and calculate desired control input.

        Args:
            r (np.array): reference trajectory.

        Returns:
            np.array: states. information of the 16 states.
            np.array: U. desired control input.

        """

        U = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1,1)

        # Fetch the states from the BaseController method
        x_t = super().getStates()

        # update integral term
        self.int_e1 += float((x_t[0]-r[0])*(self.timestep*1e-3))
        self.int_e2 += float((x_t[1]-r[1])*(self.timestep*1e-3))
        self.int_e3 += float((x_t[2]-r[2])*(self.timestep*1e-3))
        self.int_e4 += float((x_t[5]-r[3])*(self.timestep*1e-3))

        # Assemble error-based states into array
        error_state = np.array([self.int_e1, self.int_e2, self.int_e3, self.int_e4]).reshape((-1,1))
        states = np.concatenate((x_t, error_state))

        # initialize adaptive controller
        if self.have_initialized_adaptive == False:
            print("Initialize adaptive controller")
            self.x_m = states
            self.have_initialized_adaptive = True
        else:
            # ----------------- Your Code Here ----------------- #
            # adaptive controller update law
            # Update self.K_ad by first order approximation:
            error = states - self.x_m
            rate_of_change = -self.Gamma @ states @ error.T @ self.P @ self.B
            self.K_ad = self.K_ad + rate_of_change * self.timestep * 1e-3

            # ----------------- Your Code Ends Here ----------------- #
            
            # compute x_m at k+1
            self.x_m = self.A_d @ self.x_m + self.B_d @ self.Kbl @ self.x_m + self.Bc_d @ r 
            # Compute control input
            U = self.K_ad.T @ states

        # calculate control input
        U[0] += self.g * self.m


        # Return all states and calculated control inputs U
        return states, U