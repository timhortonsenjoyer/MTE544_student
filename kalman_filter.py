import numpy as np

# TODO Part 3: Comment the code explaining each part
class kalman_filter:
    """
    A class to implement a Kalman filter for state estimation.
    
    Attributes:
        P (np.ndarray): The initial covariance matrix.
        Q (np.ndarray): The process noise covariance matrix.
        R (np.ndarray): The measurement noise covariance matrix.
        x (np.ndarray): The state vector.
        dt (float): The time step.
    """
    # TODO Part 3: Initialize the covariances and the states    
    def __init__(self, P, Q, R, x, dt):
        """
        Initializes the Kalman filter with the given parameters.
        
        Parameters:
            P (list or np.ndarray): Initial covariance matrix.
            Q (list or np.ndarray): Process noise covariance matrix.
            R (list or np.ndarray): Measurement noise covariance matrix.
            x (list or np.ndarray): Initial state vector.
            dt (float): Time step.
        """
        
        self.P = np.array(P)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.x = np.array(x)
        self.dt = np.array(dt)
        
    # TODO Part 3: Replace the matrices with Jacobians where needed        
    def predict(self):
        """
        Predicts the next state and updates the covariance matrix.
        """
        self.A = self.jacobian_A()
        self.C = self.jacobian_H()
        
        self.motion_model()
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    # TODO Part 3: Replace the matrices with Jacobians where needed
    def update(self, z):
        """
        Updates the state with the measurement z.
        
        Parameters:
            z (np.ndarray): The measurement vector.
        """
        S = np.dot(np.dot(self.C, self.P), self.C.T) + self.R
            
        kalman_gain = np.dot(np.dot(self.P, self.C.T), np.linalg.inv(S))
        surprise_error = z - self.measurement_model()
        
        self.x = self.x + np.dot(kalman_gain, surprise_error)
        self.P = np.dot((np.eye(self.A.shape[0]) - np.dot(kalman_gain, self.C)), self.P)
    
    # TODO Part 3: Implement here the measurement model
    def measurement_model(self):
        """
        Defines the measurement model.
        
        Returns:
            np.ndarray: The predicted measurement based on the current state.
        """
        x, y, th, w, v, vdot = self.x
        return np.array([
            v,    # v
            w,    # w
            vdot, # ax
            w * v # ay
        ])
        
    # TODO Part 3: Impelment the motion model (state-transition matrice)
    def motion_model(self):
        """
        Implements the motion model (state-transition matrix).
        """
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            th + w * dt,
            w,
            v + vdot * dt,
            vdot,
        ])
        
    def jacobian_A(self):
        """
        Computes the Jacobian of the state transition model.
        
        Returns:
            np.ndarray: The Jacobian matrix A.
        """
        x, y, th, w, v, vdot = self.x
        dt = self.dt
        
        return np.array([
            [1, 0,              -v * np.sin(th) * dt, 0,          np.cos(th) * dt,  0],
            [0, 1,              v * np.cos(th) * dt, 0,          np.sin(th) * dt,  0],
            [0, 0,                1, dt,           0,  0],
            [0, 0,                0, 1,            0,  0],
            [0, 0,                0, 0,            1,  dt],
            [0, 0,                0, 0,            0,  1]
        ])
    
    # TODO Part 3: Implement here the jacobian of the H matrix (measurements)    
    def jacobian_H(self):
        """
        Computes the Jacobian of the measurement model.
        
        Returns:
            np.ndarray: The Jacobian matrix H.
        """
        x, y, th, w, v, vdot = self.x
        return np.array([
            [0, 0, 0, 0, 1, 0], # v
            [0, 0, 0, 1, 0, 0], # w
            [0, 0, 0, 0, 0, 1], # ax
            [0, 0, 0, v, w, 0], # ay
        ])
        
    # TODO Part 3: return the states here    
    def get_states(self):
        """
        Returns the current state vector.
        
        Returns:
            np.ndarray: The current state vector x.
        """
        return self.x
