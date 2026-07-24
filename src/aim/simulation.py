"""Reusable simulation functions for the Affective Ising Model."""

import numpy as np

from .model import BETA, grad_F


def euler_maruyama(y0, dt, T, D, seed=None):
    """Simulate the SDE using the Euler-Maruyama method."""

    random_number_generator = np.random.default_rng(seed) #We create a random number generator with the given seed for reproducibility
    n_steps = int(T / dt) #We calculate the number of steps needed to reach the final time T 
    print(n_steps)
    times=np.linspace(0, T, n_steps+1) #function to create the time vector

    #we need now to create amatrix to store the values of y1 and y2 at each time step
    traj=np.zeros((n_steps+1, 2)) #We create a matrix of zeros with n_steps rows and 2 columns (for y1 and y2), the +1 is because we include the initial condition
    print("Number of steps:", n_steps)
    print("Times shape:", times.shape)
    print("Trajectory shape:", traj.shape)
    #we set the initial condition
    traj[0, :] = y0
    #for every time step, we calculate the new value of y1 and y2 using the Euler-Maruyama method
    for i in range(n_steps):
        y1, y2 = traj[i, :]
        drift = -BETA*D*np.array(grad_F(y1, y2)) #We calculate the drift term using the gradient of the free energy landscape
        noise = np.sqrt(2 * D * dt) * random_number_generator.normal(size=2) #We calculate the noise term using a normal distribution with mean 0 and variance 1
        new_state = traj[i, :] + drift * dt + noise #We update the values of y1 and y2 using the Euler-Maruyama method
        traj[i + 1, :] = np.clip(new_state, 0.00001, 0.99999) #We force the value to be betwwen 0 and 1

    return times, traj
