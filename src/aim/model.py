"""Free-energy model used throughout the project."""

import numpy as np


# The exact parameters used in the paper's example figures are not publicly
# reported. We therefore use a synthetic symmetric configuration that produces
# a clearly bistable landscape. These values are not empirical estimates.
L1 = L2 = 300
L12 = 100
T1 = T2 = 250
N1 = N2 = 100
BETA = 1


def F(y1, y2):
    """Calculate the AIM free energy (Eq. 2 of the paper)."""
    free_energy = -L1 * y1**2 + T1 * y1
    free_energy += -L2 * y2**2 + T2 * y2
    free_energy += L12 * y1 * y2

    free_energy += (N1 / BETA) * (
        y1 * np.log(y1) + (1 - y1) * np.log(1 - y1)
    )
    free_energy += (N2 / BETA) * (
        y2 * np.log(y2) + (1 - y2) * np.log(1 - y2)
    )

    return free_energy


def grad_F(y1, y2):
    """Calculate both components of the AIM free-energy gradient."""
    dF_dy1 = (
        -2 * L1 * y1
        + T1
        + L12 * y2
        + (N1 / BETA) * np.log(y1 / (1 - y1))
    )

    dF_dy2 = (
        -2 * L2 * y2
        + T2
        + L12 * y1
        + (N2 / BETA) * np.log(y2 / (1 - y2))
    )

    return dF_dy1, dF_dy2
