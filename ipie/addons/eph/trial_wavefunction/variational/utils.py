import numpy as np

def D2_to_DTilde(shift: np.ndarray):
    """Takes D2 shift and reformats to DTilde initial guess."""
    shift = np.squeeze(shift)
    return np.vstack([shift, np.zeros(len(shift))]).T

def DTilde_to_D1(shift: np.ndarray):
    """Takes DTilde initial guess and returns shift matrix."""
    shift = np.squeeze(shift)
    alpha = shift[:,0]
    beta = shift[:,1]
    shift_mat = np.repeat(alpha[:, np.newaxis], len(alpha), axis=1)
    
    circs = beta
    for s in range(1, len(beta)):
        new_circ = np.roll(beta, s)
        circs = np.vstack([circs, new_circ])
    
    shift_mat -= circs.T

    return shift_mat

def D2_to_D1(shift: np.ndarray):
    return DTilde_to_D1(D2_to_DTilde(shift))

