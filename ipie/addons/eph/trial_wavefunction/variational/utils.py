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

def Dn_to_Dm(shift: np.ndarray, n: int, m: int):
    print('shift shape: ', shift.shape, n, m)
    diff = (m - n) // 2
    sites = shift.shape[0]
#    shift_mat = np.einsum('ij,jk->ik', shift.conj()[:, :n//2], shift[:, n//2 :].T)
    new_shift = np.hstack([shift[:, :n//2], np.zeros((sites, diff)), shift[:, n//2:], np.zeros((sites, diff))]) 
#    print('shift mat shape: ', shift_mat.shape)
#    s,v,d = np.linalg.svd(shift_mat)
#    print('s shape: ', s[:, : m//2].shape)
#    print('d shape: ', d[: m//2, :].shape)
#    new_shift = np.hstack([s[:, : m//2] * v[: m//2], d[: m//2, :].T])
    print('new_shift shape: ', new_shift.shape)
    return new_shift
