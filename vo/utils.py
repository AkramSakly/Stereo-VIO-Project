import numpy as np


def align_umeyama(model, data):
    model = np.asarray(model).T
    data = np.asarray(data).T

    mu_m = model.mean(axis=1, keepdims=True)
    mu_d = data.mean(axis=1, keepdims=True)

    model_c = model - mu_m
    data_c = data - mu_d

    cov = (data_c @ model_c.T) / model.shape[1]

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1

    R = U @ S @ Vt

    scale = np.trace(np.diag(D) @ S) / np.sum(model_c**2)

    t = mu_d - scale * R @ mu_m

    return (scale * R @ model + t).T


def smooth(traj, w=5):
    out = []
    for i in range(len(traj)):
        start = max(0, i-w)
        out.append(np.mean(traj[start:i+1], axis=0))
    return np.array(out)