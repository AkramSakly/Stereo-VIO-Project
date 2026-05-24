import numpy as np

def compute_ate(gt, est):

    error = gt - est

    return np.sqrt(np.mean(np.sum(error**2, axis=1)))


def compute_rpe(gt, est):

    gt_rel = gt[1:] - gt[:-1]
    est_rel = est[1:] - est[:-1]

    error = gt_rel - est_rel

    return np.sqrt(np.mean(np.sum(error**2, axis=1)))


def align_umeyama(model, data):

    model_mean = model.mean(axis=0)
    data_mean = data.mean(axis=0)

    model_c = model - model_mean
    data_c = data - data_mean

    cov = data_c.T @ model_c / len(model)

    U, D, Vt = np.linalg.svd(cov)

    R = U @ Vt

    scale = np.trace(np.diag(D)) / np.var(model_c)

    t = data_mean - scale * R @ model_mean

    aligned = scale * (R @ model.T).T + t

    return aligned