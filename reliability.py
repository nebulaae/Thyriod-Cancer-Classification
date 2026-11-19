"""
Calibration utilities: expected calibration error and temperature scaling.

Temperature scaling (Guo et al., ICML 2017) divides the logit by a single
scalar T > 0 fitted on a held-out selection set by minimising the negative
log-likelihood. It changes confidence but never the ranking, so AUROC is unchanged.
"""

import numpy as np
import torch
import torch.nn.functional as F


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def expected_calibration_error(y, p, n_bins=10):
    """Equal-width-bin ECE for binary probabilities P(y=1)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(ece)


def fit_temperature(logits, y, max_iter=200):
    """Fit T > 0 minimising BCE(logits / T, y). Returns a float."""
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    t = torch.as_tensor(np.asarray(y, dtype=np.float64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / torch.exp(log_t), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())
