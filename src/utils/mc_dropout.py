# src/utils/mc_dropout.py
import math
import torch

@torch.no_grad()
def mc_predict(model, batch, T=50):
    """
    Monte Carlo dropout prediction.
    Args:
        model: nn.Module
        batch: (x_enc, x_mark_enc, dec_inp, y_mark)
        T: number of stochastic forward passes
    Returns:
        mean: [B, P, F]
        std:  [B, P, F]
    """
    x_enc, x_mark_enc, dec_inp, y_mark = batch

    # enable dropout during inference
    prev_training = model.training
    model.train()

    preds = []
    for _ in range(T):
        out = model(x_enc, x_mark_enc, dec_inp, y_mark)
        y = out[0] if isinstance(out, tuple) else out   # unwrap if (pred, attn)
        preds.append(y.unsqueeze(0))                    # [1,B,P,F]

    preds = torch.cat(preds, dim=0)  # [T,B,P,F]
    mean = preds.mean(dim=0)
    std  = preds.std(dim=0, unbiased=False)

    if not prev_training:
        model.eval()

    return mean, std


def _normal_ppf(p: float, device=None, dtype=None) -> torch.Tensor:
    """
    Standard normal inverse CDF using erf^-1:
      Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
    """
    t = torch.tensor(2.0 * p - 1.0, device=device, dtype=dtype)
    return math.sqrt(2.0) * torch.erfinv(t)


@torch.no_grad()
def gaussian_pi(mean: torch.Tensor, std: torch.Tensor, alpha: float):
    """
    Gaussian (1-alpha) predictive interval given mean and std.
    Args:
        mean: [B,P,F]
        std:  [B,P,F]
        alpha: e.g., 0.05 for 95% PI
    Returns:
        lower, upper: [B,P,F]
    """
    # clamp std to avoid zero-width intervals
    std = torch.clamp(std, min=1e-8)
    p = 1.0 - alpha / 2.0
    z = _normal_ppf(p, device=mean.device, dtype=mean.dtype)
    lower = mean - z * std
    upper = mean + z * std
    return lower, upper


@torch.no_grad()
def picp_mpiw(lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor):
    """
    PICP/MPIW for predictive intervals.
    Args:
        lower, upper: [B,P,F]
        y_true:       [B,P,F]
    Returns:
        picp (float in [0,1]), mpiw (float)
    """
    # ensure broadcast safety
    covered = (y_true >= lower) & (y_true <= upper)
    picp = covered.float().mean().item()
    mpiw = (upper - lower).mean().item()
    return picp, mpiw
