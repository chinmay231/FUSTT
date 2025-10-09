# src/analysis/shap_fusst.py
import numpy as np
import torch
import shap

class PredictWrapper:
    """
    Wraps model inference so SHAP can call it on flat vectors.
    Returns ONE scalar per sample: mean forecast over horizon for the target feature.
    This makes Kernel SHAP return a single [n, LF] SHAP matrix (not a list).
    """
    def __init__(self, model, build_batch_fn, target_feature_index: int, seq_len: int, n_features: int, device="cpu"):
        self.model = model.eval()
        self.build = build_batch_fn
        self.fidx = target_feature_index
        self.L = seq_len
        self.F = n_features
        self.device = device

    @torch.no_grad()
    def __call__(self, Xflat: np.ndarray):
        # Xflat: [n, L*F] (Kernel SHAP perturbs in flat space)
        if Xflat.ndim == 1:
            Xflat = Xflat[None, :]
        n = Xflat.shape[0]
        X = torch.tensor(Xflat, dtype=torch.float32, device=self.device).view(n, self.L, self.F)  # [n,L,F]

        x_enc, x_mark_enc, x_dec, x_mark_dec = self.build(X)  # all on device

        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(out, (tuple, list)):
            out = out[0]                       # [n, P, F]
        y = out[..., self.fidx]                # [n, P]
        y_scalar = y.mean(dim=1)               # [n]  (mean over horizon)
        return y_scalar.detach().cpu().numpy()

def compute_global_shap(model, build_batch_fn, X_bg, X_test,
                        target_feature_index, seq_len, n_features,
                        nsamples=300, device="cpu"):
    """
    X_bg, X_test: tensors or np arrays shaped [n, L, F]
    Returns:
      - feat_imp: [F] global mean |SHAP| per feature (summed over time, averaged over samples)
      - lag_imp:  [L] global mean |SHAP| per lag (summed over features, averaged over samples)
      - shap_vals_lf: [n, L, F] SHAP values per sample reshaped back to [L,F]
    """
    # flatten for KernelExplainer
    n_bg, L, F = X_bg.shape
    n_te, _, _ = X_test.shape
    X_bg_flat = X_bg.reshape(n_bg, L * F)
    X_te_flat = X_test.reshape(n_te, L * F)

    f = PredictWrapper(model, build_batch_fn, target_feature_index, seq_len=L, n_features=F, device=device)
    explainer = shap.KernelExplainer(f, X_bg_flat)
    shap_vals_flat = explainer.shap_values(X_te_flat, nsamples=nsamples)   # [n_te, L*F]
    shap_vals_flat = np.asarray(shap_vals_flat)  # ensure ndarray

    # reshape back to [n, L, F]
    shap_vals_lf = shap_vals_flat.reshape(n_te, L, F)

    # aggregate
    mean_abs = np.mean(np.abs(shap_vals_lf), axis=0)  # [L, F], mean over samples
    feat_imp = mean_abs.sum(axis=0)                   # [F]
    lag_imp  = mean_abs.sum(axis=1)                   # [L]
    return feat_imp, lag_imp, shap_vals_lf
