# src/tools/analyze_shap.py
import os
import numpy as np
import torch
import shap
from types import SimpleNamespace

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider

# --------- 1) Rebuild the same args you trained with ----------
SETTING = os.environ.get(
    "FUSST_SETTING",
    "long_term_forecast_DiscoveryPassage_and_others_720_360_96_FUSST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"
)

args = SimpleNamespace(
    # dataset / task
    task_name="long_term_forecast",
    data="custom",
    root_path="C:/Users/kapoo/FUSTT/src/dataset",
    data_path="Data_run.csv",
    target="Chlorophyll (ug/l)",
    features="M",
    freq="h",
    embed='timeF',
    scale=True,
    inverse=False,

    # model choice and ID
    model="FUSST",
    model_id="DiscoveryPassage_and_others_720_360_96",

    # architecture (must match training)
    seq_len=96,
    label_len=48,
    pred_len=96,
    e_layers=2,
    d_layers=1,
    factor=3,
    enc_in=8,
    dec_in=8,
    c_out=8,
    n_heads=8,
    d_model=512,
    d_ff=2048,
    dropout=0.1,
    activation="gelu",
    moving_avg=25,
    distil=True,
    top_k=5,
    seasonal_patterns="Monthly",

    # misc flags
    output_attention=True,
    use_amp=False,
    use_gpu=False,
    use_multi_gpu=False,
    gpu=0,
    device_ids=None,
    num_workers=0,
    batch_size=32,
    checkpoints="./checkpoints/",
    lradj="type2",
    learning_rate=1e-4,
    w_lin=0.01,
    train_epochs=1,
    patience=5,
    loss="MSE",
    p_hidden_dims=[128, 128],
    p_hidden_layers=2,
    anomaly_ratio=0.25,
    mask_rate=0.25,
    des="Exp",
    itr=1,
)

# Match run.py's setting string
args.setting = (
    f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}"
    f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}"
    f"_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_ebtimeF_dt{args.distil}"
    f"_{args.des}_0"
)

# --------- 2) Build model and load checkpoint ----------
exp = Exp_Long_Term_Forecast(args)
model = exp._build_model()  # CPU (use_gpu=False)

ckpt_path = (
    "C:/Users/kapoo/FUSTT/src/checkpoints/"
    "long_term_forecast_DiscoveryPassage_and_others_720_360_96_FUSST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
)
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

# --------- 3) Get test data ----------
_, test_loader = data_provider(args, flag='test')

# Collect a small background set and a test set (X only)
X_list = []
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    X_list.append(batch_x.numpy())  # [B,L,F]
    if len(X_list) >= 10:
        break

X_all = np.concatenate(X_list, axis=0)  # [N,L,F]
X_bg   = X_all[:32] if X_all.shape[0] >= 32 else X_all          # background
X_test = X_all[:64] if X_all.shape[0] >= 64 else X_all          # test subset

# --------- 4) Build batch maker ----------
def build_batch_fn(X_np: np.ndarray):
    n, L, F = X_np.shape
    device = next(model.parameters()).device
    x_enc = torch.tensor(X_np, dtype=torch.float32, device=device)               # [n,L,F]
    x_mark_enc = torch.zeros((n, L, 0), dtype=torch.float32, device=device)      # no time marks
    x_dec = torch.zeros((n, args.pred_len, F), dtype=torch.float32, device=device)
    x_mark_dec = torch.zeros((n, args.pred_len, 0), dtype=torch.float32, device=device)
    return x_enc, x_mark_enc, x_dec, x_mark_dec

# --------- 5) SHAP wrapper (expects flattened inputs) ----------
class PredictWrapper:
    def __init__(self, model, build_batch_fn, target_feature_index: int, L: int, F: int, device="cpu"):
        self.model = model.eval()
        self.build = build_batch_fn
        self.fidx = target_feature_index
        self.L = L
        self.F = F
        self.device = device

    def __call__(self, X_flat):
        X_flat = np.array(X_flat, dtype=np.float32)
        if X_flat.ndim == 1:
            X_flat = X_flat[None, :]
        X = X_flat.reshape(X_flat.shape[0], self.L, self.F)
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.build(X)
        with torch.no_grad():
            y = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [n,P,F] or (y,attn)
            if isinstance(y, tuple):
                y = y[0]
            y = y[..., self.fidx]              # [n,P] (target feature)
            y = y.mean(dim=1, keepdim=True)    # [n,1] (average horizon)
            return y.detach().cpu().numpy()

# figure out the feature index for Chlorophyll
COLS = [
    "Conductivity (S/m)",
    "Density (kg/m3)",
    "Practical Salinity (psu)",
    "Pressure (decibar)",
    "Temperature (C)",
    "Turbidity (NTU)",
    "Chlorophyll (ug/l)",
    "Oxygen Concentration Corrected (ml/l)",
]
CHL_IDX = COLS.index("Chlorophyll (ug/l)")

# ---- flatten windows for SHAP (2-D) ----
L = args.seq_len
F = X_test.shape[-1]
X_bg_flat   = X_bg.reshape(X_bg.shape[0], L * F)
X_test_flat = X_test.reshape(X_test.shape[0], L * F)

f = PredictWrapper(model, build_batch_fn, CHL_IDX, L=L, F=F)

# --------- 6) Run Sampling SHAP (robust for n_features >> n_samples) ----------
nsamples = min(2000, 4 * L * F)  # tune for speed/accuracy
explainer = shap.SamplingExplainer(f, X_bg_flat)
sv = explainer.shap_values(X_test_flat, nsamples=nsamples)

# --- Robust post-processing: handle list / 3D / 2D returns ---
if isinstance(sv, list):
    # If the model returned multiple outputs, SHAP may give a list; take the first.
    assert len(sv) >= 1, "SHAP returned an empty list."
    sv = sv[0]

sv = np.array(sv)
if sv.ndim == 3 and sv.shape[1] == 1:
    # [n,1,L*F] -> [n,L*F]
    sv = sv[:, 0, :]
elif sv.ndim == 2:
    # [n,L*F] -> already fine
    pass
else:
    raise RuntimeError(f"Unexpected SHAP shape {sv.shape}; expected [n,L*F] or [n,1,L*F].")

# Reshape SHAP values back to [n, L, F]
n, LF = sv.shape
assert LF == L * F, f"Unexpected SHAP shape {sv.shape}; expected L*F={L*F}"
sv = sv.reshape(n, L, F)  # [n, L, F]

# Aggregate importances
feat_imp = np.mean(np.abs(sv), axis=(0, 1))  # [F]
lag_imp  = np.mean(np.abs(sv), axis=(0, 2))  # [L]

# --------- 7) Save under resultfile/<SETTING>/shap ----------
out_dir = os.path.join("./resultfile", args.setting, "shap")
os.makedirs(out_dir, exist_ok=True)

np.save(os.path.join(out_dir, "feat_imp_chl.npy"), feat_imp)
np.save(os.path.join(out_dir, "lag_imp_chl.npy"),  lag_imp)

with open(os.path.join(out_dir, "shap_summary.txt"), "w") as ftxt:
    order = np.argsort(feat_imp)[::-1]
    ftxt.write("Feature order: " + ", ".join(COLS) + "\n")
    ftxt.write("Top-5 drivers for Chlorophyll:\n")
    for k in range(min(5, len(order))):
        ftxt.write(f"{k+1}. {COLS[order[k]]}: {feat_imp[order[k]]:.6f}\n")
    ftxt.write("\n")
    top_lags = np.argsort(lag_imp)[::-1][:10]
    ftxt.write("Top-10 lags (time steps in the 96-step lookback):\n")
    ftxt.write(", ".join([f"{int(i)}({lag_imp[i]:.6f})" for i in top_lags]) + "\n")

print("Saved SHAP artifacts under:", out_dir)
