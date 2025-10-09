# src/tools/plot_mcd.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider
from utils.mc_dropout import mc_predict  # uses your existing MC utility

# ----------------------------- 0) ARGS (match training) -----------------------------
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
    embed="timeF",
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

# Match run.py's "setting" string
args.setting = (
    f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}"
    f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}"
    f"_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_ebtimeF_dt{args.distil}"
    f"_{args.des}_0"
)

# ----------------------------- 1) Build model & load checkpoint -----------------------------
print("Use CPU")
exp = Exp_Long_Term_Forecast(args)
model = exp._build_model()  # CPU (use_gpu=False)

ckpt_path = (
    "C:/Users/kapoo/FUSTT/src/checkpoints/"
    "long_term_forecast_DiscoveryPassage_and_others_720_360_96_FUSST_custom_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
)
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
state = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state)
model.eval()  # mc_predict will toggle dropout when needed

# ----------------------------- 2) Data -----------------------------
_, test_loader = data_provider(args, flag='test')
print("test", len(test_loader.dataset))

# Feature index list (same order used earlier)
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

# ----------------------------- 3) Helper: MC on one batch -----------------------------
def do_mc(batch_x, batch_y, batch_x_mark, batch_y_mark, T=50):
    """
    Returns mean, std of predictions over T MC passes.
    Shapes:
      mean, std: [B, P, F]
    """
    device = next(model.parameters()).device

    # Ensure dtype float32 + correct device for ALL tensors
    batch_x = batch_x.to(device=device, dtype=torch.float32)
    batch_x_mark = batch_x_mark.to(device=device, dtype=torch.float32)
    batch_y_mark = batch_y_mark.to(device=device, dtype=torch.float32)

    # Decoder input: zeros for the prediction horizon
    dec_inp = torch.zeros(
        (batch_y.shape[0], args.pred_len, batch_x.shape[-1]),
        dtype=torch.float32,
        device=device,
    )

    # Run MC Dropout
    with torch.no_grad():
        mean, std = mc_predict(model, (batch_x, batch_x_mark, dec_inp, batch_y_mark), T=T)
        # mean,std -> torch tensors [B,P,F]
        return mean.cpu().numpy(), std.cpu().numpy()

# ----------------------------- 4) Run MC on first test batch -----------------------------
batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))  # [B,L,F], [B,P,F], ...
MC_PASSES = int(os.environ.get("MC_PASSES", "50"))
mean, std = do_mc(batch_x, batch_y, batch_x_mark, batch_y_mark, T=MC_PASSES)  # [B,P,F]

y_true = batch_y.numpy()  # [B,P,F]

# ----------------------------- 5) Metrics: PICP/MPIW for 95% CI -----------------------------
alpha = float(os.environ.get("ALPHA", "0.05"))  # 0.05 -> 95%
z = 1.959963984540054  # ~N(0,1) 97.5% quantile

lower = mean - z * std   # [B,P,F]
upper = mean + z * std

inside = (y_true >= lower) & (y_true <= upper)
PICP = inside.mean().item()

MPIW = (upper - lower).mean().item()  # average interval width across all B,P,F

# Also compute deterministic metrics on the mean prediction for context (Chlorophyll only)
pred_chl = mean[..., CHL_IDX]
true_chl = y_true[..., CHL_IDX]
MSE = np.mean((pred_chl - true_chl) ** 2)
MAE = np.mean(np.abs(pred_chl - true_chl))
RMSE = float(np.sqrt(MSE))

# ----------------------------- 6) Outputs folder -----------------------------
out_dir = os.path.join("./resultfile", args.setting, "mcd")
os.makedirs(out_dir, exist_ok=True)

# ----------------------------- 7) Figure 1: Forecast with 95% band (Chlorophyll) -----------------------------
b = 0  # first sample in the batch
t = np.arange(args.pred_len)

plt.figure(figsize=(10, 6))
plt.plot(t, true_chl[b], label="True")
plt.plot(t, pred_chl[b], label="MC Mean")
plt.fill_between(t, (pred_chl[b] - z * std[b, :, CHL_IDX]),
                    (pred_chl[b] + z * std[b, :, CHL_IDX]),
                 alpha=0.3, label="95% band")
plt.title("Chlorophyll Forecast with 95% MC Dropout Interval")
plt.xlabel("Forecast step")
plt.ylabel("Scaled value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mcd_forecast_chl.png"), dpi=200)
plt.close()

# ----------------------------- 8) Figure 2: Z-score histogram (all B,P,F) -----------------------------
z_all = (y_true - mean) / (std + 1e-8)
z_flat = z_all.flatten()

plt.figure(figsize=(10, 6))
plt.hist(z_flat, bins=40, density=True)
plt.title("Standardized Residuals (z = (y - mean) / std)")
plt.xlabel("z")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mcd_uncert_hist.png"), dpi=200)
plt.close()

# ----------------------------- 9) Figure 3: Coverage vs. nominal (simple calibration curve) -----------------------------
# For nominal coverages in [50%, 60%, ..., 95%], compute empirical coverage using |z| <= z_nom
nom_levels = np.arange(50, 96, 5)  # 50..95
emp_cov = []
from scipy.stats import norm  # available by default with SciPy if installed; if not, fallback below

try:
    z_levels = norm.ppf(0.5 + nom_levels / 200.0)  # two-sided
except Exception:
    # fallback approximate z's
    approx = {50:0.674,55:0.76,60:0.842,65:0.934,70:1.036,75:1.150,80:1.282,85:1.440,90:1.645,95:1.960}
    z_levels = np.array([approx[n] for n in nom_levels])

abs_z = np.abs(z_flat)
for zq in z_levels:
    emp_cov.append((abs_z <= zq).mean())

plt.figure(figsize=(8, 6))
plt.plot(nom_levels/100.0, nom_levels/100.0, linestyle="--", label="Ideal")
plt.plot(nom_levels/100.0, emp_cov, marker="o", label="Empirical")
plt.xlabel("Nominal coverage")
plt.ylabel("Empirical coverage")
plt.title("MC Dropout Calibration (All Features)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "mcd_calibration.png"), dpi=200)
plt.close()

# ----------------------------- 10) Save summary -----------------------------
with open(os.path.join(out_dir, "mcd_summary.txt"), "w") as f:
    f.write(f"SETTING: {args.setting}\n")
    f.write(f"MSE: {MSE:.6f}, MAE: {MAE:.6f}, RMSE: {RMSE:.6f}\n")
    f.write(f"MC_PASSES: {MC_PASSES}, ALPHA: {alpha}\n")
    f.write(f"PICP: {PICP:.6f}\n")
    f.write(f"MPIW: {MPIW:.6f}\n")

print("Saved MCD artifacts under:", out_dir)
print(f"MSE: {MSE:.6f}, MAE: {MAE:.6f}, RMSE: {RMSE:.6f}")
print(f"PICP@{int((1-alpha)*100)}%: {PICP:.6f}  |  MPIW: {MPIW:.6f}")
