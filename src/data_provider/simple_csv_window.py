import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

COLS = [
    "Conductivity (S/m)","Density (kg/m3)","Practical Salinity (psu)","Pressure (decibar)",
    "Temperature (C)","Turbidity (NTU)","Chlorophyll (ug/l)","Oxygen Concentration Corrected (ml/l)"
]

def load_csv_windows(path_csv, seq_len, pred_len, stride=1):
    df = pd.read_csv(path_csv)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    X = df[COLS].values.astype(np.float32)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    L, P = seq_len, pred_len
    T, F = Xn.shape
    xs, ys = [], []
    for s in range(0, T-L-P+1, stride):
        xs.append(Xn[s:s+L, :])
        ys.append(Xn[s+L:s+L+P, :])
    return np.stack(xs,0), np.stack(ys,0), scaler  # [N,L,F], [N,P,F]
