#!/usr/bin/env bash
set -euo pipefail

# Absolute paths (Windows Git Bash friendly)
PY="C:/Users/kapoo/FUSTT/.venv/Scripts/python.exe"
RUNPY="C:/Users/kapoo/FUSTT/src/run.py"
DATA_DIR="C:/Users/kapoo/FUSTT/src/dataset"
DATA_FILE="Data_run.csv"

# A single canonical setting name (used by resultfile/ and SHAP later)
SETTING="DiscoveryPassage_and_others_720_360_96"

echo "Using Python: $PY"
"$PY" --version || { echo "Python not found"; exit 1; }

# Verify dataset exists
if [ ! -f "$DATA_DIR/$DATA_FILE" ]; then
  echo "ERROR: dataset file not found at: $DATA_DIR/$DATA_FILE"
  ls -la "$DATA_DIR" || true
  exit 1
fi

# Export module path for imports + setting for SHAP tool
export PYTHONPATH="C:/Users/kapoo/FUSTT/src"
export FUSST_SETTING="$SETTING"

echo "Starting FUSST training..."
"$PY" -u "$RUNPY" \
  --activation gelu \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "$DATA_DIR" \
  --data_path "$DATA_FILE" \
  --model_id "$SETTING" \
  --model FUSST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des Exp \
  --learning_rate 0.0001 \
  --lradj type2 \
  --w_lin 0.01 \
  --itr 1 \
  --target "Chlorophyll (ug/l)" \
  --output_attention \
  --mc_passes 50 \
  --pi_alpha 0.05

echo ">>> Training + test complete!"
echo "Artifacts in: ./resultfile/$SETTING/"
