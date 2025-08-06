model_name=DLinear

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path DiscoveryPassage_and_others_merged.csv \
  --model_id DiscoveryPassage_and_others_720_360_96_tn \
  --model $model_name \
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
  --des 'Exp' \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --w_lin 0.01 \
  --target "Chlorophyll (ug/l)" \
  --itr 1

