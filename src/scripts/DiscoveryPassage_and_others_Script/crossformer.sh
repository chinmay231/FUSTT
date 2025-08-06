model_name=Crossformer

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path DiscoveryPassage_and_others_merged.csv \
  --model_id DiscoveryPassage_and_others_720_360_96cf \
  --model $model_name \
  --lradj 'type1' \
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
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --batch_size 16  \
  --learning_rate '0.0001' \
  --w_lin 0.01 \
  --itr 1 \
  --target "Chlorophyll (ug/l)"