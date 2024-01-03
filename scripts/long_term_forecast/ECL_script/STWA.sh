export CUDA_VISIBLE_DEVICES=1

model_name=STWA

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --num_nodes 321 \
  --subgraph_size 20 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --num_nodes 321 \
  --subgraph_size 20 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --num_nodes 321 \
  --subgraph_size 20 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --num_nodes 321 \
  --subgraph_size 20 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1