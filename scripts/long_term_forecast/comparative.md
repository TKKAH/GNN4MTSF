# Comparative Experiment
First,cd to the Long_term_forecast dir:
```
cd Long_term_forecast
```
## ETTh1
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data ETTh1 --root_path dataset/ETT-small --data_path ETTh1.csv --graph_path None --freq h --seq_len 96 --pred_len 96 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data ETTh1 --root_path dataset/ETT-small --data_path ETTh1.csv --graph_path None --freq h --seq_len 96 --pred_len 192 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data ETTh1 --root_path dataset/ETT-small --data_path ETTh1.csv --graph_path None --freq h --seq_len 96 --pred_len 336 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data ETTh1 --root_path dataset/ETT-small --data_path ETTh1.csv --graph_path None --freq h --seq_len 96 --pred_len 720 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
## ETTh2
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data ETTh2 --root_path dataset/ETT-small --data_path ETTh2.csv --graph_path None --freq h --seq_len 96 --pred_len 96 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data ETTh2 --root_path dataset/ETT-small --data_path ETTh2.csv --graph_path None --freq h --seq_len 96 --pred_len 192 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data ETTh2 --root_path dataset/ETT-small --data_path ETTh2.csv --graph_path None --freq h --seq_len 96 --pred_len 336 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data ETTh2 --root_path dataset/ETT-small --data_path ETTh2.csv --graph_path None --freq h --seq_len 96 --pred_len 720 --num_nodes 7 --input_dim 1 --split_type time --train_ratio 0.6
```
## ETTm1
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data ETTm1 --root_path dataset/ETT-small --data_path ETTm1.csv --graph_path None --freq t --seq_len 96 --pred_len 96 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data ETTm1 --root_path dataset/ETT-small --data_path ETTm1.csv --graph_path None --freq t --seq_len 96 --pred_len 192 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data ETTm1 --root_path dataset/ETT-small --data_path ETTm1.csv --graph_path None --freq t --seq_len 96 --pred_len 336 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data ETTm1 --root_path dataset/ETT-small --data_path ETTm1.csv --graph_path None --freq t --seq_len 96 --pred_len 720 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
## ETTm2
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data ETTm2 --root_path dataset/ETT-small --data_path ETTm2.csv --graph_path None --freq t --seq_len 96 --pred_len 96 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data ETTm2 --root_path dataset/ETT-small --data_path ETTm2.csv --graph_path None --freq t --seq_len 96 --pred_len 192 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data ETTm2 --root_path dataset/ETT-small --data_path ETTm2.csv --graph_path None --freq t --seq_len 96 --pred_len 336 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data ETTm2 --root_path dataset/ETT-small --data_path ETTm2.csv --graph_path None --freq t --seq_len 96 --pred_len 720 --num_nodes 7 --input_dim 1 --train_ratio 0.6
```
## Weather
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data weather --root_path dataset\weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 96 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 192 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 336 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 720 --num_nodes 21 --input_dim 1 
```
## Exchange_rate
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_96 --model HHAGCRN --data exchange_rate --root_path dataset\exchange_rate --data_path exchange_rate.csv --graph_path None --freq d --seq_len 96 --pred_len 96 --num_nodes 8 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_192 --model HHAGCRN --data exchange_rate --root_path dataset/exchange_rate --data_path exchange_rate.csv --graph_path None --freq d --seq_len 96 --pred_len 192 --num_nodes 8 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_336 --model HHAGCRN --data exchange_rate --root_path dataset/exchange_rate --data_path exchange_rate.csv --graph_path None --freq d --seq_len 96 --pred_len 336 --num_nodes 8 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRN_96_720 --model HHAGCRN --data exchange_rate --root_path dataset/exchange_rate --data_path exchange_rate.csv --graph_path None --freq d --seq_len 96 --pred_len 720 --num_nodes 8 --input_dim 1 
```