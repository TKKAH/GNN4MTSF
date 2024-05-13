# Ablation Study
First,cd to the Long_term_forecast dir:
```
cd Long_term_forecast
```
## Myen
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
## without AGL
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutAGL_96_96 --model HHAGCRNwithoutAGL --data weather --root_path dataset\weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 96 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutAGL_96_192 --model HHAGCRNwithoutAGL --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 192 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutAGL_96_336 --model HHAGCRNwithoutAGL --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 336 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutAGL_96_720 --model HHAGCRNwithoutAGL --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 720 --num_nodes 21 --input_dim 1 
```
## without HiPPO
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutHiPPO_96_96 --model HHAGCRNwithoutHiPPO --data weather --root_path dataset\weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 96 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutHiPPO_96_192 --model HHAGCRNwithoutHiPPO --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 192 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutHiPPO_96_336 --model HHAGCRNwithoutHiPPO --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 336 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutHiPPO_96_720 --model HHAGCRNwithoutHiPPO --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 720 --num_nodes 21 --input_dim 1 
```
## without NPW
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutNPW_96_96 --model HHAGCRNwithoutNPW --data weather --root_path dataset\weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 96 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutNPW_96_192 --model HHAGCRNwithoutNPW --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 192 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutNPW_96_336 --model HHAGCRNwithoutNPW --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 336 --num_nodes 21 --input_dim 1 
```
```
python -u run.py --task_name long_term_forecast --is_training 0 --model_id HHAGCRNwithoutNPW_96_720 --model HHAGCRNwithoutNPW --data weather --root_path dataset/weather --data_path weather.csv --graph_path None --freq d --seq_len 96 --pred_len 720 --num_nodes 21 --input_dim 1 
```
