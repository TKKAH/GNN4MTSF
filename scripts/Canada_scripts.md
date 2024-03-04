# AGCRN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id AGCRN_31_31 --model AGCRN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3 
# MTGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MTGNN_31_31 --model MTGNN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3 
# MSGNet
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MSGNet_31_31 --model MSGNet --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 1 --MSGNet_enc_dim 300
# HiPPOAGCRN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id HiPPOAGCRN_31_31 --model HiPPOAGCRN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3
# CrossGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id CrossGNN_31_31 --model CrossGNN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 1
# FCSTGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id FCSTGNN_31_31 --model FCSTGNN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3
# MTGAT
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MTGAT_31_31 --model MTGAT --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 1
# GTS
python -u run.py --task_name long_term_forecast --is_training 1 --model_id GTS_31_31 --model GTS --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3
# DCRNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id DCRNN_31_31 --model DCRNN --data Canada --predefined_graph --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path Canada_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3
# STSGCN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id STSGCN_31_31 --model STSGCN --data Canada --predefined_graph --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path Canada_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3 --STSGCN_hidden_dims "[[64], [64]]" --STSGCN_first_layer_embedding_size 1
# ASTGCN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id ASTGCN_31_31 --model ASTGCN --data Canada --predefined_graph --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path Canada_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3
# HHAGCRN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id HHAGCRN_31_31 --model HHAGCRN --data Canada --root_path dataset\Exit-and-entry --data_path Canada_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 260 --input_dim 3