# MTGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MTGNN_31_31 --model MTGNN --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 
# AGCRN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id AGCRN_31_31 --model AGCRN --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 
# FCSTGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id FCSTGNN_31_31 --model FCSTGNN --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 
# HiPPOAGCRN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id HiPPOAGCRN_31_31 --model HiPPOAGCRN --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 
# GTS
python -u run.py --task_name long_term_forecast --is_training 1 --model_id GTS_31_31 --model GTS --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 --loss_with_regularization

python -u run.py --task_name long_term_forecast --is_training 1 --model_id GTS_31_31 --model GTS --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4 
# MTGAT
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MTGAT_31_31 --model MTGAT --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 1
# MSGNet
python -u run.py --task_name long_term_forecast --is_training 1 --model_id MSGNet_31_31 --model MSGNet --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 1
# CrossGNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id CrossGNN_31_31 --model CrossGNN --data HKda --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path None --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 1
# DCRNN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id DCRNN_31_31 --model DCRNN --data HKda --predefined_graph --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path HK_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4
# STSGCN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id STSGCN_31_31 --model STSGCN --data HKda --predefined_graph --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path HK_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4
# ASTGCN
python -u run.py --task_name long_term_forecast --is_training 1 --model_id ASTGCN_31_31 --model ASTGCN --data HKda --predefined_graph --root_path dataset\Exit-and-entry --data_path HK_Daily_Arrivals.npz --graph_path HK_stations_graph.pkl --freq d --seq_len 31 --pred_len 31 --num_nodes 14 --input_dim 4