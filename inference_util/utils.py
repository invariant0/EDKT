import argparse



def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description="DeepGP Training Script")
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")
    parser.add_argument("--dataset", type=str, help="what dataset to use", default='fsmol')
    parser.add_argument("--encode_method", type=str, help="what encoder to use", default='FP')
    parser.add_argument("--num_encoder", type=int, default=2, help="num_encoder")
    parser.add_argument("--allow_NCL", action="store_true", help="whether use NCL")
    
    # First parse to get dataset and encode_method
    temp_args, _ = parser.parse_known_args(args_list)
    
    if temp_args.dataset == 'fsmol':
        parser.add_argument("--FP_input_dim", type=int, default=2024, help="FP input dimension")
        parser.add_argument("--Graph_input_dim", type=int, default=32, help="Graph input dimension")
    elif temp_args.dataset == 'pQSAR':
        parser.add_argument("--FP_input_dim", type=int, default=1024, help="FP input dimension")
        parser.add_argument("--Graph_input_dim", type=int, default=30, help="Graph input dimension")
        parser.add_argument("--group_id", type=int, default=0, help="group_id")
    
    parser.add_argument("--world_size", type=int, default=6, help="number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    if 'FP' in temp_args.encode_method:
        parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    elif 'Graph' in temp_args.encode_method:
        parser.add_argument("--lr", type=float, help="learning rate", default=0.0005)
    
    args = parser.parse_args(args_list)
    
    # Import required modules based on arguments
    if 'FP' in args.encode_method:
        from deep_ensemble import deep_ensemble_FP as deep_ensemble
        if args.dataset == 'fsmol':
            from EDKT_data.Data_FP_fsmol import deep_gp_data
            from DKT_dataset import MLP_train_dataset_fsmol as train_dataset 
            from DKT_dataset import MLP_eval_dataset_fsmol as eval_dataset
        elif args.dataset == 'pQSAR':
            from EDKT_data.Data_FP_pQSAR import deep_gp_data
            from DKT_dataset import MLP_train_dataset_pQSAR as train_dataset
            from DKT_dataset import MLP_eval_dataset_pQSAR as eval_dataset
    elif 'Graph' in args.encode_method:
        from deep_ensemble import deep_ensemble_Graph as deep_ensemble
        if args.dataset == 'fsmol':
            from EDKT_data.Data_Graph_fsmol import deep_gp_data
            from DKT_dataset import Graph_train_dataset_fsmol as train_dataset
            from DKT_dataset import Graph_eval_dataset_fsmol as eval_dataset
        elif args.dataset == 'pQSAR':
            from EDKT_data.Data_Graph_pQSAR import deep_gp_data
            from DKT_dataset import Graph_train_dataset_pQSAR as train_dataset
            from DKT_dataset import Graph_eval_dataset_pQSAR as eval_dataset
    
    # Add imported modules to args
    args.deep_ensemble = deep_ensemble
    args.deep_gp_data = deep_gp_data
    args.train_dataset = train_dataset
    args.eval_dataset = eval_dataset
    
    return args


