import torch 
import argparse
import importlib
from deep_ensemble import deep_ensemble_FP as deep_ensemble_MLP
from deep_ensemble import deep_ensemble_Graph as deep_ensemble_Graph
from collections import OrderedDict
importlib.reload(deep_ensemble_MLP)
importlib.reload(deep_ensemble_Graph)

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


def load_fp_model(model_args):
    try:
        fp_model = deep_ensemble_MLP.ensemble_deep_gp(model_args)
        # Load state dict
        if model_args.dataset == 'fsmol':
            model_path = f'../Model_for_publication/Dataset:{model_args.dataset}_Method:{model_args.encode_method}_Num:{model_args.num_encoder}_NCL:{model_args.allow_NCL}_seed:{model_args.random_seed}.pth'
        elif model_args.dataset == 'pQSAR':
            model_path = f'../Model_for_publication/Dataset:{model_args.dataset}_Groupid:{model_args.group_id}_Method:{model_args.encode_method}_Num:{model_args.num_encoder}_NCL:{model_args.allow_NCL}_seed:{model_args.random_seed}.pth'
        state_dict = torch.load(model_path)
        # Process state dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        # Load and verify
        fp_model.load_state_dict(new_state_dict)
        fp_model.eval()
        print(f"✓ FP Model loaded successfully from {model_path}")
        return fp_model
        
    except Exception as e:
        print(f"✗ Failed to load FP model: {str(e)}")
        return None

def load_graph_model(model_args):
    try:
        graph_model = deep_ensemble_Graph.ensemble_deep_gp(model_args)
        
        # Load state dict
        if model_args.dataset == 'fsmol':
            model_path = f'../Model_for_publication/Dataset:{model_args.dataset}_Method:{model_args.encode_method}_Num:{model_args.num_encoder}_NCL:{model_args.allow_NCL}_seed:{model_args.random_seed}.pth'
        elif model_args.dataset == 'pQSAR':
            model_path = f'../Model_for_publication/Dataset:{model_args.dataset}_Groupid:{model_args.group_id}_Method:{model_args.encode_method}_Num:{model_args.num_encoder}_NCL:{model_args.allow_NCL}_seed:{model_args.random_seed}.pth'

        state_dict = torch.load(model_path)
        
        # Process state dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # Load and verify
        graph_model.load_state_dict(new_state_dict)
        graph_model.eval()
        
        print(f"✓ Model loaded successfully from {model_path}")
        return graph_model
        
    except Exception as e:
        # print(f"✗ Failed to load model: {str(e)}")
        return None
    