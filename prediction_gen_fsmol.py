import torch 
import numpy as np 
import math 
import sys
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pickle 
import os
from inference_util.utils import parse_args 
from inference_util.utils import load_fp_model, load_graph_model
from EDKT_data.Data_FP_fsmol import deep_gp_data as deep_gp_data_fp
from EDKT_data.Data_Graph_fsmol import deep_gp_data as deep_gp_data_graph

def get_prediction_dict_per_model(assay_ls, device, fold_id, mode = 'valid'):
    data_test_graph = deep_gp_data_graph(data_path, data_path_test)
    data_valid_graph = deep_gp_data_graph(data_path, data_path_valid)
    for model_architecture in ['GraphGCN', 'GraphSAGE', 'GraphGIN', 'GraphGAT']:
        prediction_dic = {}
        for random_seed in range(60):
            model_args = [
                "--num_encoder", "2",
                "--dataset", "fsmol",
                "--encode_method", model_architecture,
                "--random_seed", str(random_seed),
            ]
            args_graph = parse_args(model_args)
            graph_model = load_graph_model(args_graph)
            if graph_model is None:
                continue
            graph_model.eval()
            with torch.no_grad():
                graph_model.to(device)
                for assay_id in assay_ls:
                    if assay_id not in prediction_dic:
                        prediction_dic[assay_id] = dict()
                        prediction_dic[assay_id]['prediction'] = []
                        prediction_dic[assay_id]['variance'] = []
                    if mode == 'test':
                        # task_data_fp = data_test_fp.tensorize_test(assay_id, fold_id)
                        task_data_graph = data_test_graph.smiles_to_graph_test(assay_id, fold_id)
                    elif mode == 'valid':
                        # task_data_fp = data_valid_fp.tensorize_test(assay_id, fold_id)
                        task_data_graph = data_valid_graph.smiles_to_graph_test(assay_id, fold_id)
                    result_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate(task_data_graph, device)]
                    variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate_var(task_data_graph, device)]
                    prediction_dic[assay_id]['prediction'].extend(result_ls)
                    prediction_dic[assay_id]['variance'].extend(variance_ls)
                    torch.cuda.empty_cache()
        if not os.path.exists(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}'):
            os.makedirs(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}')
        with open(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/{mode}_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    model_architectures = ['FPRGB', 'FPaugmentRGB']
    model_args = [
        "--encode_method", "FP",
    ]
    args_fp = parse_args(model_args)
    data_test_fp = deep_gp_data_fp(data_path_fp, data_path_fp_test, args_fp)
    data_valid_fp = deep_gp_data_fp(data_path_fp, data_path_fp_valid, args_fp)
    for model_architecture in model_architectures:
        prediction_dic = {}
        label_dic = {}
        for random_seed in range(60):
            model_args = [
            "--num_encoder", "2",
            "--dataset", "fsmol",
            "--encode_method", model_architecture,
            "--random_seed", str(random_seed),
        ]
            args_fp = parse_args(model_args)
            fp_model = load_fp_model(args_fp)
            if fp_model is None:
                continue
            fp_model.eval()
            with torch.no_grad():
                fp_model.to(device)
                for assay_id in assay_ls:
                    if assay_id not in prediction_dic:
                        prediction_dic[assay_id] = dict()
                        prediction_dic[assay_id]['prediction'] = []
                        prediction_dic[assay_id]['variance'] = []
                    if mode == 'test':
                        task_data_fp = data_test_fp.tensorize_test(assay_id, fold_id)
                    elif mode == 'valid':
                        task_data_fp = data_valid_fp.tensorize_test(assay_id, fold_id)
                    result_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate(task_data_fp, device)]
                    variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate_var(task_data_fp, device)]
                    prediction_dic[assay_id]['prediction'].extend(result_ls)
                    prediction_dic[assay_id]['variance'].extend(variance_ls)
                    label_dic[assay_id] = task_data_fp[3].numpy()
                    torch.cuda.empty_cache()
            if not os.path.exists(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/'):
            # Create the folder (and intermediate directories, if necessary)
                os.makedirs(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/')
            with open(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/{mode}_{model_architecture}_prediction_dic.pkl', 'wb') as f:
                pickle.dump(prediction_dic, f)
    model_architectures = ['FPaugment', 'FP']
    model_args = [
        "--encode_method", "FPaugment",
    ]
    args_fp = parse_args(model_args)
    data_test_fp = deep_gp_data_fp(data_path_fp, data_path_fp_test, args_fp)
    data_valid_fp = deep_gp_data_fp(data_path_fp, data_path_fp_valid, args_fp)
    for model_architecture in model_architectures:
        prediction_dic = {}
        label_dic = {}
        model_args = [
        "--num_encoder", "50",
        "--dataset", "fsmol",
        "--encode_method", model_architecture,
        "--random_seed", str(0),
    ]
        args_fp = parse_args(model_args)
        fp_model = load_fp_model(args_fp)
        if fp_model is None:
            continue
        fp_model.eval()
        with torch.no_grad():
            fp_model.to(device)
            for assay_id in assay_ls:
                if mode == 'test':
                    # data_test_fp = deep_gp_data_fp(data_path_fp, data_path_fp_test, args_fp)
                    task_data_fp = data_test_fp.tensorize_test(assay_id, fold_id)
                elif mode == 'valid':
                    # data_valid_fp = deep_gp_data_fp(data_path_fp, data_path_fp_valid, args_fp)
                    task_data_fp = data_valid_fp.tensorize_test(assay_id, fold_id)
                result_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate(task_data_fp, device)]
                variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate_var(task_data_fp, device)]
                prediction_dic[assay_id] = dict()
                prediction_dic[assay_id]['prediction'] = result_ls
                prediction_dic[assay_id]['variance'] = variance_ls
                label_dic[assay_id] = task_data_fp[3].numpy()
                torch.cuda.empty_cache()
        if not os.path.exists(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/'):
            # Create the folder (and intermediate directories, if necessary)
            os.makedirs(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/')
        with open(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/{mode}_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    with open(f'../Result_for_publication/fsmol/{few_shot_number}/Fold_{fold_id}/{mode}_label_dic.pkl', 'wb') as f:
        pickle.dump(label_dic, f)
    return None
for few_shot_number in [16, 32, 64, 128]:
    assay_id_train_test_split = np.load('../Data_for_publication/fsmol_multifp/split_dic.pkl', allow_pickle=True)
    test_assay_ls = assay_id_train_test_split['test_assays']
    eval_assay_ls = assay_id_train_test_split['valid_assays']

    # load fp data
    data_path_fp = '../Data_for_publication/fsmol_multifp/all_data_fp.pkl'
    data_path_fp_test = f'../Data_for_publication/fsmol_multifp/all_data_fp_test_10fold_{few_shot_number}.pkl'
    data_path_fp_valid = f'../Data_for_publication/fsmol_multifp/all_data_fp_valid_10fold_{few_shot_number}.pkl'

    # load graph data
    data_path = '../Data_for_publication/fsmol_multifp/all_data_graph.pkl'
    data_path_test = f'../Data_for_publication/fsmol_multifp/all_data_graph_test_10fold_{few_shot_number}.pkl'
    data_path_valid = f'../Data_for_publication/fsmol_multifp/all_data_graph_valid_10fold_{few_shot_number}.pkl'
    device = 'cuda:0'
    print(few_shot_number)
    for fold_id in range(10):
        get_prediction_dict_per_model(test_assay_ls, device, fold_id, 'test')
        get_prediction_dict_per_model(eval_assay_ls, device, fold_id, 'valid')
