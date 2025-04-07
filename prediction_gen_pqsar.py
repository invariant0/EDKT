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
from EDKT_data.Data_FP_pQSAR import deep_gp_data as deep_gp_data_fp
from EDKT_data.Data_Graph_pQSAR import deep_gp_data as deep_gp_data_graph

group_id = 1
assay_id_train_test_split = np.load('../Data_for_publication/pQSAR/pQSAR_split_dic.pkl', allow_pickle=True)
train_assay_ls = list(assay_id_train_test_split[group_id]['train'])
valid_assay_ls = list(assay_id_train_test_split[group_id]['val'])
test_assay_ls = list(assay_id_train_test_split[group_id]['test'])
data_path = '../Data_for_publication/pQSAR/ci9b00375_si_002.txt'
print(len(train_assay_ls), len(valid_assay_ls), len(test_assay_ls))
# FP 
data_fp = deep_gp_data_fp(data_path)
# Graph
# included_assay = np.load('../Data_for_publication/pQSAR/included_assay.pkl', allow_pickle = True)
# train_assay_ls = [x for x in train_assay_ls if x in included_assay]
# test_assay_ls = [x for x in test_assay_ls if x in included_assay]
# print(len(train_assay_ls), len(test_assay_ls))
data_graph = deep_gp_data_graph(data_path)

def get_prediction_dict_per_model(assay_ls, device):
    if not os.path.exists(f'../Result_for_publication/pQSAR/group_{1}'):
        os.makedirs(f'../Result_for_publication/pQSAR/group_{1}')
    
    for model_architecture in ['GraphGCN', 'GraphGAT', 'GraphGIN', 'GraphSAGE']:
        prediction_dic = {}
        for random_seed in range(30):
            model_args = [
                "--num_encoder", "2",
                "--dataset", "pQSAR",
                "--encode_method", model_architecture,
                "--group_id", "1",
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
                    task_data_graph = data_graph.smiles_to_graph_test_all(assay_id, 2000)
                    result_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate(task_data_graph, device)]
                    variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate_var(task_data_graph, device)]
                    prediction_dic[assay_id]['prediction'].extend(result_ls)
                    prediction_dic[assay_id]['variance'].extend(variance_ls)
                    torch.cuda.empty_cache()
        with open(f'../Result_for_publication/pQSAR/group_{args_graph.group_id}/test_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    # FP model prediction, first load then predict
    for model_architecture in ['FPRGB', 'FP']:
        model_args = [
        "--num_encoder", "50",
        "--dataset", "pQSAR",
        "--encode_method", model_architecture,
        "--group_id", "1",
        "--random_seed", "0",
    ]
        args_fp = parse_args(model_args)
        fp_model = load_fp_model(args_fp).to(device)
        fp_model.eval()
        prediction_dic = {}
        label_dic = {}
        with torch.no_grad():
            for assay_id in assay_ls:
                task_data_fp = data_fp.tensorize_test(assay_id, 2000)
                result_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate(task_data_fp, device)]
                variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate_var(task_data_fp, device)]
                prediction_dic[assay_id] = dict()
                prediction_dic[assay_id]['prediction'] = result_ls
                prediction_dic[assay_id]['variance'] = variance_ls
                label_dic[assay_id] = task_data_fp[3].numpy()
                torch.cuda.empty_cache()
        with open(f'../Result_for_publication/pQSAR/group_{args_fp.group_id}/test_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    with open(f'../Result_for_publication/pQSAR/group_{args_fp.group_id}/test_label_dic.pkl', 'wb') as f:
        pickle.dump(label_dic, f)
    return None

def get_prediction_dict_per_model_valid(assay_ls, device, fold_id):

    if not os.path.exists(f'../Result_for_publication/pQSAR/group_{1}_valid/fold_{fold_id}'):
        os.makedirs(f'../Result_for_publication/pQSAR/group_{1}_valid/fold_{fold_id}')
    
    for model_architecture in ['GraphGCN', 'GraphGAT', 'GraphGIN', 'GraphSAGE']:
        prediction_dic = {}
        for random_seed in range(30):
            model_args = [
                "--num_encoder", "2",
                "--dataset", "pQSAR",
                "--encode_method", model_architecture,
                "--group_id", "1",
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
                    task_data_graph = data_graph.smiles_to_graph_valid_all_random(assay_id, 2000, fold_id)
                    result_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate(task_data_graph, device)]
                    variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in graph_model.prediction_seperate_var(task_data_graph, device)]
                    prediction_dic[assay_id]['prediction'].extend(result_ls)
                    prediction_dic[assay_id]['variance'].extend(variance_ls)
                    torch.cuda.empty_cache()
        with open(f'../Result_for_publication/pQSAR/group_{args_graph.group_id}_valid/fold_{fold_id}/valid_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    # FP model prediction, first load then predict
    for model_architecture in ['FPRGB', 'FP']:
        model_args = [
        "--num_encoder", "50",
        "--dataset", "pQSAR",
        "--encode_method", model_architecture,
        "--group_id", "1",
        "--random_seed", "0",
    ]
        args_fp = parse_args(model_args)
        fp_model = load_fp_model(args_fp).to(device)
        fp_model.eval()
        prediction_dic = {}
        label_dic = {}
        with torch.no_grad():
            for assay_id in assay_ls:
                task_data_fp = data_fp.tensorize_valid_random(assay_id, 2000, fold_id)
                result_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate(task_data_fp, device)]
                variance_ls = [x.detach().cpu().numpy().reshape(-1) for x in fp_model.prediction_seperate_var(task_data_fp, device)]
                prediction_dic[assay_id] = dict()
                prediction_dic[assay_id]['prediction'] = result_ls
                prediction_dic[assay_id]['variance'] = variance_ls
                label_dic[assay_id] = task_data_fp[3].numpy()
                torch.cuda.empty_cache()
        with open(f'../Result_for_publication/pQSAR/group_{args_fp.group_id}_valid/fold_{fold_id}/valid_{model_architecture}_prediction_dic.pkl', 'wb') as f:
            pickle.dump(prediction_dic, f)
    with open(f'../Result_for_publication/pQSAR/group_{args_fp.group_id}_valid/fold_{fold_id}/valid_label_dic.pkl', 'wb') as f:
        pickle.dump(label_dic, f)
    return None

device = 'cuda:0'
get_prediction_dict_per_model(test_assay_ls, device)
for i in range(1,31): 
    get_prediction_dict_per_model_valid(valid_assay_ls, device, i)