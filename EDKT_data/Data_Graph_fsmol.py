import numpy as np  
import pandas as pd 
import torch
import random
from torch_geometric.data import Batch, Data
from statistics import mode
import pickle 
from data_classes import experiment, experiment_test, assay, total_assays

def split_list(lst, ratio=0.5):
    # Shuffle the list in place
    random.shuffle(lst)
    
    # Calculate split point
    split_point = int(len(lst) * ratio)
    
    # Split the list
    return lst[:split_point], lst[split_point:]

class deep_gp_data:

    def __init__(self, file_path, file_path_test):
        with open(file_path, 'rb') as f:
            self.all_data = pickle.load(f)
        with open(file_path_test, 'rb') as f:
            self.all_data_test = pickle.load(f)
    
    def smiles_to_graph(self, assay_id, sample_num, r_seed):
        tuple_ls = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            graph_temp = experiment.fp1
            y = torch.tensor(experiment.expt_pIC50).float()
            tuple_ls.append((graph_temp, y))
        
        support_ls, query_ls = split_list(tuple_ls)
        
        if len(support_ls) > int(sample_num/2):
            random.seed(r_seed)
            support_idx = random.sample([i for i in range(len(support_ls))], int(sample_num/2)) 
            query_idx = random.sample([i for i in range(len(query_ls))], min(len(query_ls), int(sample_num/2))) 
        else:
            support_idx = [i for i in range(len(support_ls))]
            query_idx = [i for i in range(len(query_ls))]

        graph_data_support_return = []
        label_data_support_return = []
        graph_data_query_return = []
        label_data_query_return = []
        for idx in support_idx:
            graph_data_support_return.append(support_ls[idx][0])
            label_data_support_return.append(support_ls[idx][1])
        for idx in query_idx:
            graph_data_query_return.append(query_ls[idx][0])
            label_data_query_return.append(query_ls[idx][1])
        return graph_data_support_return, label_data_support_return, graph_data_query_return, label_data_query_return

    def smiles_to_graph_test(self, assay_id, fold_id):
        tuple_ls = []
        for experiment in self.all_data_test[fold_id].assay_dic[assay_id].experiments:
            graph_temp = experiment.fp1
            y = torch.tensor(experiment.expt_pIC50).float()
            tuple_ls.append((graph_temp, y, experiment.test_flag_fold))
        
        support_ls = [tuple_item for tuple_item in tuple_ls if tuple_item[2] == 'Train']
        query_ls = [tuple_item for tuple_item in tuple_ls if tuple_item[2] == 'Test']

        graph_data_support_return = [support_tuple[0] for support_tuple in support_ls]
        label_data_support_return = [support_tuple[1] for support_tuple in support_ls]
        graph_data_query_return = [query_tuple[0] for query_tuple in query_ls]
        label_data_query_return = [query_tuple[1] for query_tuple in query_ls]
        
        return graph_data_support_return, label_data_support_return, graph_data_query_return, label_data_query_return