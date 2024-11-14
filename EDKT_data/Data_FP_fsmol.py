import numpy as np  
import pandas as pd 
import torch
import random
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
    
    def tensorize(self, assay_id, sample_num, r_seed):
        tuple_ls = []
        x_tensor_query = []
        y_tensor_query = []
        x_tensor_support = []
        y_tensor_support = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            x = experiment.cpd_id
            y = torch.tensor(experiment.expt_pIC50)
            tuple_ls.append((x, y))
        
        support_ls, query_ls = split_list(tuple_ls)
        if len(support_ls) > int(sample_num/2):
            random.seed(r_seed)
            support_idx = random.sample([i for i in range(len(support_ls))], int(sample_num/2)) 
            query_idx = random.sample([i for i in range(len(query_ls))], min(len(query_ls), int(sample_num/2))) 
        else:
            support_idx = [i for i in range(len(support_ls))]
            query_idx = [i for i in range(len(query_ls))]
        x_tensor_query = np.array([query_ls[i][0] for i in query_idx])
        y_tensor_query = np.array([query_ls[i][1] for i in query_idx])
        x_tensor_support = np.array([support_ls[i][0] for i in support_idx])
        y_tensor_support = np.array([support_ls[i][1] for i in support_idx])
        return torch.tensor(x_tensor_support).float(), torch.tensor(y_tensor_support).float(), torch.tensor(x_tensor_query).float(), torch.tensor(y_tensor_query).float()
    def tensorize_test(self, assay_id, fold_id):
        
        x_tensor_query = []
        y_tensor_query = []
        x_tensor_support = []
        y_tensor_support = []
        tuple_ls = []
        for experiment in self.all_data_test[fold_id].assay_dic[assay_id].experiments:
            x = experiment.cpd_id
            y = torch.tensor(experiment.expt_pIC50)
            tuple_ls.append((x, y, experiment.test_flag_fold))
        
        support_ls = [tuple_item for tuple_item in tuple_ls if tuple_item[2] == 'Train']
        query_ls = [tuple_item for tuple_item in tuple_ls if tuple_item[2] == 'Test']
        
        x_tensor_query = np.array([query_tuple[0] for query_tuple in query_ls])
        y_tensor_query = np.array([query_tuple[1] for query_tuple in query_ls])
        x_tensor_support = np.array([support_tuple[0] for support_tuple in support_ls])
        y_tensor_support = np.array([support_tuple[1] for support_tuple in support_ls])
        return torch.tensor(x_tensor_support).float(), torch.tensor(y_tensor_support).float(), torch.tensor(x_tensor_query).float(), torch.tensor(y_tensor_query).float()