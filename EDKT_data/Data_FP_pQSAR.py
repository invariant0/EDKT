import numpy as np  
import pandas as pd 
import torch
import random
from torch_geometric.data import Batch, Data
from statistics import mode

class experiment:
    def __init__(self, assay_id, cpd_id, expt_pIC50, Clustering):
        self.assay_id = assay_id
        self.cpd_id = cpd_id
        self.expt_pIC50 = expt_pIC50
        self.Clustering = Clustering

class assay:
    def __init__(self, assay_id):
        self.assay_id = assay_id
        self.experiments = []

    def add_experiment(self, exp):
        self.experiments.append(exp)

class total_assays:

    def __init__(self):
        self.assay_dic = dict()

    def add_assay(self, assay_id):
        self.assay_dic[assay_id] = assay(assay_id)

class deep_gp_data:

    def __init__(self, file_path):
        self.all_data = self.get_data(file_path)
        self.fp_dic = np.load('../Data_for_publication/pQSAR/compound_fp.pickle', allow_pickle = True)
        self.small_cpd_id = np.load('../Data_for_publication/pQSAR/small_cpd_id.pkl', allow_pickle=True)
        
    def get_data(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
            all_data = total_assays()
            for line in lines:
                line = line.strip().split(' ')
                if 'CHEMBL' not in line[0]:
                    continue
                assay_id = int(line[1])
                if assay_id not in all_data.assay_dic:
                    all_data.add_assay(assay_id)
                all_data.assay_dic[assay_id].add_experiment(experiment(int(line[1]), line[0], float(line[2].strip('‐')), line[4]))
        return all_data

    def tensorize(self, assay_id, sample_num, r_seed):
        tuple_ls = []
        x_tensor_query = []
        y_tensor_query = []
        x_tensor_support = []
        y_tensor_support = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            x = self.fp_dic[cpd_id]
            y = torch.tensor(experiment.expt_pIC50)
            tuple_ls.append((x, y, experiment.Clustering))
        support_ls = [temp_tuple for temp_tuple in tuple_ls if temp_tuple[2] == 'TRN']
        query_ls = [temp_tuple for temp_tuple in tuple_ls if temp_tuple[2] == 'TST']
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
        return torch.tensor(x_tensor_support).float(), torch.tensor(y_tensor_support), torch.tensor(x_tensor_query).float(), torch.tensor(y_tensor_query)
    def tensorize_test(self, assay_id, test_num):
        x_tensor_query = []
        y_tensor_query = []
        x_tensor_support = []
        y_tensor_support = []
        support_num = 0
        query_num = 0
        encoder_id_ls = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            x = self.fp_dic[cpd_id]
            y = torch.tensor(experiment.expt_pIC50)
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710':
                continue
            if experiment.Clustering == 'TST':
                if query_num > int(test_num/2):
                    continue
                x_tensor_query.append(x)
                y_tensor_query.append(y)
                query_num += 1 
            else:
                if support_num > int(test_num/2):
                    continue
                x_tensor_support.append(x)
                y_tensor_support.append(y)
                support_num += 1
        return torch.tensor(x_tensor_support).float(), torch.tensor(y_tensor_support), torch.tensor(x_tensor_query).float(), torch.tensor(y_tensor_query)
    def tensorize_valid_random(self, assay_id, test_num, r_seed):
        x_y_ls = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            x = self.fp_dic[cpd_id]
            y = torch.tensor(experiment.expt_pIC50)
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710':
                continue
            x_y_ls.append((x, y))
        random.seed(r_seed)
        index_ls = [idx for idx in range(len(x_y_ls))]
        random.shuffle(index_ls)
        x_y_ls = [x_y_ls[idx] for idx in index_ls]
        x_y_ls = x_y_ls[:test_num*2]
        support_num = int(len(x_y_ls)*0.75)
        querry_num = len(x_y_ls) - support_num
        x_tensor_support = torch.tensor([x_y_ls[i][0] for i in range(support_num)]).float()
        y_tensor_support = torch.tensor([x_y_ls[i][1] for i in range(support_num)])
        x_tensor_query = torch.tensor([x_y_ls[i][0] for i in range(support_num, support_num+querry_num)]).float()
        y_tensor_query = torch.tensor([x_y_ls[i][1] for i in range(support_num, support_num+querry_num)])
        return x_tensor_support, y_tensor_support, x_tensor_query, y_tensor_query
    def tensorize_test_shap(self, assay_id, test_num):
        x_tensor_query = []
        y_tensor_query = []
        x_tensor_support = []
        y_tensor_support = []
        support_num = 0
        query_num = 0
        encoder_id_ls = []
        test_cpd_id = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            x = self.fp_dic[cpd_id]
            y = torch.tensor(experiment.expt_pIC50)
            if experiment.Clustering == 'TST':
                if query_num > int(test_num/2):
                    continue
                x_tensor_query.append(x)
                y_tensor_query.append(y)
                query_num += 1 
            else:
                if support_num > int(test_num/2):
                    continue
                x_tensor_support.append(x)
                test_cpd_id.append(cpd_id)
                y_tensor_support.append(y)
                support_num += 1
        return torch.tensor(x_tensor_support).float(), torch.tensor(y_tensor_support), torch.tensor(x_tensor_query).float(), torch.tensor(y_tensor_query), test_cpd_id
    
    