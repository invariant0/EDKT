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
        self.small_cpd_id = np.load('../Data_for_publication/pQSAR/small_cpd_id.pkl', allow_pickle=True)
        # self.get_mol_graph = np.load('../data/mol_graph_data.pickle', allow_pickle=True)
        self.get_mol_graph = np.load('../Data_for_publication/pQSAR/pyg_graph_dic.pkl', allow_pickle=True)
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
    
    def smiles_to_graph(self, assay_id, sample_num, r_seed):
        tuple_ls = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710' or cpd_id not in self.small_cpd_id:
                continue
            y = torch.tensor(experiment.expt_pIC50)
            tuple_ls.append((cpd_id, y, experiment.Clustering))

        support_ls = [temp_tuple for temp_tuple in tuple_ls if temp_tuple[2] == 'TRN']
        query_ls = [temp_tuple for temp_tuple in tuple_ls if temp_tuple[2] == 'TST']
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
            graph_data_support_return.append(self.get_mol_graph[support_ls[idx][0]])
            label_data_support_return.append(support_ls[idx][1])
        for idx in query_idx:
            graph_data_query_return.append(self.get_mol_graph[query_ls[idx][0]])
            label_data_query_return.append(query_ls[idx][1])
        return graph_data_support_return, label_data_support_return, graph_data_query_return, label_data_query_return

    def smiles_to_graph_test(self, assay_id, test_num):
        graph_data_support, graph_data_query, label_data_support_return, label_data_query_return = [], [], [], []
        all_data = []
        support_num = 0
        query_num = 0
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710' or cpd_id not in self.small_cpd_id:
                continue
            if experiment.Clustering == 'TRN':
                if support_num > int(test_num/2):
                    continue
                graph_data_support.append(self.get_mol_graph[cpd_id])
                label_data_support_return.append(experiment.expt_pIC50)
                support_num += 1
            else:
                if query_num > int(test_num/2):
                    continue
                graph_data_query.append(self.get_mol_graph[cpd_id])
                label_data_query_return.append(experiment.expt_pIC50)
                query_num += 1
        return graph_data_support, label_data_support_return, graph_data_query, label_data_query_return
    
    def smiles_to_graph_test_all(self, assay_id, test_num):
        graph_data_support, graph_data_query, label_data_support_return, label_data_query_return = [], [], [], []
        all_data = []
        support_num = 0
        query_num = 0
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710':# or cpd_id not in self.small_cpd_id:
                continue
            if experiment.Clustering == 'TST':
                if query_num > int(test_num/2):
                    continue
                graph_data_query.append(self.get_mol_graph[cpd_id])
                label_data_query_return.append(experiment.expt_pIC50)
                query_num += 1
            else:
                if support_num > int(test_num/2):
                    continue
                graph_data_support.append(self.get_mol_graph[cpd_id])
                label_data_support_return.append(experiment.expt_pIC50)
                support_num += 1
        return graph_data_support, label_data_support_return, graph_data_query, label_data_query_return
    def smiles_to_graph_valid_all_random(self, assay_id, test_num, r_seed):
        graph_data_support, graph_data_query, label_data_support_return, label_data_query_return = [], [], [], []
        graph_label_data_ls = []
        for experiment in self.all_data.assay_dic[assay_id].experiments:
            cpd_id = experiment.cpd_id
            if cpd_id == 'CHEMBL542448' or cpd_id == 'CHEMBL69710':# or cpd_id not in self.small_cpd_id:
                continue
            graph_label_data_ls.append((self.get_mol_graph[cpd_id], experiment.expt_pIC50))
        random.seed(r_seed)
        index_ls = [idx for idx in range(len(graph_label_data_ls))]
        random.shuffle(index_ls)
        graph_label_data_ls = [graph_label_data_ls[idx] for idx in index_ls]
        graph_label_data_ls = graph_label_data_ls[:test_num*2]
        support_num = int(len(graph_label_data_ls) * 0.75)
        query_num = len(graph_label_data_ls) - support_num
        graph_data_support = [graph_label_data_ls[i][0] for i in range(support_num)]
        label_data_support = [graph_label_data_ls[i][1] for i in range(support_num)]
        graph_data_query = [graph_label_data_ls[i][0] for i in range(support_num, support_num+query_num)]
        label_data_query = [graph_label_data_ls[i][1] for i in range(support_num, support_num+query_num)]
        return graph_data_support, label_data_support, graph_data_query, label_data_query