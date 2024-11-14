from torch.utils.data import Dataset, DataLoader

class MLP_train_dataset_fsmol(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        r_seed = self.current_epoch
        task_data = self.data.tensorize(assay_id, 200, r_seed)
        return task_data

class MLP_eval_dataset_fsmol(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        fold_id = 0
        task_data = self.data.tensorize_test(assay_id, fold_id)
        return task_data, assay_id

class MLP_train_dataset_pQSAR(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        r_seed = self.current_epoch
        task_data = self.data.tensorize(assay_id, 200, r_seed)
        return task_data

class MLP_eval_dataset_pQSAR(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        task_data = self.data.tensorize_test(assay_id, 2000)
        return task_data, assay_id


class Graph_train_dataset_fsmol(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        r_seed = self.current_epoch
        task_data = self.data.smiles_to_graph(assay_id, 200, r_seed)
        return task_data

class Graph_eval_dataset_fsmol(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        fold_id = 0
        task_data = self.data.smiles_to_graph_test(assay_id, fold_id)
        return task_data, assay_id
    
class Graph_train_dataset_pQSAR(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        r_seed = self.current_epoch
        task_data = self.data.smiles_to_graph(assay_id, 200, r_seed)
        return task_data

class Graph_eval_dataset_pQSAR(Dataset):
    def __init__(self, assay_ls, data):
        self.assay_ls = assay_ls
        self.data = data
        
    def __len__(self):
        return len(self.assay_ls)
    
    def __getitem__(self, index):
        assay_id = self.assay_ls[index]
        task_data = self.data.smiles_to_graph_test(assay_id, 2000)
        return task_data, assay_id