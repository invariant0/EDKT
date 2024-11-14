# data_classes.py
class experiment:
    def __init__(self, assay_id, cpd_id, expt_pIC50):
        self.assay_id = assay_id
        self.cpd_id = cpd_id
        self.expt_pIC50 = expt_pIC50

class experiment_test:
    def __init__(self, assay_id, cpd_id, expt_pIC50, test_flag_fold):
        self.assay_id = assay_id
        self.cpd_id = cpd_id
        self.expt_pIC50 = expt_pIC50
        self.test_flag_fold = test_flag_fold

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