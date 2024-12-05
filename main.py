import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
import math
import argparse
import os
import gc

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGP Training Script")
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")
    parser.add_argument("--dataset", type=str, help="what dataset to use", default='fsmol')
    parser.add_argument("--encode_method", type=str, help="what encoder to use", default='FP')
    parser.add_argument("--num_encoder", type=int, default=2, help="num_encoder")
    parser.add_argument("--allow_NCL", action="store_true", help="whether use NCL")
    
    args, _ = parser.parse_known_args()
    
    if args.dataset == 'fsmol':
        parser.add_argument("--FP_input_dim", type=int, default=2024, help="FP input dimension")
        parser.add_argument("--Graph_input_dim", type=int, default=32, help="Graph input dimension")
    elif args.dataset == 'pQSAR':
        parser.add_argument("--FP_input_dim", type=int, default=1024, help="FP input dimension")
        parser.add_argument("--Graph_input_dim", type=int, default=30, help="Graph input dimension")
        parser.add_argument("--group_id", type=int, default=0, help="group_id")
    
    parser.add_argument("--world_size", type=int, default=6, help="number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    if 'FP' in args.encode_method:
        parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
    elif 'Graph' in args.encode_method:
        parser.add_argument("--lr", type=float, help="learning rate", default=0.0005)
    return parser.parse_args()

args = parse_args()

if 'FP' in args.encode_method:
    from deep_ensemble import deep_ensemble_FP as deep_ensemble
    if args.dataset == 'fsmol':
        from EDKT_data.Data_FP_fsmol import deep_gp_data
        from DKT_dataset import MLP_train_dataset_fsmol as MLP_train_dataset 
        from DKT_dataset import MLP_eval_dataset_fsmol as MLP_eval_dataset
    elif args.dataset == 'pQSAR':
        from EDKT_data.Data_FP_pQSAR import deep_gp_data
        from DKT_dataset import MLP_train_dataset_pQSAR as MLP_train_dataset
        from DKT_dataset import MLP_eval_dataset_pQSAR as MLP_eval_dataset
elif 'Graph' in args.encode_method:
    from deep_ensemble import deep_ensemble_Graph as deep_ensemble
    if args.dataset == 'fsmol':
        from EDKT_data.Data_Graph_fsmol import deep_gp_data
        from DKT_dataset import Graph_train_dataset_fsmol as Graph_train_dataset
        from DKT_dataset import Graph_eval_dataset_fsmol as Graph_eval_dataset
    elif args.dataset == 'pQSAR':
        from EDKT_data.Data_Graph_pQSAR import deep_gp_data
        from DKT_dataset import Graph_train_dataset_pQSAR as Graph_train_dataset
        from DKT_dataset import Graph_eval_dataset_pQSAR as Graph_eval_dataset

def custom_collate_fn(batch):
    return [item for item in batch if item is not None]

def train_epoch(model, data, train_assay_ls, eval_assay_ls, optimizer, device, rank, world_size):

    if 'FP' in args.encode_method:
        dataset_train = MLP_train_dataset(train_assay_ls, data)
        dataset_eval = MLP_eval_dataset(eval_assay_ls, data)
    elif 'Graph' in args.encode_method:
        dataset_train = Graph_train_dataset(train_assay_ls, data)
        dataset_eval = Graph_eval_dataset(eval_assay_ls, data)

    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
    train_data_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler, collate_fn=custom_collate_fn)
    eval_data_loader = DataLoader(dataset_eval, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    if rank == 0:
        print(args.batch_size)
        if args.dataset == 'fsmol':
            log_file_path = f"./temp_result/Dataset:{args.dataset}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.txt"
            result_file_path = f"./pred_result/Dataset:{args.dataset}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.txt"
        elif args.dataset == 'pQSAR':
            log_file_path = f"./temp_result/Dataset:{args.dataset}_Groupid:{args.group_id}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.txt"
            result_file_path = f"./pred_result/Dataset:{args.dataset}_Groupid:{args.group_id}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.txt"
        with open(log_file_path, 'w') as f:
            f.writelines('training...\n')
        with open(result_file_path, 'w') as f:
            f.writelines('Prediction:\n')
    if 'FP' in args.encode_method:
        itter_num = 40
    elif 'Graph' in args.encode_method:
        itter_num = 80
    for epoch in range(itter_num):
        train_data_loader.sampler.set_epoch(epoch)
        dataset_train.set_epoch(epoch)
        model.train()
        epoch_loss = []
        for batch in train_data_loader:
            batch_loss = 0
            for idx in range(len(batch)):
                task_data = batch[idx]
                loss = model(task_data, device)
                batch_loss += loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch}, Loss: {np.mean(epoch_loss)}, Learning Rate: {current_lr}")
        if (epoch+1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                r2_ls = []
                for batch in eval_data_loader:
                    for idx in range(len(batch)):
                        task_data, assay_id = batch[idx]
                        prediction = model.module.prediction(task_data, device)
                        y_label = task_data[3]
                        r2 = np.corrcoef(torch.tensor(y_label).reshape(-1), prediction.detach().cpu().numpy().reshape(-1))[0,1]**2
                        if math.isnan(r2):
                            r2 = 0
                        r2_ls.append(r2)
                        if epoch == itter_num - 1 and rank == 0:
                            with open(result_file_path, 'a') as f:
                                f.writelines(f'{assay_id}, {r2}\n')
                # save log file
                if rank == 0:
                    print(f"Mean: {np.mean(r2_ls)}, Median: {np.median(r2_ls)}, Num: {len(r2_ls)}")
                    with open(log_file_path, 'a') as f:
                        f.writelines('{} {} {}\n'.format(np.mean(r2_ls), np.median(r2_ls), len(r2_ls)))
                    # save model after last epoch
                    if epoch == itter_num - 1:
                        if args.dataset == 'fsmol':
                            torch.save(model.module.state_dict(), f'../Model_for_publication/Dataset:{args.dataset}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.pth')
                        elif args.dataset == 'pQSAR':
                            torch.save(model.module.state_dict(), f'../Model_for_publication/Dataset:{args.dataset}_Groupid:{args.group_id}_Method:{args.encode_method}_Num:{args.num_encoder}_NCL:{args.allow_NCL}_seed:{args.random_seed}.pth')
        # Clear memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, args):
    setup(rank, world_size)
    seed_all(args.random_seed)
    
    device = torch.device(f"cuda:{rank}")
    
    if 'FP' in args.encode_method:
        if args.dataset == 'fsmol':
            assay_id_train_test_split = np.load('../Data_for_publication/fsmol_multifp/split_dic.pkl', allow_pickle=True)
            train_assay_ls = assay_id_train_test_split['train_assays']
            print(len(train_assay_ls))
            eval_assay_ls = assay_id_train_test_split['test_assays']
            data_path = '../Data_for_publication/fsmol_multifp/all_data_fp.pkl'
            data_path_test = '../Data_for_publication/fsmol_multifp/all_data_fp_test_10fold_32.pkl'
            data = deep_gp_data(data_path, data_path_test, args)
        elif args.dataset == 'pQSAR':
            assay_id_train_test_split = np.load('../Data_for_publication/pQSAR/split_dic.pkl', allow_pickle=True)
            train_assay_ls = assay_id_train_test_split[args.group_id]['train_assays']
            print(len(train_assay_ls))
            eval_assay_ls = assay_id_train_test_split[args.group_id]['test_assays']
            data_path = '../Data_for_publication/pQSAR/ci9b00375_si_002.txt'
            data = deep_gp_data(data_path)
    elif 'Graph' in args.encode_method:
        if args.dataset == 'fsmol':
            assay_id_train_test_split = np.load('../Data_for_publication/fsmol_multifp/split_dic.pkl', allow_pickle=True)
            train_assay_ls = assay_id_train_test_split['train_assays']
            print(len(train_assay_ls))
            eval_assay_ls = assay_id_train_test_split['test_assays']
            data_path = '../Data_for_publication/fsmol_multifp/all_data_graph.pkl'
            data_path_test = '../Data_for_publication/fsmol_multifp/all_data_graph_test_10fold_32.pkl'
            data = deep_gp_data(data_path, data_path_test)
        elif args.dataset == 'pQSAR':
            # assay_id_train_test_split = np.load('../Data_for_publication/pQSAR/split_dic.pkl', allow_pickle = True)
            # included_assay = np.load('../Data_for_publication/pQSAR/included_assay.pkl', allow_pickle = True)
            # train_assay_ls = assay_id_train_test_split[args.group_id]['train_assays']
            # eval_assay_ls = assay_id_train_test_split[args.group_id]['test_assays']
            # train_assay_ls = [x for x in train_assay_ls if x in included_assay]
            # eval_assay_ls = [x for x in eval_assay_ls if x in included_assay]
            assay_id_train_test_split = np.load('../Data_for_publication/pQSAR/pQSAR_split_dic.pkl', allow_pickle=True)
            train_assay_ls = list(assay_id_train_test_split[args.group_id]['train'])
            eval_assay_ls = list(assay_id_train_test_split[args.group_id]['test'])
            included_assay = np.load('../Data_for_publication/pQSAR/included_assay.pkl', allow_pickle = True)
            train_assay_ls = [x for x in train_assay_ls if x in included_assay]
            eval_assay_ls = [x for x in eval_assay_ls if x in included_assay]
            print(len(train_assay_ls))
            data_path = '../Data_for_publication/pQSAR/ci9b00375_si_002.txt'
            data = deep_gp_data(data_path)
    print(f"num_encoder: {args.num_encoder}, methods: {args.encode_method}, dataset: {args.dataset}, NCL: {args.allow_NCL}")
    model = deep_ensemble.ensemble_deep_gp(args)
    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_epoch(model, data, train_assay_ls, eval_assay_ls, optimizer, device, rank, world_size)
    cleanup()

def main():
    args = parse_args()
    world_size = args.world_size
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()