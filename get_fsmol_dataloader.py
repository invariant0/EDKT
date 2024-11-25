import argparse
from dataset import dataset_constructor
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', default='chembl', type=str)
    parser.add_argument('--model_name', default='actfound', type=str)
    parser.add_argument('--dim_w', default=2048, type=int, help='dimension of w')
    parser.add_argument('--hid_dim', default=2048, type=int, help='dimension of w')
    parser.add_argument('--num_stages', default=2, type=int, help='num stages')
    parser.add_argument('--per_step_bn_statistics', default=True, action='store_false')
    parser.add_argument('--learnable_bn_gamma', default=True, action='store_false', help='learnable_bn_gamma')
    parser.add_argument('--learnable_bn_beta', default=True, action='store_false', help='learnable_bn_beta')
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', default=False, action='store_true', help='enable_inner_loop_optimizable_bn_params')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', default=True, action='store_false', help='learnable_per_layer_per_step_inner_loop_learning_rate')
    parser.add_argument('--use_multi_step_loss_optimization', default=True, action='store_false', help='use_multi_step_loss_optimization')
    parser.add_argument('--second_order', default=1, type=int, help='second_order')
    parser.add_argument('--first_order_to_second_order_epoch', default=10, type=int, help='first_order_to_second_order_epoch')

    parser.add_argument('--transfer_lr', default=0.004, type=float,  help='transfer_lr')
    parser.add_argument('--test_sup_num', default="0", type=str)
    parser.add_argument('--test_repeat_num', default=10, type=int)

    parser.add_argument('--test_write_file', default="./test_result_debug/", type=str)
    parser.add_argument('--expert_test', default="", type=str)
    parser.add_argument('--similarity_cut', default=1., type=float)

    parser.add_argument('--train_seed', default=1111, type=int, help='train_seed')
    parser.add_argument('--val_seed', default=1111, type=int, help='val_seed')
    parser.add_argument('--test_seed', default=1111, type=int, help='test_seed')

    parser.add_argument('--metatrain_iterations', default=80, type=int,
                        help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
    parser.add_argument('--meta_batch_size', default=30, type=int, help='number of tasks sampled per meta-update')
    parser.add_argument('--min_learning_rate', default=0.0001, type=float, help='min_learning_rate')
    parser.add_argument('--update_lr', default=0.001, type=float, help='inner learning rate')
    parser.add_argument('--meta_lr', default=0.00015, type=float, help='the base learning rate of the generator')
    parser.add_argument('--num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in maml')
    parser.add_argument('--multi_step_loss_num_epochs', default=5, type=int, help='multi_step_loss_num_epochs')
    parser.add_argument('--norm_layer', default='batch_norm', type=str, help='norm_layer')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in beta distribution')


    ## Logging, saving, and testing options
    parser.add_argument('--logdir', default='', type=str,
                        help='directory for summaries and checkpoints.')
    parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
    parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
    parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

    parser.add_argument('--new_ddg', default=False, action='store_true')

    parser.add_argument('--input_celline', default=False, action='store_true')
    parser.add_argument('--cell_line_feat', default='./datas/gdsc/cellline_to_feat.pkl')
    parser.add_argument('--cross_test', default=False, action='store_true')
    parser.add_argument('--gdsc_pretrain', default="none", type=str)
    parser.add_argument('--use_byhand_lr', default=False, action='store_true')

    parser.add_argument('--inverse_ylabel', default=False, action='store_true')
    parser.add_argument('--knn_maml', default=False, action='store_true')
    parser.add_argument('--train_assay_feat_all', default='')
    parser.add_argument('--train_assay_idxes', default='')
    parser.add_argument('--knn_dist_thres', default=0.3, type=float)
    parser.add_argument('--begin_lrloss_epoch', default=50, type=int)
    parser.add_argument('--lrloss_weight', default=35., type=float)
    parser.add_argument('--no_fep_lig', default=False, action='store_true')
    parser.add_argument('--act_cliff_test', default=False, action='store_true')

    return parser

def get_fsmol_dataloader():
    # Define the configuration parameters
    FIXED_PARAM_FSMOL = {
        'test_sup_num': 32,
        'test_repeat_num': 10,
        'train': 1,
        'test_epoch': -1
    }

    FSMOL_KNN_MAML = {}  # Empty dict since original is empty
    FSMOL_DIR = "./checkpoints_all/checkpoints_fsmol"
    FSMOL_RES = "./test_results/result_indomain/fsmol"

    parser = get_args()
    # Set the fixed parameters
    args = parser.parse_args([])  # Create empty args
    args.datasource = 'fsmol'
    args.logdir = f"{FSMOL_DIR}/checkpoint_fsmol_actfound"
    args.model_name = 'actfound'
    args.test_write_file = FSMOL_RES
    for key, value in FIXED_PARAM_FSMOL.items():
        setattr(args, key, value)
    dataloader = dataset_constructor(args)
    return dataloader