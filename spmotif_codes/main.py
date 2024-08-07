import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os.path as osp
import random
## dataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from spmotif_dataset import SPMotif
## training
from model import  Bias_Augmention,FedGR
from utils import init_weights, get_args,train_with_both_data_theory,train_bias_augmention_model,get_bias_data, eval_test_bias,eval_test_class_bias
import copy
import json
import logging


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

args_first = get_args()
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG,  #INFO,
)
logger = logging.getLogger(__name__)

def main(args,seed):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    datadir = './data/'
    bias = args.bias
    train_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(datadir, f'SPMotif-{bias}/'), mode='test')
    
    valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ## split the client
    random.seed(42)
    np.random.seed(42)

    def partition_class_samples_with_dirichlet_distribution(N, alpha, client_num, idx_batch, idx_k):
        np.random.shuffle(idx_k)
        # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
        # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))
        print(proportions)
        weight = proportions
        # get the index in idx_k according to the dirichlet distribution
        proportions = np.array([p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # generate the batch list for each client
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

        return idx_batch, min_size,weight

    def create_non_uniform_split(alpha, idxs, client_number, is_train=True):
        
        N = len(idxs)
        logging.info("sample number = %d, client_number = %d" % (N, client_number))
        logging.info(idxs)
        idx_batch_per_client = [[] for _ in range(client_number)]
        (
            idx_batch_per_client,
            min_size,
            weight,
        ) = partition_class_samples_with_dirichlet_distribution(
            N, alpha, client_number, idx_batch_per_client, idxs
        )
        logging.info(idx_batch_per_client)
        sample_num_distribution = []

        for client_id in range(client_number):
            sample_num_distribution.append(len(idx_batch_per_client[client_id]))
            logging.info(
                "client_id = %d, sample_number = %d"
                % (client_id, len(idx_batch_per_client[client_id]))
            )
        return idx_batch_per_client,weight

    def get_fed_dataset(train,client_number,alpha ):
        num_train_samples = len(train)
        train_idxs = list(range(num_train_samples))
        random.shuffle(train_idxs)

        clients_idxs_train,weight = create_non_uniform_split(
        alpha,  train_idxs, client_number, True
        )
        # print(clients_idxs_train)
        partition_dicts = [None] * client_number

        for client in range(client_number):
            client_train_idxs = clients_idxs_train[client]

            train_client = [
                train[idx] for idx in client_train_idxs
            ]

            partition_dict = {
            "train": train_client,
            }

            partition_dicts[client] = partition_dict

        return partition_dicts,weight

    partition_dicts_clients,client_weights = get_fed_dataset(train_dataset, client_number=3, alpha=args.alpha)

    partition_dicts = [None] * args.client_number
    for client in range(args.client_number):
        train_loader = DataLoader(partition_dicts_clients[client]['train'], batch_size=args.batch_size, shuffle=True, num_workers = 0)
        partition_dict = {
        "train": train_loader,
        }
        partition_dicts[client] = partition_dict
        print(len(train_loader))

    set_seed(seed)

    for step, batched_data in enumerate(valid_loader):
        if step>2:break
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

    server_model = eval(args.model_name)( gnn_type = args.gnn, num_tasks = 3, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)   
                          
    init_weights(server_model, args.initw_name, init_gain=0.02)

    models = [copy.deepcopy(server_model) for idx in range(args.client_number)]


    bias_augmention_model = Bias_Augmention( gnn_type = args.gnn, num_tasks = x.size(1), num_layer = 2,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device)

    bias_augmention_models = [copy.deepcopy(bias_augmention_model) for idx in range(args.client_number)]

    bias_optimizers = [optim.Adam(bias_augmention_model.parameters(), lr=args.lr, weight_decay=args.l2reg) for bias_augmention_model in bias_augmention_models]


    def get_opt(args,model):
        opt_separator = optim.Adam(list(model.separator.parameters())  + list(model.node_enoder.parameters()), lr=args.lr, weight_decay=args.l2reg)
        opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters())  + list(model.node_enoder.parameters()), lr=args.lr, weight_decay=args.l2reg)

        optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
        if args.use_lr_scheduler:
            schedulers = {}
            for opt_name, opt in optimizers.items():
                schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
        else:
            schedulers = None
        return optimizers,schedulers

    def communication(server_model, models, client_weights):
        client_num = len(models)
        with torch.no_grad():
            for key in server_model.state_dict().keys():
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                # 参数聚合
                # print(client_num)
                for client_idx in range(client_num):
                    #  print(client_weights[client_idx])
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                # 参数分发
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        return server_model, models

    cnt_wait = 0
    best_epoch = 0
    loss_logger = []
    valid_logger = []
    test_logger = []
    task_type = ["classification"]
    for epoch in range(args.epochs):
        
        all_opt_and_sch = [ get_opt(args, models[idx])  for idx in range(args.client_number)]
        
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'

        for client_idx, model in enumerate(models):
            temp_time = 0
            for i in tqdm(range(10)):
                model.train()
                optimizers,schedulers = all_opt_and_sch[client_idx]
                train_loader = partition_dicts[client_idx]['train']
                if epoch == 0:
                    train_with_both_data_theory(args, model, device, train_loader, optimizers, task_type, optimizer_name,loss_logger,bias_augmention_loader=None)
                else:  
                    if temp_time ==0:
                        # 1.train bias_augmention_model
                        train_bias_augmention_model(args, train_loader, device,task_type,bias_augmention_models[client_idx],bias_optimizers[client_idx], bias_model =  bias_models[client_idx], unbias_model = model)
                        # 2.generate bias_augmention_data
                        bias_augmention_loader = get_bias_data(args,partition_dicts_clients[client_idx]['train'], bias_augmention_models[client_idx],device)
                        # 3.training with bias_augmention_data
                        temp_time = 1
                    else:
                        train_with_both_data_theory(args, model, device, train_loader, optimizers, task_type, optimizer_name,loss_logger,bias_augmention_loader)

                if schedulers != None:
                    schedulers[optimizer_name].step()
                
        with torch.no_grad():
            # 保存本地模型作为bias model
            bias_models = [copy.deepcopy(bias_model) for bias_model in models]
            # 进行参数的聚合和分发；models作为新的全局 unbias model
            server_model, models = communication( server_model, models, client_weights)

        server_model.eval()
        valid_perf = eval_test_bias(args, server_model, device, valid_loader)[0]
        test_logger_perfs = eval_test_bias(args, server_model, device, test_loader)[0]
        valid_logger.append(valid_perf)
        test_logger.append(test_logger_perfs)
        update_test = False

        test_perfs = eval_test_bias(args, server_model, device, test_loader)
        class_test_perfs = eval_test_class_bias(args, server_model, device, test_loader)
        test_auc  = test_perfs[0]
        class_test_auc = class_test_perfs[0]
        print("=====Epoch {}, Metric: {}, Validation: {}, Test: {}, Class_Test:{}".format(epoch, 'AUC', valid_perf, test_auc,class_test_auc))

        update_test = False
        if epoch != 0:
            if valid_perf == 1.0:
                update_test = False
            elif 'classification' in task_type and valid_perf >  best_valid_perf:
                update_test = True
            elif 'classification' not in task_type and valid_perf <  best_valid_perf:
                update_test = True

        if update_test or epoch == 0:
            best_valid_perf = valid_perf
            cnt_wait = 0
            best_epoch = epoch
            test_auc1 = test_auc
        else:

            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    logger.info('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))



    
    logger.info('Test auc: {}'.format(test_auc))
    return [best_valid_perf, test_auc1]

    

def config_and_run(args):
    
    if args.by_default:
        if args.dataset == 'spmotif':
            args.epochs = 10
            if args.gnn == 'gin-virtual' or args.gnn == 'gin':
                args.gnn = 'gin'
                args.l2reg = 1e-3
                args.gamma = 0.55
                args.num_layer = 2  
                args.batch_size = 32
                args.emb_dim = 64
                args.use_lr_scheduler = True
                args.patience = 40
                args.drop_ratio = 0.3
                args.initw_name = 'orthogonal' 
            if args.gnn == 'gcn-virtual' or args.gnn == 'gcn':
                args.gnn = 'gcn'
                args.patience = 40
                args.initw_name = 'orthogonal' 
                args.num_layer = 2
                args.emb_dim = 64
                args.batch_size = 32

    for k, v in vars(args).items():
        logger.info("{:20} : {:10}".format(k, str(v)))

    args.plym_prop = 'none' 
    
    results = {'valid_auc': [], 'test_auc': []}
    
    for seed in range(args.trails):

        set_seed(seed)
        valid_auc, test_auc = main(args,seed)
        results['valid_auc'].append(valid_auc)
        results['test_auc'].append(test_auc)

    for mode, nums in results.items():
        logger.info('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))

        print('{}: {:.4f}+-{:.4f} {}'.format(
            mode, np.mean(nums), np.std(nums), nums))
        xx = '{}: {:.4f}+-{:.4f} {}'.format(mode, np.mean(nums), np.std(nums), nums)
        json_str = json.dumps({'result': xx }, ensure_ascii=False)
        f_json.write(json_str)
        f_json.write('\n')
    
if __name__ == "__main__":
    args = get_args()
    config_and_run(args)
    




