import torch
import wandb
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import get_data_and_loaders, forward_pass, process_data, task_specific_modifiers
import copy
import random
from utils import get_network, get_forward_op, get_experiment_name
import numpy as np
from utils import count_trainable_parameters, save_model
### THIS SCRIPT MAY BE USED TO RUN THE 3 LINEAR INVERSE PROBLEMS IN THE PAPER:  'deblur' (inverse source estimation),  'mask' (property completion), 'path' (inverse graph transport) 

##################################
########### ARGUMENTS: ###########
##################################
# NOTE: Set arguments for the run here. Read through each setting and make sure it is consistent with the experiment. For example, check to see that for classification tasks, default_classify is set to 1 (and so on)
# NOTE: If there are arguments that do not seem applicable for an experiment, such as 'regnet' when 'method' was NOT chosen to be 'drip', then the argument does not matter and it is safe to leave it as it is. 
# NOTE: (Continued from above) but "regnet" affects the name of the experiment outputted by "get_experiment_name", so you can choose regnet='None' for tikhonov regularization, or 'pgd' (when method='pgd), or 'laplace' (when using a 'method' that does laplacian regularization) for a consistent experiment name.

# set the number of runs for this script. Each will be initiated with a seed 0, 1...num_seeds in integer steps
num_seeds = 1 
# the datapath where you're storing the dataset 
default_datapath = './data' 
# put your wandb username here
default_wandb_user = 'nafi007'
# the default learning rate during training  
default_lr = 1e-3 
# choose from: 'CLUSTER'  'METRLA', 'CPOX' 'SHAPENET'
default_dataset = 'CPOX' 
# the inverse problem: can be 'deblur' (inv. source estimation),  'mask' (property completion), 'path' (inv. graph transport)
default_task = 'mask'                      
# for classification problems set this to 1. For regression problems set this to 0
default_classify = 1     
# the batch size to use during training  
# NOTE: when using 'laplacian_explicit' or 'laplacian_regularization' for "method" argument, set this to 1.             
default_train_batch_size = 4
# the number of training epochs        
default_epochs = 60000   
# Valid only for classification problems. The number of nodes 'seen' per class per graph. Only if dataset == 'SHAPENET' or dataset == 'CLUSTER'  
default_mask_per_class_budget = 4       
# Valid only for regression problems. The number of nodes 'seen' per snapshot (graph). Only if dataset == 'METRLA' or dataset == 'CPOX' 
default_mask_per_snapshot_budget = 16   
# the date and time at which this script was initiated, will be included in the wandb label for the run
time_ = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

from enum import Enum

class DataSplit(Enum):
    train = 1
    validation = 2
    test = 3
    non = 4

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=int, default=0)
parser.add_argument('--datapath', type=str, default=default_datapath)
parser.add_argument('--savepath', type=str, default='/checkpoints')
parser.add_argument('--method', type=str, default='drip')  # choose: 'drip' for Var-GNN / 'pgd' for Prox-GNN / 'inv_scale_space' for ISS-GNN / 'tikhonov_regularization' / 'laplacian_explicit' / 'laplacian_regularization' 
parser.add_argument('--regnet', type=str, default='hyper')  # this only makes a difference if you chose 'drip' for 'method', for which you should choose 'hyper'. You can also choose 'None' or 'pgd' or 'laplace' for labeling convenience if you chose the corresponding 'method' for these. It does not matter what you choose here IF you DID NOT choose 'drip' for 'method'
parser.add_argument('--channels', type=int, default=32)  
parser.add_argument('--layers', type=int, default=16) 
parser.add_argument('--train_batch_size', type=int, default=default_train_batch_size) 
parser.add_argument('--task', type=str, default=default_task)
parser.add_argument('--lr', type=float, default=default_lr)
parser.add_argument('--wd', type=float, default=4e-6)
parser.add_argument('--cglsIter', type=int, default=50) 
parser.add_argument('--solveIter', type=int, default=50) 
parser.add_argument('--dataset', type=str, default=default_dataset)
parser.add_argument('--classify', type=int, default=default_classify)
parser.add_argument('--rnfPE', type=int, default=1)
parser.add_argument('--epochs', type=int, default=default_epochs)
parser.add_argument('--head_epochs', type=int, default=0)
parser.add_argument('--test_head_epochs', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--pathLength', type=int, default=32)
parser.add_argument('--mu', type=float, default=0.01)
parser.add_argument('--wandb_user', type=str, default=default_wandb_user)
parser.add_argument('--blur_count', type=str, default='4') #number of times the graph/image is blurred. The number of diffusion steps
parser.add_argument('--mask_per_class_budget', type=int, default=default_mask_per_class_budget)
parser.add_argument('--mask_per_snapshot_budget', type=int, default=default_mask_per_snapshot_budget)
parser.add_argument('--test_batch_size', type=int, default = int(default_train_batch_size) ) 
parser.add_argument('--CPOX_lags', type=int, default=1) # keep this 1
parser.add_argument('--num_seeds', type=int, default=num_seeds) 
parser.add_argument('--LapNoRegNet_tol', type=float, default=0.005/2)  # tolerance (only for laplace and no-reg cases #0.005/2)
parser.add_argument('--train_frac', type=float, default=1.0)  #fraction of train set to use. Default is whole train set (= 1.0)
parser.add_argument('--test_frac', type=float, default=1.0)  #fraction of test set to use. Default is whole test set (= 1.0), currently only for METRLA
parser.add_argument('--project_name', type=str, default="test")
parser.add_argument('--use_meta_data', type=int, default=1) # if 1 , then meta_data used if available. 0 implies it won't be used.
parser.add_argument('--max_patience', type=int, default=100) #35
parser.add_argument('--seed', type=float, default=0) 
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--noise', type = bool, default = False) # if True, adds noise to the input data. If False, no noise is added.
parser.add_argument('--painting', type = bool, default = False) # if True, applies inpainting/masking to the input data. If False, no inpainting is applied.
parser.add_argument('--blurring', type = bool, default = False) # if True, applies source localization/deblurring to the input data. If False, no deblurring is applied.
parser.add_argument('--sensoring', type = bool, default = False) # if True, applies sensor recovery to the input data. If False, no sensor recovery is applied.
parser.add_argument('--pdessm', type = bool, default = False) # if True, applies PDE-state reconstruction to the input data. If False, no PDE-state reconstruction is applied.
parser.add_argument('--n_sensors', type=int, default=5, 
                    help='Number of sensor nodes kept per snapshot')
parser.add_argument('--val_frac', type=float, default=0.1) # validation = last val_frac of the training time series
parser.add_argument('--pde_tau', type=float, default=20.0) # PDE diffusion time; tau=1 was trivially invertible (cond ~3)
parser.add_argument('--train_datasets', type=str, default='CPOX,PEDALME,WIKIMATHS,MONTEVIDEO') # only with --dataset MULTI
parser.add_argument('--test_dataset', type=str, default='WINDMILL') # only with --dataset MULTI: held-out dataset, tested on all of it

args = parser.parse_args()
args.test_batch_size = args.train_batch_size
print(f"{args.noise=}, {args.painting=}, {args.blurring=}, {args.sensoring=}, {args.pdessm=}")
# Set experiment name
exp_name = get_experiment_name(args, time_)

##################################
########### PARSING: ###########
##################################

print(f"Project Name: {args.project_name}")
# Initialize wandb with the sweep configuration
wandb.init(name = exp_name, project=args.project_name) 
# config = wandb.config

device = args.device
# Aggregate metrics for all seeds
# overall_train_losses = []
# overall_test_losses = []
# overall_train_accuracies = []
# overall_test_accuracies = []
best_test_losses = []
best_test_accs = []
best_test_losses_corr_X = []
best_test_losses_corr_data = []
for seed_temp in range(args.num_seeds):

    # seed = abs(torch.randn(1)[0].item() + torch.randn(1)[0].item())
    # seed = seed + 1
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels, val_loader = get_data_and_loaders(args)
    if val_loader is None:
        print('WARNING: no validation split for this dataset -> model selection falls back to the TEST set')

    hid_channels = args.channels

    sample_graph = next(iter(test_dataset))
    num_nodes = sample_graph.x.shape[0]
    args.num_nodes = num_nodes

    forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test = False) # flag=False means that the forward operator is for training, so we want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    test_forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test=True) or forward_op # all tasks trained (leave-one-dataset-out) -> test every task. flag=True means that the forward operator is for testing, so we don't want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    net = get_network(args, forward_op + test_forward_op, hid_channels, label_channels, feat_channels, device=device)
    net = net.to(device)  #already in device from get_network function

    #### count trainable parameters ###
    total_params = count_trainable_parameters(net)
    print(f"Total number of trainable parameters: {total_params}")
    #####################################

    if args.method not in ['tikhonov_regularization', 'laplacian_regularization', 'laplacian_explicit']:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True, eps=1e-3)
    else:
        optimizer = None
    niters = args.epochs
    avg_freq = 1
    av_loss = 0
    av_test_loss = 0

    ##################################
    ########### TRAIN & EVAL: ###########
    ##################################
    def train(net, epoch=0, forward_op=None, loader=train_loader):
        av_loss = 0
        av_loss_X = 0
        av_loss_data = 0
        av_train_acc = 0
        total_train_batches = 0   # in a window
        total_acc = 0
        total_loss = 0
        total_loss_X = 0
        total_loss_data = 0
        number_all_batches = 0  # in whole train epoch

        ops = forward_op if isinstance(forward_op, list) else None

        net.train()
        for graph_idx, graph in enumerate(loader):
            if ops is not None:  # random task per batch (one task per epoch let the backbone drift to the last task)
                forward_op, op_name = random.choice(ops)
                net.set_task(op_name)
            
            actual_bs = len(graph.batch.unique())
            total_train_batches += actual_bs
            number_all_batches += actual_bs
            graph = process_data(args, graph)
            graph, forward_op =  task_specific_modifiers(graph, args, forward_op, train_dataset)
            
            graph = graph.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            
            clean_forward_data = forward_op(graph.y, graph.edge_index, graph.edge_weight, emb=False)
            forward_data = clean_forward_data
            if hasattr(forward_op, 'corrupt'):
                forward_data = forward_op.corrupt(clean_forward_data)
            if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
                # print(graph_idx, graph)
                X = net(forward_data, graph)
                # print("went through net")
            else:
                X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
            

            forward_data_rec = forward_op(X, graph.edge_index, graph.edge_weight, emb=False)

            if args.classify: 
                loss_X = F.cross_entropy(X, graph.y) 
                loss_data = F.cross_entropy(forward_data_rec, clean_forward_data)
                loss = loss_X + loss_data
                pred = torch.argmax(X, dim=-1)
                acc = torch.eq(pred, torch.argmax(graph.y, dim=-1)).sum() / graph.x.shape[0]
                av_train_acc += acc.item() * actual_bs
                # loss_X = torch.tensor([0])
                # loss_data = torch.tensor([0])
            else:
                # regression
                # loss = F.mse_loss(X, graph.y) / F.mse_loss(torch.zeros_like(graph.y), graph.y)
                # loss += F.mse_loss(forward_data_rec, forward_data)
                loss_X = F.mse_loss(X, graph.y)/F.mse_loss(torch.zeros_like(graph.y), graph.y) # we can put this in because the datasets we're using are not one big graph but have train/val/test sets 
                loss_data = F.mse_loss(forward_data_rec, clean_forward_data) / F.mse_loss(torch.zeros_like(clean_forward_data), clean_forward_data)
                loss = 0.5 * (loss_X + loss_data)
                acc = torch.tensor([0])

            if optimizer is not None:
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

            av_loss += loss.item()
            av_loss_X += loss_X.item()
            av_loss_data += loss_data.item()
            total_loss_X += loss_X.item() * actual_bs
            total_loss_data += loss_data.item() * actual_bs
            total_loss += loss.item() * actual_bs
            total_acc += acc * actual_bs

            if graph_idx % avg_freq == avg_freq - 1 and args.cluster == 0:
                if args.classify:
                    #print("Epoch:", i, "Iter:", graph_idx, ", Avg loss:", av_loss / avg_freq, ", avg acc:", av_train_acc / total_train_batches,
                    #    flush=True)
                    av_train_acc = 0
                    total_train_batches = 0
                #else:
                    #print("Epoch:", i, "Iter:", graph_idx, ", Avg loss:", av_loss / avg_freq, "Avg_loss_X:",av_loss_X/avg_freq,"Avg_loss_data:",av_loss_data/avg_freq,   flush=True)
                av_loss = 0
                av_loss_data = 0
                av_loss_X = 0

        avg_loss = total_loss / number_all_batches 
        avg_loss_X = total_loss_X / number_all_batches
        avg_loss_data = total_loss_data / number_all_batches
        avg_acc  = total_acc / number_all_batches  

        # if args.classify != 1:
        #     return net, avg_loss, avg_acc.item(), avg_loss_X, avg_loss_data
            
        # else:
        #     return net, avg_loss, avg_acc.item()
        
        return net, avg_loss, avg_acc.item(), avg_loss_X, avg_loss_data


    def eval(net, loader, forward_op=None, split='Test'):

        if isinstance(forward_op, list) and len(forward_op) > 1:  # several test tasks: report the mean over them
            res = [eval(net, loader, [op], split) for op in forward_op]
            return (net,) + tuple(float(np.mean([r[k] for r in res])) for k in range(1, 5))
        if(isinstance(forward_op, list)):
            print(f"{split} task:", forward_op[0][1], end=" ")
            op_name = forward_op[0][1]
            forward_op = forward_op[0][0]
            if hasattr(net, 'set_task'):
                net.set_task(op_name)

        net.eval()
        av_test_loss = 0
        av_test_loss_X = 0
        av_test_loss_data = 0
        total_acc = 0
    
        total_test_batches = 0
        for test_idx, graph in enumerate(loader):
            
            with torch.no_grad():
                actual_bs = len(graph.batch.unique())
                total_test_batches += actual_bs
                graph = process_data(args, graph)
                graph, forward_op =  task_specific_modifiers(graph, args, forward_op, test_dataset)

                graph = graph.to(device)
                if optimizer is not None:
                    optimizer.zero_grad()

                clean_forward_data = forward_op(graph.y, graph.edge_index, graph.edge_weight, emb=False)
                forward_data = clean_forward_data
                if hasattr(forward_op, 'corrupt'):
                    forward_data = forward_op.corrupt(clean_forward_data)
            if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
                X = net(forward_data, graph)
            else:
                with torch.set_grad_enabled(args.task == 'edgeRecovery'):  # no activations kept (Windmill would not fit)
                    X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
            
            with torch.no_grad():
                forward_data_rec = forward_op(X, graph.edge_index, graph.edge_weight, emb=False)
            
                if args.classify:
                    loss_X = F.cross_entropy(X, graph.y) 
                    loss_data = F.cross_entropy(forward_data_rec, clean_forward_data)
                    loss = loss_X + loss_data
                    pred = torch.argmax(X, dim=-1)
                    acc = torch.eq(pred, torch.argmax(graph.y, dim=-1)).sum() / graph.x.shape[0]

                else:
                    # regression
                    loss_X = F.mse_loss(X, graph.y)/F.mse_loss(torch.zeros_like(graph.y), graph.y)
                    loss_data = F.mse_loss(forward_data_rec, clean_forward_data) / F.mse_loss(torch.zeros_like(clean_forward_data), clean_forward_data)
                    loss = 0.5 * (loss_X + loss_data)
                    acc = torch.tensor([0])
                    
                # av_loss += loss.item()
                av_test_loss += loss.item() * actual_bs
                av_test_loss_X += loss_X.item() * actual_bs
                av_test_loss_data += loss_data.item() * actual_bs
                total_acc += acc * actual_bs

        test_acc = (total_acc / total_test_batches) if args.classify else torch.tensor([0])
        test_loss = av_test_loss / total_test_batches
        test_loss_X = av_test_loss_X / total_test_batches
        test_loss_data = av_test_loss_data / total_test_batches

        if args.cluster == 0:
            if args.classify:
                print("Iter: ", test_idx, f"{split} loss:", test_loss, ", acc:", test_acc,
                        flush=True)
            else:
                print("Iter: ", test_idx, f"{split} loss:", test_loss, flush=True)

        # if args.classify != 1:
        #     return net, test_loss, test_acc.item(), test_loss_X, test_loss_data
        # else:
        #     return net, test_loss, test_acc.item()
        return net, test_loss, test_acc.item(), test_loss_X, test_loss_data


    ##############################
    max_patience = args.max_patience #35 #50
    curr_patience = 0 
    ##############################
    best_test_acc = 0
    best_test_loss = 1000000
    best_test_loss_corr_X_loss = 1000000
    best_test_loss_corr_data_loss = 1000000
    best_state_dict = None
    best_score = 0 if args.classify else 1e6  # selection score (val, or test if no val split); lower is better

    for i in tqdm(range(niters + args.head_epochs + args.test_head_epochs)):
        if args.method == 'foundation' and i in (niters, niters + args.head_epochs):
            # end of a phase: keep its best-validation state, then select afresh within the next phase
            print(f"[phase end, epoch {i}] best validation score {best_score:.4f} (compare across phases: do the heads help?)")
            if best_state_dict is not None:
                net.load_state_dict(best_state_dict)
            best_state_dict, best_score, curr_patience = None, (0 if args.classify else 1e6), 0
        if args.method == 'foundation':
            if i == niters:

                save_name = f"no_head_training_{args.project_name}_seed_{seed}" if args.num_seeds > 1 else f"no_head_training_{args.project_name}"
                save_model(net, save_name)
                net.freeze_backbone()
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr/10, weight_decay=args.wd)
            if i == niters + args.head_epochs:
                 forward_op = test_forward_op

        if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
            # no need for training when there are no learnable parameters
            train_loss = train_acc = train_loss_X = train_loss_data = 0 #temporary 
        else:
            net, train_loss, train_acc, train_loss_X, train_loss_data = train(net, epoch=i, forward_op=forward_op, loader=train_loader)

        # Evaluate and log test metrics every 1 iteration
        if i % 1 == 0:
            net, test_loss, test_acc, test_loss_X, test_loss_data = eval(net, test_loader, forward_op=test_forward_op)
            # select on validation (current training tasks); the test numbers are only reported at the selected epoch
            if val_loader is not None:
                _, val_loss, val_acc, _, _ = eval(net, val_loader, forward_op=forward_op, split='Val')
            else:
                val_loss, val_acc = test_loss, test_acc
            score = -val_acc if args.classify else val_loss
            if (best_score - score) > abs(best_score * (0.005 if args.classify else 0.01)):
                best_score = score
                best_test_acc, best_test_loss = test_acc, test_loss
                best_test_loss_corr_data_loss = test_loss_data
                best_test_loss_corr_X_loss = test_loss_X
                curr_patience = 0
                best_state_dict = copy.deepcopy(net.state_dict())
            else:
                curr_patience += 1

            metrics = {
                f"best_test_loss_{seed}": best_test_loss,
                f"best_test_acc_{seed}": best_test_acc if args.classify else 0,
                f"test_acc_{seed}": test_acc if args.classify else 0,
                f"test_loss_{seed}": test_loss,
                f"val_loss_{seed}": val_loss,
                f"train_acc_{seed}": train_acc if args.classify else 0,
                f"train_loss_{seed}": train_loss,
                f"train_loss_X_{seed}": train_loss_X,
                f"train_loss_data_{seed}": train_loss_data,
                f"best_test_loss_or_acc_corr_data_loss_{seed}":best_test_loss_corr_data_loss,
                f"best_test_loss_or_acc_corr_X_loss_{seed}":best_test_loss_corr_X_loss,
                f"epoch_{seed}":i
            }
            wandb.log(metrics)


        else:

            metrics = {
                f"train_acc_{seed}": train_acc if args.classify else 0,
                f"train_loss_{seed}": train_loss,
                f"train_loss_X_{seed}": train_loss_X,
                f"train_loss_data_{seed}": train_loss_data,
                f"best_test_loss_{seed}": best_test_loss,
                f"best_test_loss_or_acc_corr_data_loss_{seed}":best_test_loss_corr_data_loss,
                f"best_test_loss_or_acc_corr_X_loss_{seed}":best_test_loss_corr_X_loss,
                f"epoch_{seed}":i
            }
            wandb.log(metrics)
        
        if curr_patience > max_patience:
            break
    print(f"[phase end, final] best validation score {best_score:.4f}, test loss at that epoch {best_test_loss:.4f}")
    if best_state_dict is not None:  # final model = best-validation state of the last phase
        net.load_state_dict(best_state_dict)
    best_test_losses.append(best_test_loss)
    best_test_accs.append(best_test_acc)
    best_test_losses_corr_X.append(best_test_loss_corr_X_loss)
    best_test_losses_corr_data.append(best_test_loss_corr_data_loss)
    print(f'done with seed {seed}')
    print(f'this run name: {exp_name}')
    save_name = f"{args.project_name}_seed_{seed}" if args.num_seeds > 1 else args.project_name
    save_model(net, save_name)


mean_best_test_loss = np.mean(best_test_losses)
mean_best_test_loss_X = np.mean(best_test_losses_corr_X)
mean_best_test_loss_data = np.mean(best_test_losses_corr_data)

std_best_test_loss = np.std(best_test_losses)
std_best_test_loss_X = np.std(best_test_losses_corr_X)
std_best_test_loss_data = np.std(best_test_losses_corr_data)

mean_best_test_accs = np.mean(best_test_accs)
std_best_test_accs = np.std(best_test_accs)
maximum_best_test_acc = np.max(best_test_accs)

minimum_best_test_loss = np.min(best_test_losses)
minimum_best_test_loss_X = np.min(best_test_losses_corr_X)
minimum_best_test_loss_data = np.min(best_test_losses_corr_data)


metrics = { "minimum_best_test_loss": minimum_best_test_loss,
            "mean_best_test_loss": mean_best_test_loss,
            "std_best_test_loss": std_best_test_loss,
           "maximum_best_test_acc": maximum_best_test_acc if args.classify else 0,
            "mean_best_test_accs": mean_best_test_accs if args.classify else 0,
            "std_best_test_accs": std_best_test_accs if args.classify else 0,
            "mean_best_test_loss_X": mean_best_test_loss_X,
            "mean_best_test_loss_data": mean_best_test_loss_data,
            "minimum_best_test_loss_X": minimum_best_test_loss_X,
            "minimum_best_test_loss_data": minimum_best_test_loss_data,
            "std_best_test_loss_X": std_best_test_loss_X,
            "std_best_test_loss_data": std_best_test_loss_data
        }
wandb.log(metrics)

