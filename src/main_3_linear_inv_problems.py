import os
import sys
import torch
import wandb
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import get_data_and_loaders, forward_pass
from utils import process_data
from utils import task_specific_modifiers
from utils import get_experiment_name
from utils import get_network, get_forward_op
import numpy as np
from utils import count_trainable_parameters, save_model
from utils import str2bool, FLAG_TO_TASK
from utils import (get_data_and_loaders_foundation, foundation_train_epoch,
                   foundation_validate, run_phase, evaluate_all_operators,
                   save_foundation_checkpoint)
import functools
import copy
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
parser.add_argument('--noise', type=str2bool, default=False) # if True, adds noise to the input data.
parser.add_argument('--painting', type=str2bool, default=False) # if True, applies inpainting/masking to the input data.
parser.add_argument('--blurring', type=str2bool, default=False) # if True, applies source localization/deblurring.
parser.add_argument('--sensoring', type=str2bool, default=False) # if True, applies sensor recovery.
parser.add_argument('--pdessm', type=str2bool, default=False) # if True, applies PDE-state reconstruction.
parser.add_argument('--held_out_op', type=str, default=None) # Required for foundation: which operator to hold out
parser.add_argument('--denoising_bypass', type=str2bool, default=True) # If True, bypass CGLS for denoising
args = parser.parse_args()
args.test_batch_size = args.train_batch_size
print(f"{args.noise=}, {args.painting=}, {args.blurring=}, {args.sensoring=}, {args.pdessm=}")

ALL_FLAGS = ['noise', 'painting', 'blurring', 'sensoring', 'pdessm']

# Validate held-out selection for the foundation method
if args.method == 'foundation':
    if args.held_out_op not in ALL_FLAGS:
        raise ValueError(
            f"--held_out_op must be one of {ALL_FLAGS} for foundation, got '{args.held_out_op}'")
    if not getattr(args, args.held_out_op):
        raise ValueError(
            f"--held_out_op '{args.held_out_op}' must also be enabled (its --{args.held_out_op} flag True)")

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


def run_foundation(args, device, seed, exp_name):
    """Foundation training: three phases with validation-only checkpoint
    selection, explicit freezing, held-out adaptation, and matched evaluation.

    Phase 1: pretrain backbone + feat_embed + seen heads (held-out head frozen).
    Phase 2: freeze backbone + feat_embed; train seen heads.
    Phase 3: freeze everything except the held-out head; adapt on a bounded
             support split (first 10% of train) with its own validation slice.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    (train_dataset, val_dataset, test_dataset, train_loader, val_loader,
     test_loader, label_channels, feat_channels, split_info,
     norm_stats) = get_data_and_loaders_foundation(args)

    held_out_flag = args.held_out_op
    held_out_task = FLAG_TO_TASK[held_out_flag]
    seen_flags = [f for f in ALL_FLAGS if getattr(args, f) and f != held_out_flag]

    net = get_network(args, None, args.channels, label_channels, feat_channels, device=device)
    net = net.to(device)
    net.denoising_bypass = args.denoising_bypass
    print(f"[Foundation] held_out={held_out_flag}, seen={seen_flags}, "
          f"denoising_bypass={args.denoising_bypass}")

    def make_optimizer(params, lr):
        return torch.optim.Adam(params, lr=lr, weight_decay=args.wd)

    def set_requires_grad(params_iter, flag):
        for p in params_iter:
            p.requires_grad = flag

    # ---- Phase 1: backbone + feat_embed + seen heads (held-out head frozen) ----
    for p in net.parameters():
        p.requires_grad = True
    # explicitly freeze ONLY the held-out head so it never updates during pretraining;
    # backbone, feat_embed and all seen heads remain trainable.
    for p in net.task_heads[held_out_task].parameters():
        p.requires_grad = False
    held_out_snapshot = copy.deepcopy(net.task_heads[held_out_task].state_dict())

    p1_params = net.get_trainable_params_for_phase(1, held_out_task)
    opt1 = make_optimizer(p1_params, args.lr)
    train_fn1 = functools.partial(_fnd_train, active_flags=seen_flags)
    val_fn1 = functools.partial(_fnd_val, active_flags=seen_flags)
    best_state, p1_val = run_phase(net, train_loader, val_loader, opt1,
                                   max_epochs=args.epochs, patience=args.max_patience,
                                   phase_name='phase1_backbone', device=device, args=args,
                                   process_fn=process_data, metric_fn=None,
                                   train_fn=train_fn1, validate_fn=val_fn1)
    net.load_state_dict(best_state)
    # Verify held-out head unchanged during phase 1
    for k, v in net.task_heads[held_out_task].state_dict().items():
        if not torch.equal(v.cpu(), held_out_snapshot[k].cpu()):
            raise RuntimeError(f"Phase 1 modified held-out head parameter '{k}'")

    # Held-out zero-shot evaluation (before adaptation)
    zero_shot = evaluate_all_operators(net, test_loader, args, device, process_data)
    print(f"[Foundation] zero-shot held-out ({held_out_flag}): "
          f"model={zero_shot[held_out_flag]['model']:.4f}")

    # ---- Phase 2: seen-head training (backbone + feat_embed frozen) ----
    net.freeze_backbone()
    set_requires_grad(net.task_heads[held_out_task].parameters(), False)
    for name, head in net.task_heads.items():
        if name != held_out_task:
            set_requires_grad(head.parameters(), True)
    if args.head_epochs > 0 and seen_flags:
        p2_params = net.get_trainable_params_for_phase(2, held_out_task)
        opt2 = make_optimizer(p2_params, args.lr / 10)
        train_fn2 = functools.partial(_fnd_train, active_flags=seen_flags)
        val_fn2 = functools.partial(_fnd_val, active_flags=seen_flags)
        best_state, p2_val = run_phase(net, train_loader, val_loader, opt2,
                                       max_epochs=args.head_epochs, patience=args.max_patience,
                                       phase_name='phase2_seen_heads', device=device, args=args,
                                       process_fn=process_data, metric_fn=None,
                                       train_fn=train_fn2, validate_fn=val_fn2)
        net.load_state_dict(best_state)

    # ---- Phase 3: held-out adaptation (only held-out head trains) ----
    net.freeze_backbone()
    for name, head in net.task_heads.items():
        set_requires_grad(head.parameters(), name == held_out_task)

    # Bounded adaptation support: first 10% of train; validation: next 5%
    train_list = list(train_dataset)
    n_support = max(1, int(0.10 * len(train_list)))
    n_adapt_val = max(1, int(0.05 * len(train_list)))
    support_list = train_list[:n_support]
    adapt_val_list = train_list[n_support:n_support + n_adapt_val]
    from torch_geometric.loader import DataLoader as _DL
    support_loader = _DL(support_list, batch_size=args.train_batch_size, shuffle=True)
    adapt_val_loader = _DL(adapt_val_list, batch_size=args.test_batch_size, shuffle=False)

    if args.test_head_epochs > 0:
        p3_params = net.get_trainable_params_for_phase(3, held_out_task)
        opt3 = make_optimizer(p3_params, args.lr / 10)
        train_fn3 = functools.partial(_fnd_train, active_flags=[held_out_flag])
        val_fn3 = functools.partial(_fnd_val, active_flags=[held_out_flag])
        best_state, p3_val = run_phase(net, support_loader, adapt_val_loader, opt3,
                                       max_epochs=args.test_head_epochs, patience=args.max_patience,
                                       phase_name='phase3_adaptation', device=device, args=args,
                                       process_fn=process_data, metric_fn=None,
                                       train_fn=train_fn3, validate_fn=val_fn3)
        net.load_state_dict(best_state)

    # Held-out evaluation after adaptation + full evaluation of all operators
    final_eval = evaluate_all_operators(net, test_loader, args, device, process_data)
    print("\n[Foundation] Final evaluation (all operators):")
    for flag in ALL_FLAGS:
        r = final_eval[flag]
        tag = 'held-out' if flag == held_out_flag else ('seen' if r['trained'] else 'off')
        print(f"  {flag:10s} [{tag:8s}] model={r['model']:.4f} "
              f"solver={r['solver']:.4f} X=b={r['xeqb']:.4f} zero={r['zero']:.4f}")

    # Log to wandb
    wandb_metrics = {}
    for flag in ALL_FLAGS:
        for key in ['model', 'solver', 'xeqb', 'zero']:
            wandb_metrics[f"final_{flag}_{key}"] = final_eval[flag][key]
    wandb_metrics[f"zeroshot_{held_out_flag}_model"] = zero_shot[held_out_flag]['model']
    wandb_metrics[f"adapted_{held_out_flag}_model"] = final_eval[held_out_flag]['model']
    wandb.log(wandb_metrics)

    # Save checkpoint with full metadata (preserves denoising_bypass etc.)
    os.makedirs('models', exist_ok=True)
    ckpt_path = os.path.join('models', exp_name + '.pth')
    save_foundation_checkpoint(
        net, ckpt_path,
        held_out_flag=held_out_flag,
        enabled_flags=[f for f in ALL_FLAGS if getattr(args, f)],
        denoising_bypass=args.denoising_bypass,
        split_info=split_info,
        hid_channels=args.channels, label_channels=label_channels,
        niter=args.solveIter, cgls_iter=args.cglsIter,
        normalization_stats=norm_stats)

    return final_eval, zero_shot


# Module-level training/validation callables usable by run_phase (bound via partial)
def _fnd_train(net, loader, optimizer, device, args, process_fn, active_flags):
    return foundation_train_epoch(net, loader, optimizer, device, args, process_fn, active_flags)


def _fnd_val(net, loader, device, args, process_fn, metric_fn, active_flags):
    return foundation_validate(net, loader, device, args, process_fn, active_flags)


if args.method == 'foundation':
    # Dedicated foundation workflow (bypasses the legacy loop below).
    run_foundation(args, device, int(args.seed), exp_name)
    sys.exit(0)

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

    train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels = get_data_and_loaders(args)

    hid_channels = args.channels
    forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test = False) # flag=False means that the forward operator is for training, so we want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    test_forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test=True) # flag=True means that the forward operator is for testing, so we don't want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    net = get_network(args, forward_op, hid_channels, label_channels, feat_channels, device=device)
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

        if(isinstance(forward_op, list)):
            op_name = forward_op[epoch % len(forward_op)][1]
            forward_op = forward_op[epoch % len(forward_op)][0]
            net.set_task(op_name)

        net.train()
        for graph_idx, graph in enumerate(loader):
            
            actual_bs = len(graph.batch.unique())
            total_train_batches += actual_bs
            number_all_batches += actual_bs
            graph = process_data(args, graph)
            graph, forward_op =  task_specific_modifiers(graph, args, forward_op, train_dataset)
            
            graph = graph.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            
            forward_data = forward_op(graph.y, graph.edge_index, graph.edge_weight,
                                        emb=False)  # replace graph.edge_Weight to target edge velocity

            if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
                # print(graph_idx, graph)
                X = net(forward_data, graph)
                # print("went through net")
            else:
                X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
            

            forward_data_rec = forward_op(X, graph.edge_index, graph.edge_weight, emb=False)

            if args.classify: 
                loss_X = F.cross_entropy(X, graph.y) 
                loss_data = F.cross_entropy(forward_data_rec, forward_data)
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
                loss_data = F.mse_loss(forward_data_rec, forward_data) / F.mse_loss(torch.zeros_like(forward_data), forward_data)
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


    def eval(net, loader, forward_op=None):

        if(isinstance(forward_op, list)):
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

                forward_data = forward_op(graph.y, graph.edge_index, graph.edge_weight, emb=False)
        
            if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
                X = net(forward_data, graph)
            else:
                X, Xref, R = net(forward_data, graph.edge_index, graph.edge_weight, graph.x)
            
            with torch.no_grad():
                forward_data_rec = forward_op(X, graph.edge_index, graph.edge_weight, emb=False)
            
                if args.classify:
                    loss_X = F.cross_entropy(X, graph.y) 
                    loss_data = F.cross_entropy(forward_data_rec, forward_data)
                    loss = loss_X + loss_data
                    pred = torch.argmax(X, dim=-1)
                    acc = torch.eq(pred, torch.argmax(graph.y, dim=-1)).sum() / graph.x.shape[0]

                else:
                    # regression
                    loss_X = F.mse_loss(X, graph.y)/F.mse_loss(torch.zeros_like(graph.y), graph.y)
                    loss_data = F.mse_loss(forward_data_rec, forward_data) / F.mse_loss(torch.zeros_like(forward_data), forward_data)
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
                print("Iter: ", test_idx,"Test loss:", test_loss, ", test acc:", test_acc,
                        flush=True)
            else:
                print("Iter: ", test_idx, "Test loss:", test_loss, flush=True)

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
    for i in tqdm(range(niters + args.head_epochs + args.test_head_epochs)):
        # NOTE: foundation is handled entirely by run_foundation() above and never
        # reaches this legacy loop.
        if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
            # no need for training when there are no learnable parameters
            train_loss = train_acc = train_loss_X = train_loss_data = 0 #temporary 
        else:
            net, train_loss, train_acc, train_loss_X, train_loss_data = train(net, epoch=i, forward_op=forward_op, loader=train_loader)

        # Evaluate and log test metrics every 1 iteration
        if i % 1 == 0:
            net, test_loss, test_acc, test_loss_X, test_loss_data = eval(net, test_loader, forward_op=test_forward_op)

            if args.classify==1:
                # best test loss not changed for classification problems.
                if (test_acc-best_test_acc) > (best_test_acc*0.005):
                    best_test_acc = test_acc
                    best_test_loss_corr_data_loss = test_loss_data
                    best_test_loss_corr_X_loss = test_loss_X
                    curr_patience = 0
                else:
                    curr_patience += 1
            else:
                #regression
                if (best_test_loss - test_loss) > abs(best_test_loss*0.01):
                    best_test_loss = test_loss
                    best_test_loss_corr_data_loss = test_loss_data
                    best_test_loss_corr_X_loss = test_loss_X
                    curr_patience = 0
                else:
                    curr_patience += 1

            metrics = {
                f"best_test_loss_{seed}": best_test_loss,
                f"best_test_acc_{seed}": best_test_acc if args.classify else 0,
                f"test_acc_{seed}": test_acc if args.classify else 0,
                f"test_loss_{seed}": test_loss,
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
    best_test_losses.append(best_test_loss)
    best_test_accs.append(best_test_acc)
    best_test_losses_corr_X.append(best_test_loss_corr_X_loss)
    best_test_losses_corr_data.append(best_test_loss_corr_data_loss)
    print(f'done with seed {seed}')
    print(f'this run name: {exp_name}')
    save_model(net, exp_name)
    #compare_operators_and_save_table(net, forward_op, test_forward_op, train_loader, test_loader, args)


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

