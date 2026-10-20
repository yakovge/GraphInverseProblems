import torch
import wandb
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import get_data_and_loaders, forward_pass, process_data, task_specific_modifiers
import copy
from utils import get_network, get_forward_op, get_experiment_name
import numpy as np
import random
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
parser.add_argument('--single_task', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False,
                    help='from-scratch single-task baseline: train and test the one operator whose flag is True')
parser.add_argument('--dc', type=str, default='cgls', choices=['cgls', 'prox']) # data consistency: cgls = fixed CGLS steps, prox = learned per-task lam
parser.add_argument('--heads', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=True) # False: skip both head phases (shared encoder/head only)
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
parser.add_argument('--seed', type=int, default=0)  # run k of --num_seeds uses seed + k
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--noise', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False) # if True, adds noise to the input data. If False, no noise is added.
parser.add_argument('--painting', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False) # if True, applies inpainting/masking to the input data. If False, no inpainting is applied.
parser.add_argument('--blurring', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False) # if True, applies source localization/deblurring to the input data. If False, no deblurring is applied.
parser.add_argument('--sensoring', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False) # if True, applies sensor recovery to the input data. If False, no sensor recovery is applied.
parser.add_argument('--pdessm', type=lambda s: s.lower() in ('1', 'true', 'yes'), default=False) # if True, applies PDE-state reconstruction to the input data. If False, no PDE-state reconstruction is applied.
parser.add_argument('--n_sensors', type=int, default=5, 
                    help='Number of sensor nodes kept per snapshot')
parser.add_argument('--val_frac', type=float, default=0.1) # validation = last val_frac of every training time series (checkpoint selection)
parser.add_argument('--pde_tau', type=float, default=20.0) # PDE diffusion time; tau=1 was trivially invertible (cond ~3)
parser.add_argument('--train_datasets', type=str, default='CPOX,PEDALME,WIKIMATHS,MONTEVIDEO') # only with --dataset MULTI
parser.add_argument('--test_dataset', type=str, default='WINDMILL') # only with --dataset MULTI: held-out dataset, tested on all of it

args = parser.parse_args()
if not args.heads:  # no task-specific heads: drop the head-training and test-head phases
    args.head_epochs = args.test_head_epochs = 0
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
    seed = args.seed + seed_temp  # was `args.seed` for every run: identical runs that overwrote each other's checkpoint
    torch.manual_seed(seed)
    random.seed(seed)      # MultiLoader / mask sampling use python's random
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels, val_loader = get_data_and_loaders(args)
    if val_loader is None:
        print('WARNING: no validation split for this dataset -> checkpoint selection falls back to the TEST set', flush=True)

    hid_channels = args.channels

    sample_graph = next(iter(test_dataset))
    num_nodes = sample_graph.x.shape[0]
    args.num_nodes = num_nodes

    forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test = False) # flag=False means that the forward operator is for training, so we want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    test_forward_op = get_forward_op(args, hid_channels, label_channels, device=device, test=True) or forward_op # all tasks trained (leave-one-dataset-out) -> test every task. flag=True means that the forward operator is for testing, so we don't want to apply any noise, masking, blurring, sensor recovery, or PDE-state reconstruction to the input data.
    if args.single_task:  # from-scratch single-task baseline: the test operator is the (single) trained operator
        assert len(forward_op) == 1, 'set exactly one task flag with --single_task'
        test_forward_op = forward_op
    net = get_network(args, forward_op if args.single_task else forward_op + test_forward_op, hid_channels, label_channels, feat_channels, device=device)
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

        if isinstance(forward_op, list) and len(forward_op) > 1:  # several tasks: report the mean over them
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
    # Checkpoint selection and early stopping use the VALIDATION split (last --val_frac of the training
    # series) with the tasks being trained in the current phase. The held-out test split (held-out task in
    # script.py, held-out dataset in script2.py) is evaluated once at the end of every phase, with the
    # selected checkpoint. Previously the best epoch was chosen on the test loss itself.
    ##############################
    max_patience = args.max_patience
    curr_patience = 0
    best_score = None        # validation selection score (lower is better): loss, or -accuracy
    best_state_dict = None
    sel_loader = val_loader if val_loader is not None else test_loader
    test_reports = {}        # phase name -> test metrics of the selected checkpoint
    total_epochs = niters + args.head_epochs + args.test_head_epochs

    def finish_phase(name):
        """Restore the phase's best-validation state, evaluate the test split ONCE, reset the selection."""
        global best_state_dict, best_score, curr_patience
        if best_state_dict is not None:
            net.load_state_dict(best_state_dict)
        _, t_loss, t_acc, t_X, t_data = eval(net, test_loader, forward_op=test_forward_op, split='Test')
        test_reports[name] = {'loss': t_loss, 'acc': t_acc, 'loss_X': t_X, 'loss_data': t_data, 'val_score': best_score}
        print(f"[phase '{name}' done, seed {seed}] selected on validation (score {best_score}) -> TEST loss {t_loss:.4f} "
              f"(loss_X {t_X:.4f}, loss_data {t_data:.4f})", flush=True)
        wandb.log({f"test_loss_{name}_{seed}": t_loss, f"test_loss_X_{name}_{seed}": t_X,
                   f"test_loss_data_{name}_{seed}": t_data, f"test_acc_{name}_{seed}": t_acc if args.classify else 0,
                   f"val_score_{name}_{seed}": best_score if best_score is not None else float('nan')})
        best_state_dict, best_score, curr_patience = None, None, 0

    phase = 'pretrain' if args.method == 'foundation' else 'final'
    i = 0
    pbar = tqdm(total=total_epochs)
    while i < total_epochs:
        if args.method == 'foundation':
            if i == niters:
                finish_phase('pretrain')  # zero-shot: best-validation backbone with the shared head
                save_name = f"no_head_training_{args.project_name}_seed_{seed}" if args.num_seeds > 1 else f"no_head_training_{args.project_name}"
                save_model(net, save_name)
                net.freeze_backbone()
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr/10, weight_decay=args.wd)
                phase = 'heads'
            if i == niters + args.head_epochs:
                if args.head_epochs > 0:
                    finish_phase('heads')  # task heads trained on the training tasks, backbone frozen
                forward_op = test_forward_op  # supervised head adaptation on the held-out task (training snapshots)
                phase = 'adapt'

        if args.method == 'laplacian_regularization' or args.method == 'tikhonov_regularization' or args.method=='laplacian_explicit':
            # no need for training when there are no learnable parameters
            train_loss = train_acc = train_loss_X = train_loss_data = 0 #temporary
        else:
            net, train_loss, train_acc, train_loss_X, train_loss_data = train(net, epoch=i, forward_op=forward_op, loader=train_loader)

        # validation on the tasks of this phase drives selection; the test split is not touched here
        net, val_loss, val_acc, val_loss_X, val_loss_data = eval(net, sel_loader, forward_op=forward_op, split='Val')
        score = -val_acc if args.classify else val_loss
        tol = 0.005 if args.classify else 0.01
        if best_score is None or (best_score - score) > abs(best_score * tol):
            best_score = score
            curr_patience = 0
            best_state_dict = copy.deepcopy(net.state_dict())
        else:
            curr_patience += 1

        wandb.log({
            f"train_acc_{seed}": train_acc if args.classify else 0,
            f"train_loss_{seed}": train_loss,
            f"train_loss_X_{seed}": train_loss_X,
            f"train_loss_data_{seed}": train_loss_data,
            f"val_loss_{seed}": val_loss,
            f"val_loss_X_{seed}": val_loss_X,
            f"val_loss_data_{seed}": val_loss_data,
            f"val_acc_{seed}": val_acc if args.classify else 0,
            f"best_val_score_{seed}": best_score,
            f"epoch_{seed}": i,
        })

        i += 1
        pbar.update(1)
        if curr_patience > max_patience:  # early stop THIS phase, continue with the next one
            nxt = {'pretrain': niters, 'heads': niters + args.head_epochs}.get(phase, total_epochs)
            print(f"[early stop] phase '{phase}' at epoch {i}: no validation improvement for {max_patience} epochs", flush=True)
            pbar.update(max(0, nxt - i))
            i = max(i, nxt)
    pbar.close()
    finish_phase(phase)  # the last phase ('adapt' / 'heads' / 'pretrain' with --heads False, or 'final')

    final = test_reports[phase]
    best_test_losses.append(final['loss'])
    best_test_accs.append(final['acc'])
    best_test_losses_corr_X.append(final['loss_X'])
    best_test_losses_corr_data.append(final['loss_data'])
    print('test loss per phase (selected checkpoints):', {k: round(v['loss'], 4) for k, v in test_reports.items()}, flush=True)
    if hasattr(net, 'dc_lam'):
        print('learned data-consistency lam per task:', {t: round(torch.exp(20 * p).item(), 4) for t, p in net.dc_lam.items()})
    print(f'done with seed {seed}')
    print(f'this run name: {exp_name}')
    save_name = f"{args.project_name}_seed_{seed}" if args.num_seeds > 1 else args.project_name
    if args.method == 'foundation' and not args.heads:
        save_name = "no_head_training_" + save_name  # plot_2/plot_fixed read this prefix as "shared head"
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

