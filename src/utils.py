import os
import copy
import argparse
import torch
from torch_geometric.datasets.gnn_benchmark_dataset import GNNBenchmarkDataset
from torch_geometric_temporal.signal import temporal_signal_split

from customMETRLA import METRLADatasetLoader
from pygt_dataloader import DataLoader as BatchDataLoader
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from graphForwardOps import graph_smooth, graphMask, graphPath, graph_edgeRecovery, AddNoise, SensorRecovery, PDESSM
import networks
from customCPOX import ChickenpoxDatasetLoader
import math
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant


# Task/Flag mapping
FLAG_TO_TASK = {
    'noise': 'denoising',
    'painting': 'inpainting',
    'blurring': 'source_localization',
    'sensoring': 'sensor_recovery',
    'pdessm': 'pde_reconstruction'
}
TASK_TO_FLAG = {v: k for k, v in FLAG_TO_TASK.items()}


def str2bool(v):
    """Parse boolean arguments from command line."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def TSVD_recovery(A, data):
     U,S,V =  torch.linalg.svd(A)
     Z = U.t() @ data
     S_inv = S / (S**2 + 1e-3) 
     Z = Z * S_inv.unsqueeze(1) 
     out = V @ Z
     return out 

    
def get_experiment_name(args, time_):
    if args.method == 'foundation':
        return args.project_name
    if args.task == 'mask':
            if args.classify == 1:
                # classification task
                exp_name = args.task + '_mask_per_class_budget_' + str(args.mask_per_class_budget) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
                args.channels) + '_cglsIter_' + str(
                args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
            else:
                # regression task
                exp_name = args.task + '_mask_per_snapshot_budget_' + str(args.mask_per_snapshot_budget) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
                args.channels) + '_cglsIter_' + str(
                args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_

    elif args.task == 'deblur':
            exp_name = args.task + '_blurCount_' + args.blur_count + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_

    elif args.task == 'path':
            exp_name = args.task + '_pathLength_' + str(args.pathLength) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
         
    else:
        exp_name = args.task + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
    return exp_name

def get_forward_op(args, hid_channels, label_channels, device, test=False):

    if args.method in ["tikhonov_regularization", "laplacian_regularization", "laplacian_explicit"]:
        learn_embedding = False
    else:
        learn_embedding = True

    # If all new flags are False, run the original unchanged logic
    if not (args.noise or args.painting or args.blurring or args.sensoring or args.pdessm):
        if args.task == 'deblur':
            forward_op = graph_smooth(nin=label_channels, embdsize=hid_channels, learnEmb=learn_embedding, device=device, k=int(args.blur_count))
        elif args.task == 'mask':
            forward_op = graphMask(nin=label_channels, ind=torch.arange(25),
                            embdsize=hid_channels, learnEmb=learn_embedding, device=device)
        elif args.task == 'path':
            forward_op = graphPath(embdsize=hid_channels, nin=label_channels, learnEmb=learn_embedding, device=device,
                            pathLength=args.pathLength)
        elif args.task == 'edgeRecovery':
            forward_op = graph_edgeRecovery(nin=label_channels, embdsize=hid_channels, learnEmb=learn_embedding, K=3, device=device)
            
        return forward_op

    # Otherwise, build the forward operator chain
    ops = []
    
    #append any requested augmentations
    if args.pdessm is not test:
        ops.append([PDESSM(nin=label_channels, embdsize=hid_channels, dim=32, 
                          learnEmb=(learn_embedding and len(ops)==0), device=device), 'pde_reconstruction'])
                          
    if args.blurring is not test:
        ops.append([graph_smooth(nin=label_channels, embdsize=hid_channels, 
                                k=int(args.blur_count), 
                                learnEmb=(learn_embedding and len(ops)==0), device=device), 'source_localization'])
                                
    if args.painting is not test:
        ops.append([graphMask(nin=label_channels, ind=torch.arange(25), 
                             embdsize=hid_channels, 
                             learnEmb=(learn_embedding and len(ops)==0), device=device), 'inpainting'])
                             
    if args.sensoring is not test:
        ops.append([SensorRecovery(sensor_indices=torch.arange(25), nin=label_channels, 
                                  embdsize=hid_channels, 
                                  learnEmb=(learn_embedding and len(ops)==0), device=device), 'sensor_recovery'])
                                  
    if args.noise is not test:
        ops.append([AddNoise(nin=label_channels, embdsize=hid_channels, noise_std=0.1, 
                            learnEmb=(learn_embedding and len(ops)==0), device=device), 'denoising'])

    return ops


def get_network(args, forward_op, hid_channels, label_channels, feat_channels, device):
    
    # feat_channels = number of dimensions of the feature vector for a sample
    # label_channels = number of dimensions of the target vector for a sample
    if args.method == 'drip':
        if args.regnet == 'LA':
            reg_model = networks.graphLeastActionNet(nlayers=args.layers, nchanels=hid_channels, nfixPointIter=2, imsize=10).to(
                device)  
        elif args.regnet == 'hyper':
            reg_model = networks.graphHyperResNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                        dropout=args.dropout)
            if args.task == 'edgeRecovery':
                reg_model = networks.graphHyperResNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                            dropout=args.dropout)

        if args.task == 'edgeRecovery':
            proj_model = networks.edge_recovery_proj(forward_op, mu=args.mu)
        else:
            proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)

        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)

        net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
                                            input_feat_dim=feat_channels, rnfPE=args.rnfPE,
                                            task=args.task, learn_emb=True)
        num_params_total_net = count_trainable_parameters(net)

    elif args.method == 'pgd':
        args.cglsIter = 1
        reg_model = networks.graphResNetFO(num_layers=args.layers, nopen=hid_channels, nfeatures=label_channels,
                                dropout=args.dropout)
        proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)

        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)
        net = networks.graph_NeuralProximalGradient(reg_model, proj_model, forward_op, niter=args.solveIter,
                                                    input_feat_dim=feat_channels, rnfPE=args.rnfPE)
        num_params_total_net = count_trainable_parameters(net)
    
    elif args.method == "tikhonov_regularization":
        #  reg_model = None
        #  proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)
        #  net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
        #                                     input_feat_dim=feat_channels, rnfPE=args.rnfPE,
        #                                     task=args.task, learn_emb=False)
        net = networks.Laplace_noReg_Net(forward_op, args, reg='tikhonov_regularization', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model
    elif args.method == 'laplacian_regularization':
            
        net = networks.Laplace_noReg_Net(forward_op, args, reg='laplacian_regularization', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model
    elif args.method == 'inv_scale_space':
       
        reg_model = networks.graphScaleSpaceNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                        dropout=args.dropout)
        proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)
        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)
        net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
                                            input_feat_dim=feat_channels, rnfPE=args.rnfPE,
                                            task=args.task, learn_emb=True)
        num_params_total_net = count_trainable_parameters(net)

    elif args.method == 'laplacian_explicit':
        net = networks.Laplace_noReg_Net(forward_op, args, reg='laplacian_explicit', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model
        
    elif args.method == 'foundation':
        # Instantiate the unified foundation model
        net = networks.GraphInverseFoundationModel(
            num_layers=args.layers,
            hid_channels=hid_channels,
            input_feat_dim=feat_channels,
            label_channels=label_channels, # Pass label_channels here
            niter=args.solveIter,
            cgls_iter=args.cglsIter,
            device=device,
            blur_k=int(getattr(args, 'blur_count', 4))
        )
        
        # Map the existing args.task terminology to the foundation model's dictionary keys
        task_mapping = {
            'deblur': 'source_localization',
            'mask': 'inpainting',
            # Add mapping for new custom flags if necessary, else it uses args.task directly
        }
        
        mapped_task = task_mapping.get(args.task, args.task)
        try:
            net.set_task(mapped_task)
        except ValueError as e:
            print(f"Warning: {e}")
            
        # Count parameters matching the existing script's logging logic
        num_params_reg_model = count_trainable_parameters(net.backbone)
        num_params_proj_model = count_trainable_parameters(net.task_heads) + count_trainable_parameters(net.solver)
        num_params_total_net = count_trainable_parameters(net)

    else:
        print("Error! Your args.method is not a valid choice!")
    
    print(f"Number of parameters in regnet = {num_params_reg_model}")
    print(f"Number of parameters in proj_model = {num_params_proj_model}")
    print(f"Number of parameters in total net = {num_params_total_net}")
    return net


def get_fractional_dataset(dataset, fraction):
    num_samples = len(list(dataset))
    num_samples_to_use = math.ceil(fraction * num_samples)
    
    # Randomly select num_samples_to_use samples from the dataset
    selected_indices = random.sample(range(num_samples), num_samples_to_use)
    
    # Create a subset of the dataset using the selected indices
    fractional_dataset = [dataset[i] for i in selected_indices]
    
    return fractional_dataset


def get_data_and_loaders(args):
    if args.dataset in ['CLUSTER', 'PATTERN']:
        train_dataset = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='train')
        test_dataset = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='test')
        # Get only the specified fraction of the train dataset
        # train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        label_channels = test_dataset.num_classes if args.classify else 1  #
        feat_channels = test_dataset.num_features

    elif args.dataset == 'CPOX':
        lags = args.CPOX_lags
        loader = ChickenpoxDatasetLoader()
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)
        # Get only the specified fraction of the train dataset
        # train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        
        train_loader = DataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        test_loader = DataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        label_channels = 1 #lags #1
        feat_channels =  1 #just a 1 dimensional target (number of cases) #lags  #1 

    elif 'METRLA' in args.dataset:
        datapath = os.path.join(args.datapath, 'temporal_data')
        datapath = os.path.join(datapath, args.dataset)
        loader = METRLADatasetLoader(raw_data_dir=datapath)

        # METRLADatasetLoader
        # dataset = loader.get_dataset(num_timesteps_in=1, num_timesteps_out=0)  # args.pred #1 time step for training
        dataset = loader.get_dataset(num_timesteps_in=1, num_timesteps_out=0) # 3 time steps for training
        # train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
        number_of_train_snapshots = int(0.7 * dataset.snapshot_count)
        number_of_val_snapshots = int(0.8 * dataset.snapshot_count)

        # batch_of_train_snapshots = args.train_batch_size  #5   #int(0.01 * dataset.snapshot_count) #1
        # val_snapshots = int(0.2 * dataset.snapshot_count)

        train_dataset = dataset[0:number_of_train_snapshots] 
        # Get only the specified fraction of the train dataset
        train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        # val_dataset = dataset[number_of_train_snapshots:number_of_val_snapshots]
        test_dataset = dataset[number_of_val_snapshots:]
        test_dataset = get_fractional_dataset(test_dataset, args.test_frac)
        train_loader = BatchDataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        test_loader = BatchDataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        # val_loader = BatchDataLoader(list(val_dataset), batch_size=args.batch_size, shuffle=False)
        label_channels = dataset.num_classes if args.classify else 1  #
        feat_channels = 20

    elif 'SHAPENET' in args.dataset:
        category = None  # Pass in `None` to train on all categories.
        path = args.datapath
        path = path + 'ShapeNet'
        fixed_points_transform = T.FixedPoints(1024, replace=False)
        #################################
        transform = T.Compose([
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
            fixed_points_transform,
            T.NormalizeScale(),
            T.KNNGraph(k=10,num_workers=16)
            ])
        pre_transform = None
        train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                                pre_transform=pre_transform)
        test_dataset = ShapeNet(path, category, split='test', transform=transform,
                                pre_transform=pre_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False,
                                num_workers=6)
        label_channels = 50 #1
        feat_channels  = 6+16  #normal vectors (3), position vectors (3), and one hot encoded categories(16)

    return train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels


def foundation_temporal_splits(total, train_frac=0.8, val_frac=0.1, gap=1):
    """Contiguous chronological train/val/test index ranges with a `gap`-snapshot
    buffer between partitions (no leakage). Returns three (start, end) tuples."""
    train_end = int(train_frac * total)
    val_start = train_end + gap
    val_end = val_start + int(val_frac * total)
    test_start = val_end + gap
    return (0, train_end), (val_start, val_end), (test_start, total)


class DeterministicFixedPoints(object):
    """Deterministic point subsampling for reproducible evaluation: take the first
    `num` points (or all if fewer). Unlike T.FixedPoints, no RNG is involved, so
    repeated evaluation yields identical graph realizations."""
    def __init__(self, num):
        self.num = num

    def __call__(self, data):
        n = data.num_nodes
        k = min(self.num, n)
        idx = torch.arange(k)
        for key, value in list(data):
            if torch.is_tensor(value) and value.size(0) == n:
                data[key] = value[idx]
        data.num_nodes = k
        return data


def get_data_and_loaders_foundation(args):
    """Foundation-model data loading with disjoint train/validation/test splits.

    Returns: train_dataset, val_dataset, test_dataset, train_loader, val_loader,
             test_loader, label_channels, feat_channels, split_info, norm_stats

    Temporal datasets (CPOX, METRLA) use contiguous chronological splits with a
    one-snapshot gap between partitions to avoid leakage. Preprocessing statistics
    (METRLA) are computed on the training portion only.
    """
    norm_stats = None

    if args.dataset == 'CPOX':
        if int(args.CPOX_lags) != 1:
            raise ValueError("Foundation CPOX loader supports lags=1 only.")
        loader = ChickenpoxDatasetLoader()
        dataset = loader.get_dataset(lags=1)
        total = dataset.snapshot_count
        (tr0, tr1), (v0, v1), (te0, te1) = foundation_temporal_splits(total, 0.8, 0.1, gap=1)
        train_dataset = dataset[tr0:tr1]
        val_dataset = dataset[v0:v1]
        test_dataset = dataset[te0:te1]
        split_info = {'train': [tr0, tr1], 'val': [v0, v1], 'test': [te0, te1]}
        train_loader = DataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(list(val_dataset), batch_size=args.test_batch_size, shuffle=False)
        test_loader = DataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        label_channels = 1
        feat_channels = 1

    elif 'METRLA' in args.dataset:
        datapath = os.path.join(args.datapath, 'temporal_data')
        datapath = os.path.join(datapath, args.dataset)
        # Train-only normalization: statistics computed on the first 70% of steps.
        loader = METRLADatasetLoader(raw_data_dir=datapath, train_fraction=0.7)
        dataset = loader.get_dataset(num_timesteps_in=1, num_timesteps_out=0)
        norm_stats = loader.get_normalization_stats()
        total = dataset.snapshot_count
        (tr0, tr1), (v0, v1), (te0, te1) = foundation_temporal_splits(total, 0.7, 0.1, gap=1)
        train_dataset = dataset[tr0:tr1]
        val_dataset = dataset[v0:v1]
        test_dataset = dataset[te0:te1]
        split_info = {'train': [tr0, tr1], 'val': [v0, v1], 'test': [te0, te1]}
        train_loader = BatchDataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        val_loader = BatchDataLoader(list(val_dataset), batch_size=args.test_batch_size, shuffle=False)
        test_loader = BatchDataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        label_channels = 1
        feat_channels = 20

    elif args.dataset in ['CLUSTER', 'PATTERN']:
        # Preserve existing interface; carve a validation split from train.
        train_full = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='train')
        test_dataset = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='test')
        n_total = len(train_full)
        n_val = max(1, int(0.1 * n_total))
        val_dataset = train_full[:n_val]
        train_dataset = train_full[n_val:]
        split_info = {'train': [n_val, n_total], 'val': [0, n_val], 'test': ['dataset_test_split']}
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        label_channels = test_dataset.num_classes if args.classify else 1
        feat_channels = test_dataset.num_features

    elif 'SHAPENET' in args.dataset:
        category = None
        path = args.datapath + 'ShapeNet'
        # Training transform: augmented (random). Evaluation transform: deterministic,
        # so zero-shot and adapted evaluations see identical graph realizations.
        train_transform = T.Compose([
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
            T.FixedPoints(1024, replace=False),
            T.NormalizeScale(),
            T.KNNGraph(k=10, num_workers=16)
        ])
        eval_transform = T.Compose([
            DeterministicFixedPoints(1024),
            T.NormalizeScale(),
            T.KNNGraph(k=10, num_workers=16)
        ])
        train_full = ShapeNet(path, category, split='trainval', transform=train_transform)
        # Validation uses the deterministic eval transform (same underlying samples).
        val_full = ShapeNet(path, category, split='trainval', transform=eval_transform)
        test_dataset = ShapeNet(path, category, split='test', transform=eval_transform)
        n_total = len(train_full)
        n_val = max(1, int(0.1 * n_total))
        val_dataset = val_full[:n_val]
        train_dataset = train_full[n_val:]
        split_info = {'train': [n_val, n_total], 'val': [0, n_val], 'test': ['dataset_test_split']}
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6)
        label_channels = 50
        feat_channels = 6 + 16

    else:
        raise ValueError(f"Unknown dataset for foundation loader: {args.dataset}")

    return (train_dataset, val_dataset, test_dataset, train_loader, val_loader,
            test_loader, label_channels, feat_channels, split_info, norm_stats)


def foundation_train_epoch(net, loader, optimizer, device, args, process_fn, active_flags, seed_base=0):
    """One training epoch cycling through active operators, per-graph measurements."""
    net.train()
    total, count = 0.0, 0
    for bidx, raw in enumerate(loader):
        graph = process_fn(args, raw).to(device)
        for flag in active_flags:
            net.set_task(FLAG_TO_TASK[flag])
            seed = seed_base + bidx
            config = sample_operator_config(graph, args, flag, seed=seed)
            apply_config(net.current_forward_op, config)
            b = generate_measurement(net.current_forward_op, graph.y, graph.edge_index,
                                     graph.edge_weight, graph.batch, seed=seed)
            optimizer.zero_grad()
            pred, _, _ = net(b, graph.edge_index, graph.edge_weight, graph.x, batch=graph.batch)
            loss = compute_loss(pred, graph.y, graph.batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            count += 1
    return total / max(count, 1)


def accumulate_pergraph_ratios(pred, target, batch, eps=1e-8):
    """Return (sum_of_per_graph_relative_MSE, num_graphs) for graph-count weighting."""
    total = 0.0
    n = 0
    for gid in batch.unique():
        mask = (batch == gid)
        se = ((pred[mask] - target[mask]) ** 2).sum()
        energy = (target[mask] ** 2).sum()
        ratio = se if energy < eps else se / energy
        total += ratio.item()
        n += 1
    return total, n


def foundation_validate(net, loader, device, args, process_fn, active_flags, seed_base=100000):
    """Graph-count-weighted mean per-graph relative MSE over active operators.

    Raises on an empty operator selection or empty validation set rather than
    returning a spurious zero loss (which would look like a perfect checkpoint).
    """
    if not active_flags:
        raise ValueError("foundation_validate: empty active-operator selection.")
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for bidx, raw in enumerate(loader):
            graph = process_fn(args, raw).to(device)
            for flag in active_flags:
                net.set_task(FLAG_TO_TASK[flag])
                seed = seed_base + bidx
                config = sample_operator_config(graph, args, flag, seed=seed)
                apply_config(net.current_forward_op, config)
                b = generate_measurement(net.current_forward_op, graph.y, graph.edge_index,
                                         graph.edge_weight, graph.batch, seed=seed)
                pred, _, _ = net(b, graph.edge_index, graph.edge_weight, graph.x, batch=graph.batch)
                s, n = accumulate_pergraph_ratios(pred, graph.y, graph.batch)
                total += s
                count += n
    if count == 0:
        raise ValueError("foundation_validate: empty validation set (no graphs).")
    return total / count


def process_graph_for_shapeNet(args, graph, num_categories):
    if args.use_meta_data==0:
        # Create a tensor of ones with shape [num_nodes, 22]
        graph.x = torch.ones((graph.num_nodes, 22), dtype=torch.float)
        
        # Remove the pos attribute
        if hasattr(graph, 'pos'):
            del graph.pos
        return graph

    # Step 1: Concatenate pos (xyz positions) to x (normal vectors) 
    x = torch.cat([graph.x, graph.pos], dim=1)  # Shape: [num_nodes, 6]
    
    # Step 2: One-hot encode the categories
    category_one_hot = F.one_hot(graph.category, num_classes=num_categories).float()  # Shape: [num_graphs, 16]
    
    # Step 3: Use the batch attribute to map each node to its corresponding category
    category_one_hot_expanded = category_one_hot[graph.batch]  # Shape: [num_nodes, 16]
    
    # Step 4: Concatenate the expanded one-hot encoding to each node's 6D vector
    x = torch.cat([x, category_one_hot_expanded], dim=1)  # Shape: [num_nodes, 22]
    
    # Update the graph's x attribute
    graph.x = x

    # Remove the pos attribute (xyz positions) since it's already been concatenated into graph.x 
    if hasattr(graph, 'pos'):
        del graph.pos
    
    return graph

def sparse_adj_to_edge_index_weight(adj_t):
    # Extract row and col from the sparse adjacency matrix
    row, col, val = adj_t.coo()  # .coo() returns row (source), col (destination) and values of edges for SparseTensor

    # Stack row and col to create edge_index
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = val
    return edge_index, edge_weight

def process_data(args, graph):
    if args.use_meta_data==0:
        constant_transform = Constant(value=1.0, cat=False)
    if 'CPOX' in args.dataset:

        graph.y = graph.x.clone()[:,0] # target are cpox cases
        graph.x = graph.x.clone()[:,1] # features are time indices (weeks), so 1 dimensional feature vector per node
        if len(graph.x.shape) == 1:
            graph.x = graph.x.unsqueeze(-1)
        if len(graph.y.shape) == 1:
            graph.y = graph.y.unsqueeze(-1) 
        if args.use_meta_data == 0:
            graph = constant_transform(graph)

    if 'METRLA' in args.dataset:
        graph.x = graph.x.squeeze()
        # now take only data (speed) dimension (and not the time encoding dimensions) below
        graph.y = graph.x.clone()[:, 0].unsqueeze(-1) 
        graph.x = graph.x[:, 1:]  #Each row has a time encoding. All rows at time "t" have the same value 
        if args.use_meta_data == 0:
            graph.x = torch.full_like(graph.x, 1.0)
    if 'SHAPENET' in args.dataset:
            graph = process_graph_for_shapeNet(args, graph, num_categories=16)
         
    edge_index, _ = remove_self_loops(graph.edge_index)
    graph.edge_index = edge_index
    edge_index, edge_weight = gcn_norm(graph.edge_index, add_self_loops=True)
    graph.edge_index = edge_index
    graph.edge_weight = edge_weight

    if args.classify:
        if 'SHAPENET' in args.dataset:
                graph.y = F.one_hot(graph.y.long(), num_classes=50).float()
        else:
                graph.y = F.one_hot(graph.y.long()).float()
    else:
        graph.y = graph.y.float()

    return graph


def task_specific_modifiers(graph, args, forward_op, dataset):
    if args.task == 'mask' or args.painting:
        if args.classify == 1:
            # mask_budget = n_classes * args.mask_per_class_budget * batch_size 
            n_classes = dataset.num_classes
            total_mask_budget_across_batches_per_class = int(args.train_batch_size*args.mask_per_class_budget)
            mask_indices = []
            for c in range(n_classes):
                class_ind = torch.where(graph.y.argmax(dim=-1) == c)[0]
                sampled_shuffled_class_ind = list(class_ind[torch.randperm(len(class_ind))[:total_mask_budget_across_batches_per_class]])
                mask_indices = mask_indices + sampled_shuffled_class_ind
            mask_indices = torch.Tensor(mask_indices, device=graph.x.device).long()
            rand_mask = mask_indices
        else:
            # regression
            mask_indices = []
            for c in range(args.train_batch_size):
                # args.train_batch_size is batch_of_train_snapshots
                snapshot_ind = torch.where(graph.batch == c)[0]
                sampled_shuffled_class_ind = list(snapshot_ind[torch.randperm(len(snapshot_ind))[:args.mask_per_snapshot_budget]])
                mask_indices = mask_indices + sampled_shuffled_class_ind
            mask_indices = torch.Tensor(mask_indices, device=graph.x.device).long()
            rand_mask = mask_indices
        
        forward_op.ind = rand_mask

    elif args.task == 'path':
        # if epoch == 0:
        forward_op.gen_paths(nnodes=graph.x.shape[0], edge_index=graph.edge_index)
    # elif args.task == 'edgeRecovery':
    #     if epoch == 0:
    #         global rand_edge_weight
    #         rand_edge_weight = torch.rand(graph.edge_index.shape[-1],
    #                                         device=graph.edge_weight.device)  # normalize with D^-1
    #         # torch_geometric.utils.
    #         rand_edge_weight = rand_edge_weight  # / rand_edge_weight.sum(dim=1, keepdim=True)
    #     graph.edge_weight = rand_edge_weight  # graph.edge_attr#rand_edge_weight.clone()
    
    return graph, forward_op


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters())


from graphForwardOps import graphMask, graph_smooth, SensorRecovery, AddNoise, PDESSM
def forward_pass(x, noising=False, painting=False, blurring=False, sensoring=False, pdessm=False, **kwargs):
    """Routes input x through selected forward operations."""
    
    if pdessm:
        pdessm_op = PDESSM(**kwargs.get('pdessm_args', {'dim': x.shape[-1]}))
        x = pdessm_op(x)
        
    if blurring:
        blur_op = graph_smooth(**kwargs.get('blur_args', {}))
        x = blur_op(x)
        
    if painting:
        mask_op = graphMask(**kwargs.get('mask_args', {}))
        x = mask_op(x)
        
    if sensoring:
        sensor_op = SensorRecovery(**kwargs.get('sensor_args', {}))
        x = sensor_op(x)
        
    if noising:
        noise_op = AddNoise(**kwargs.get('noise_args', {}))
        x = noise_op(x)
        
    return x


def save_model(model, name):
    """
    Saves a PyTorch neural network to a 'models' directory.
    """
    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Ensure the name has a standard PyTorch extension
    if not name.endswith(('.pth', '.pt')):
        name += '.pth'

    # Construct the full file path
    file_path = os.path.join('models', name)

    # Save the model's state dictionary
    torch.save(model.state_dict(), file_path)

    print(f"PyTorch model saved successfully at: {file_path}")


def compute_loss(pred, target, batch, eps=1e-8):
    """Per-graph relative MSE. Zero-energy targets use summed squared error."""
    ratios = []
    for gid in batch.unique(sorted=True):
        mask = (batch == gid)
        se = ((pred[mask] - target[mask]) ** 2).sum()
        energy = (target[mask] ** 2).sum()
        if energy < eps:
            ratio = se  # Zero-energy: summed squared error (normalized by 1)
        else:
            ratio = se / energy
        ratios.append(ratio)
    return torch.stack(ratios).mean()


def compute_metric(pred, target, batch, eps=1e-8):
    """Per-graph relative MSE, detached for reporting."""
    with torch.no_grad():
        return compute_loss(pred.detach(), target.detach(), batch, eps).item()


def sample_operator_config(graph, args, flag, seed):
    """Generate reproducible operator configuration from seed.

    Observation budgets are applied PER GRAPH within the batch: each graph gets
    up to `mask_per_snapshot_budget` observed nodes. Returned indices are global
    (into the batched node array); the batched CGLS remaps them per subgraph.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    config = {}
    budget = getattr(args, 'mask_per_snapshot_budget', 16)

    batch = getattr(graph, 'batch', None)
    if batch is None:
        batch = torch.zeros(graph.y.shape[0], dtype=torch.long, device=graph.y.device)

    if flag == 'painting':
        inds = []
        for gid in batch.unique(sorted=True):
            node_ids = (batch == gid).nonzero(as_tuple=True)[0]
            n = node_ids.numel()
            k = min(budget, n)
            perm = torch.randperm(n, generator=gen).to(node_ids.device)
            inds.append(node_ids[perm[:k]])
        config['ind'] = torch.cat(inds) if inds else torch.zeros(0, dtype=torch.long)
    elif flag == 'sensoring':
        # Fixed first-k sensor positions within each graph
        inds = []
        for gid in batch.unique(sorted=True):
            node_ids = (batch == gid).nonzero(as_tuple=True)[0]
            n = node_ids.numel()
            k = min(budget, n)
            inds.append(node_ids[:k])
        config['sensor_indices'] = torch.cat(inds) if inds else torch.zeros(0, dtype=torch.long)
    # Physical-parameter operators (blurring/pdessm) carry their parameters on the
    # operator itself; those are matched between model and baseline at construction.

    return config


def apply_config(op, config):
    """Apply frozen config to operator, matching the operator's device."""
    if 'ind' in config:
        dev = op.ind.device if getattr(op, 'ind', None) is not None else 'cpu'
        op.ind = config['ind'].to(dev)
    if 'sensor_indices' in config:
        dev = op.sensor_indices.device if getattr(op, 'sensor_indices', None) is not None else 'cpu'
        op.sensor_indices = config['sensor_indices'].to(dev)


def generate_measurement(op, y, edge_index, edge_weight, batch, seed):
    """Generate deterministic measurement with batch support."""
    if hasattr(op, 'corrupt'):
        # AddNoise: use corrupt method
        return op.corrupt(y, seed)
    else:
        # Other operators: use forward
        return op.forward(y, edge_index, edge_weight, emb=False, batch=batch)


def create_operator(flag, args, device, learnEmb=False):
    """Create a physical operator (no learned embedding)."""
    hid_channels = getattr(args, 'channels', 32)
    label_channels = 1

    if flag == 'noise':
        return AddNoise(nin=label_channels, embdsize=hid_channels, noise_std=0.1,
                        device=device, learnEmb=learnEmb)
    elif flag == 'painting':
        return graphMask(ind=torch.arange(16), embdsize=hid_channels, nin=label_channels,
                         device=device, learnEmb=learnEmb)
    elif flag == 'blurring':
        blur_count = int(getattr(args, 'blur_count', '4'))
        return graph_smooth(nin=label_channels, embdsize=hid_channels, k=blur_count,
                            device=device, learnEmb=learnEmb)
    elif flag == 'sensoring':
        return SensorRecovery(sensor_indices=torch.arange(16), nin=label_channels,
                              embdsize=hid_channels, device=device, learnEmb=learnEmb)
    elif flag == 'pdessm':
        return PDESSM(nin=label_channels, embdsize=hid_channels, dim=32,
                      device=device, learnEmb=learnEmb)
    else:
        raise ValueError(f"Unknown flag: {flag}")


# Alias key prefixes produced by older models that registered current_forward_op
# and solver.forOp as submodules. They duplicate the active head's weights and are
# dropped on load (the canonical task_heads.* keys carry the real weights).
_LEGACY_ALIAS_PREFIXES = ('current_forward_op.', 'solver.forOp.')


def _normalize_stats_to_lists(stats):
    """Convert normalization statistics to plain Python lists so the checkpoint
    stays loadable under torch.load(weights_only=True) (NumPy arrays are rejected)."""
    if stats is None:
        return None
    out = {}
    for k, v in stats.items():
        if hasattr(v, 'tolist'):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def migrate_state_dict(state):
    """Drop legacy alias keys (current_forward_op.*, solver.forOp.*). The active
    head's canonical weights already live under task_heads.* and are preserved."""
    return {k: v for k, v in state.items()
            if not k.startswith(_LEGACY_ALIAS_PREFIXES)}


def save_foundation_checkpoint(model, path, *, held_out_flag, enabled_flags, denoising_bypass,
                               split_info, hid_channels, label_channels, niter, cgls_iter,
                               normalization_stats=None, blur_k=None):
    """Save foundation model checkpoint with full metadata."""
    ckpt = {
        'model_state_dict': model.state_dict(),
        'held_out_flag': held_out_flag,
        'enabled_flags': enabled_flags,
        'denoising_bypass': denoising_bypass,
        'split_info': split_info,
        'hid_channels': hid_channels,
        'label_channels': label_channels,
        'niter': niter,
        'cgls_iter': cgls_iter,
        'blur_k': blur_k if blur_k is not None else getattr(model, 'blur_k', None),
        # store as lists (weights_only-safe); avoid raw NumPy arrays
        'normalization_stats': _normalize_stats_to_lists(normalization_stats),
        'task_flag_mapping': FLAG_TO_TASK,
        'version': 2
    }
    torch.save(ckpt, path)
    print(f"Foundation checkpoint saved to: {path}")


def load_checkpoint(path, model, device):
    """Load foundation checkpoint with legacy migration, architecture validation."""
    # weights_only=False: our metadata dict is trusted (we wrote it). Stats are
    # stored as lists so this also works under weights_only=True if desired.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state = ckpt.get('model_state_dict', ckpt)
    # Migrate away legacy alias keys before loading.
    state = migrate_state_dict(state)

    # Validate architecture metadata before loading (avoids silent shape mismatch)
    arch_checks = {
        'hid_channels': getattr(model, 'hid_channels', None),
        'label_channels': getattr(model, 'label_channels', None),
        'niter': getattr(model, 'niter', None),
        'cgls_iter': getattr(model, 'cgls_iter', None),
        'blur_k': getattr(model, 'blur_k', None),
    }
    for key, model_val in arch_checks.items():
        ckpt_val = ckpt.get(key, None)
        if ckpt_val is not None and model_val is not None and ckpt_val != model_val:
            raise ValueError(
                f"Architecture mismatch on '{key}': checkpoint={ckpt_val}, model={model_val}")

    missing, unexpected = model.load_state_dict(state, strict=False)

    actual_missing = set(missing)
    actual_unexpected = set(unexpected)

    if actual_missing:
        raise ValueError(f"Missing keys after migration: {actual_missing}")
    if actual_unexpected:
        raise ValueError(f"Unexpected keys after migration: {actual_unexpected}")

    # Restore settings
    model.denoising_bypass = ckpt.get('denoising_bypass', True)

    # Re-point the solver's (non-registered) forward operator to the active head
    model.solver.set_forward_op(model.current_forward_op)

    return ckpt


def load_legacy_state_dict(path, model, device):
    """Load a plain (pre-metadata) state_dict saved by save_model(), applying the
    legacy alias-key migration so it fits the current alias-free architecture."""
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob.get('model_state_dict', blob) if isinstance(blob, dict) else blob
    state = migrate_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) or set(unexpected):
        raise ValueError(
            f"Legacy load mismatch. missing={set(missing)}, unexpected={set(unexpected)}")
    if hasattr(model, 'solver'):
        model.solver.set_forward_op(model.current_forward_op)
    return model


def run_phase(net, train_loader, val_loader, optimizer, max_epochs, patience,
              phase_name, device, args, process_fn, metric_fn, train_fn, validate_fn):
    """Run a training phase with early stopping. Raises on all-nonfinite validation."""

    # Zero-epoch handling: return entry state immediately
    if max_epochs == 0:
        return copy.deepcopy(net.state_dict()), None

    best_loss, best_state = None, None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_fn(net, train_loader, optimizer, device, args, process_fn)
        val_loss = validate_fn(net, val_loader, device, args, process_fn, metric_fn)

        if torch.isfinite(torch.tensor(val_loss)):
            if best_loss is None or val_loss < best_loss * 0.99:
                best_loss = val_loss
                best_state = copy.deepcopy(net.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"Phase '{phase_name}' failed: no finite validation loss in {epoch + 1} epochs")

    return best_state, best_loss


def update_best(val_loss, best_loss, best_state, net):
    """Update best checkpoint if validation improved. Returns (new_loss, new_state, updated)."""
    if not torch.isfinite(torch.tensor(val_loss)):
        return best_loss, best_state, False
    if best_loss is None or val_loss < best_loss * 0.99:
        return val_loss, copy.deepcopy(net.state_dict()), True
    return best_loss, best_state, False


def evaluate_all_operators(model, test_loader, args, device, process_fn):
    """Evaluate ALL five operators with proper aggregation and baselines."""
    model.eval()

    # Accumulate per-graph ratios (not batch means)
    results = {flag: {'model': [], 'solver': [], 'xeqb': [], 'zero': []}
               for flag in ['noise', 'painting', 'blurring', 'sensoring', 'pdessm']}

    for batch_idx, raw_graph in enumerate(test_loader):
        graph = process_fn(args, raw_graph).to(device)

        # Evaluate ALL operators
        for flag in ['noise', 'painting', 'blurring', 'sensoring', 'pdessm']:
            task = FLAG_TO_TASK[flag]

            # Shared config and measurement
            config = sample_operator_config(graph, args, flag, seed=batch_idx)

            model.set_task(task)
            apply_config(model.current_forward_op, config)

            b = generate_measurement(
                model.current_forward_op, graph.y,
                graph.edge_index, graph.edge_weight,
                graph.batch, seed=batch_idx)

            # Model prediction
            with torch.no_grad():
                pred, _, _ = model(b, graph.edge_index, graph.edge_weight,
                                   graph.x, batch=graph.batch)

            # Per-graph ratios
            for gid in graph.batch.unique():
                mask = (graph.batch == gid)
                se_model = ((pred[mask] - graph.y[mask]) ** 2).sum()
                se_xeqb = ((b[mask] - graph.y[mask]) ** 2).sum()
                se_zero = (graph.y[mask] ** 2).sum()
                energy = (graph.y[mask] ** 2).sum()

                if energy < 1e-8:
                    ratio_model = se_model.item()
                    ratio_xeqb = se_xeqb.item()
                    ratio_zero = se_zero.item()
                else:
                    ratio_model = (se_model / energy).item()
                    ratio_xeqb = (se_xeqb / energy).item()
                    ratio_zero = (se_zero / energy).item()

                results[flag]['model'].append(ratio_model)
                results[flag]['xeqb'].append(ratio_xeqb)
                results[flag]['zero'].append(ratio_zero)

            # Solver baseline (same measurement)
            solver_op = create_operator(flag, args, device, learnEmb=False)
            apply_config(solver_op, config)
            solver = networks.graph_CGLS(solver_op, CGLSit=getattr(args, 'cglsIter', 5))

            xref = torch.zeros_like(graph.y)
            with torch.no_grad():
                solver_pred, _ = solver(b, xref, graph.edge_index, graph.edge_weight,
                                        batch=graph.batch, emb=False)

            for gid in graph.batch.unique():
                mask = (graph.batch == gid)
                se_solver = ((solver_pred[mask] - graph.y[mask]) ** 2).sum()
                energy = (graph.y[mask] ** 2).sum()
                if energy < 1e-8:
                    ratio = se_solver.item()
                else:
                    ratio = (se_solver / energy).item()
                results[flag]['solver'].append(ratio)

    # Aggregate by total graph count (mean of individual graph ratios)
    summary = {}
    for flag in results:
        n = len(results[flag]['model'])
        if n == 0:
            summary[flag] = {'model': float('nan'), 'solver': float('nan'),
                            'xeqb': float('nan'), 'zero': float('nan'), 'trained': False}
        else:
            summary[flag] = {
                'model': sum(results[flag]['model']) / n,
                'solver': sum(results[flag]['solver']) / n,
                'xeqb': sum(results[flag]['xeqb']) / n,
                'zero': sum(results[flag]['zero']) / n,
                'trained': getattr(args, flag, False) and flag != getattr(args, 'held_out_op', None)
            }
    return summary


def _run_self_tests():
    """Production self-tests. Returns 0 on success, 1 on failure."""
    import sys
    import tempfile

    torch.manual_seed(42)
    failures = []

    # Dependency check
    try:
        import torch_geometric
        import torch_geometric_temporal
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return 1

    # ========== Test 1: Adjoint with embedding (correct dimensions) ==========
    print("Test 1: Adjoint inner-product with embedding")
    try:
        n, nin, embdsize = 20, 1, 8

        for name, op in [
            ('graphMask', graphMask(ind=torch.arange(10), embdsize=embdsize, nin=nin, device='cpu', learnEmb=True)),
            ('graph_smooth', graph_smooth(nin=nin, embdsize=embdsize, k=2, device='cpu', learnEmb=True)),
            ('AddNoise', AddNoise(nin=nin, embdsize=embdsize, device='cpu', learnEmb=True)),
            ('SensorRecovery', SensorRecovery(sensor_indices=torch.arange(10), nin=nin, embdsize=embdsize, device='cpu', learnEmb=True)),
            ('PDESSM', PDESSM(nin=nin, embdsize=embdsize, dim=32, tau=20.0, device='cpu', learnEmb=True)),
        ]:
            edge_index = torch.randint(0, n, (2, 50))
            edge_weight = torch.ones(50)

            # Domain: [n, embdsize], Codomain: [n, nin]
            x = torch.randn(n, embdsize)
            Ax = op.forward(x, edge_index, edge_weight, emb=True)

            y = torch.randn(n, nin)
            Asy = op.adjoint(y, edge_index, edge_weight, emb=True)

            inner1 = (Ax * y).sum()
            inner2 = (x * Asy).sum()

            if not torch.allclose(inner1, inner2, rtol=1e-3, atol=1e-5):
                failures.append(f"Adjoint {name}: {inner1.item():.6f} != {inner2.item():.6f}")

        print("  PASS" if not any("Adjoint" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 1: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 2: PDESSM batch independence ==========
    print("Test 2: PDESSM batch independence")
    try:
        op = PDESSM(nin=1, embdsize=8, dim=32, tau=20.0, device='cpu', learnEmb=False)

        n1, n2 = 15, 25
        x1, x2 = torch.randn(n1, 1), torch.randn(n2, 1)
        x_batch = torch.cat([x1, x2])
        batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])

        out1 = op.forward(x1, emb=False, batch=None)
        out2 = op.forward(x2, emb=False, batch=None)
        out_batch = op.forward(x_batch, emb=False, batch=batch)

        if not torch.allclose(out_batch[:n1], out1, rtol=1e-5):
            failures.append("PDESSM batch: graph 1 position mismatch")
        if not torch.allclose(out_batch[n1:], out2, rtol=1e-5):
            failures.append("PDESSM batch: graph 2 position mismatch")

        print("  PASS" if not any("PDESSM" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 2: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 3: Mask/sensor batched CGLS (outputs, residuals, restore) ==========
    print("Test 3: Mask/sensor batched CGLS")
    try:
        n1, n2 = 10, 15
        obs_per_graph = 5

        for OpClass, idx_attr in [(graphMask, 'ind'), (SensorRecovery, 'sensor_indices')]:
            x1, x2 = torch.randn(n1, 1), torch.randn(n2, 1)
            edge1 = torch.randint(0, n1, (2, 20))
            edge2 = torch.randint(0, n2, (2, 30))

            local_ind1 = torch.arange(obs_per_graph)
            local_ind2 = torch.arange(obs_per_graph)
            global_ind = torch.cat([local_ind1, n1 + local_ind2])

            def make(idx):
                if OpClass == graphMask:
                    return graphMask(ind=idx, embdsize=8, nin=1, device='cpu', learnEmb=False)
                return SensorRecovery(sensor_indices=idx, nin=1, embdsize=8, device='cpu', learnEmb=False)

            op = make(global_ind)
            op1 = make(local_ind1)
            op2 = make(local_ind2)
            solver = networks.graph_CGLS(op, CGLSit=5, eps=1e-5)
            solver1 = networks.graph_CGLS(op1, CGLSit=5, eps=1e-5)
            solver2 = networks.graph_CGLS(op2, CGLSit=5, eps=1e-5)

            b1 = op1.forward(x1, edge1, torch.ones(20), emb=False)
            out1, res1 = solver1(b1, torch.zeros_like(x1), edge1, torch.ones(20), emb=False)
            b2 = op2.forward(x2, edge2, torch.ones(30), emb=False)
            out2, res2 = solver2(b2, torch.zeros_like(x2), edge2, torch.ones(30), emb=False)

            x_batch = torch.cat([x1, x2])
            edge_batch = torch.cat([edge1, edge2 + n1], dim=1)
            batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])

            b_batch = op.forward(x_batch, edge_batch, torch.ones(50), emb=False)
            out_batch, res_batch = solver(b_batch, torch.zeros_like(x_batch),
                                          edge_batch, torch.ones(50), batch=batch, emb=False)

            if not torch.allclose(out_batch[:n1], out1, rtol=1e-4):
                failures.append(f"{OpClass.__name__} batch: output graph 1 mismatch")
            if not torch.allclose(out_batch[n1:], out2, rtol=1e-4):
                failures.append(f"{OpClass.__name__} batch: output graph 2 mismatch")
            # residuals must also match per subgraph
            if not torch.allclose(res_batch[:n1], res1, rtol=1e-4, atol=1e-5):
                failures.append(f"{OpClass.__name__} batch: residual graph 1 mismatch")
            if not torch.allclose(res_batch[n1:], res2, rtol=1e-4, atol=1e-5):
                failures.append(f"{OpClass.__name__} batch: residual graph 2 mismatch")

            # Exception restoration: force a failure mid-solve, verify indices restored
            original = getattr(op, idx_attr).clone()

            class _Boom(Exception):
                pass

            orig_forward = op.forward
            calls = [0]

            def boom_forward(*a, **k):
                calls[0] += 1
                if calls[0] >= 2:
                    raise _Boom()
                return orig_forward(*a, **k)

            op.forward = boom_forward
            raised = False
            try:
                solver(b_batch, torch.zeros_like(x_batch), edge_batch,
                       torch.ones(50), batch=batch, emb=False)
            except _Boom:
                raised = True
            finally:
                op.forward = orig_forward
            if not raised:
                failures.append(f"{OpClass.__name__}: injected exception not raised")
            if not torch.equal(getattr(op, idx_attr), original):
                failures.append(f"{OpClass.__name__}: indices not restored after exception")

        # Interleaved (non-contiguous) node ordering
        n = 6
        x = torch.randn(n, 1)
        # batch assigns alternating graph ids -> nodes of each graph are non-contiguous
        batch_il = torch.tensor([0, 1, 0, 1, 0, 1])
        edge_il = torch.tensor([[0, 2, 4, 1, 3, 5], [2, 4, 0, 3, 5, 1]])
        w_il = torch.ones(edge_il.shape[1])
        op_il = SensorRecovery(sensor_indices=torch.tensor([0, 2, 4]), nin=1, embdsize=4,
                               device='cpu', learnEmb=False)  # observe all of graph 0
        solver_il = networks.graph_CGLS(op_il, CGLSit=5, eps=1e-5)
        b_il = op_il.forward(x, edge_il, w_il, emb=False)
        out_il, _ = solver_il(b_il, torch.zeros_like(x), edge_il, w_il, batch=batch_il, emb=False)
        # graph 0 nodes (0,2,4) fully observed identity -> recovered exactly
        g0 = torch.tensor([0, 2, 4])
        if not torch.allclose(out_il[g0], x[g0], atol=1e-4):
            failures.append("Interleaved batch: graph-0 recovery wrong")

        print("  PASS" if not any(("batch" in f.lower() or "restored" in f or "Interleaved" in f)
                                   for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 3: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 3b: CGLS zero measurement with nonzero xref ==========
    print("Test 3b: CGLS zero-measurement / nonzero xref")
    try:
        n = 8
        # identity operator (all nodes observed)
        op = SensorRecovery(sensor_indices=torch.arange(n), nin=1, embdsize=4,
                            device='cpu', learnEmb=False)
        solver = networks.graph_CGLS(op, CGLSit=10, eps=1e-6)
        edge = torch.randint(0, n, (2, 12))
        w = torch.ones(12)
        b = torch.zeros(n, 1)          # zero measurement
        xref = torch.ones(n, 1)        # nonzero initial state
        x, r = solver(b, xref, edge, w, emb=False)
        # Must NOT early-return xref unchanged; should drive x toward 0 (A x = b = 0)
        if x.norm() >= xref.norm() * 0.5:
            failures.append(f"Zero-measurement: x not reduced (||x||={x.norm().item():.4f})")
        print("  PASS" if not any("Zero-measurement" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 3b: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 4: Loss gradient flow ==========
    print("Test 4: Loss gradient flow")
    try:
        pred = torch.randn(10, 1, requires_grad=True)
        target = torch.randn(10, 1)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        loss = compute_loss(pred, target, batch)
        if not isinstance(loss, torch.Tensor):
            failures.append("Loss not tensor")

        loss.backward()
        if pred.grad is None or pred.grad.abs().sum() < 1e-10:
            failures.append("Loss no gradient")

        print("  PASS" if not any("Loss" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 4: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 5: Zero-energy uses summed SE ==========
    print("Test 5: Zero-energy convention")
    try:
        pred = torch.tensor([[1.0], [2.0]])
        target = torch.tensor([[0.0], [0.0]])
        batch = torch.tensor([0, 0])

        loss = compute_loss(pred, target, batch)
        expected = 1.0 + 4.0  # Summed SE = 5.0

        if abs(loss.item() - expected) > 0.01:
            failures.append(f"Zero-energy: expected {expected}, got {loss.item()}")

        print("  PASS" if not any("energy" in f.lower() for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 5: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 6: Denoising gated by bypass ==========
    print("Test 6: Denoising bypass gating")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=2, cgls_iter=5, device='cpu')
        model.denoising_bypass = True
        model.set_task('denoising')
        model.eval()

        y = torch.randn(20, 1)
        b = model.current_forward_op.corrupt(y, seed=42)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_weight = torch.ones(50)
        batch = torch.zeros(20, dtype=torch.long)

        with torch.no_grad():
            out_bypass, _, res_bypass = model(b, edge_index, edge_weight, b.clone(), batch=batch)

        # Residual should be D - X
        if not torch.allclose(res_bypass, b - out_bypass, rtol=1e-5):
            failures.append("Denoising bypass: residual != D - X")

        print("  PASS" if not any("Denoising" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 6: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 7: Denoising trainability ==========
    print("Test 7: Denoising learning")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        model.denoising_bypass = True
        model.set_task('denoising')
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        y = torch.randn(20, 1)
        b = model.current_forward_op.corrupt(y, seed=42)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_weight = torch.ones(50)
        batch = torch.zeros(20, dtype=torch.long)

        initial_loss = None
        for step in range(10):
            opt.zero_grad()
            out, _, _ = model(b, edge_index, edge_weight, b.clone(), batch=batch)
            loss = ((out - y) ** 2).mean()
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            opt.step()

        if loss.item() >= initial_loss:
            failures.append(f"Denoising not learning: {initial_loss:.4f} -> {loss.item():.4f}")

        print("  PASS" if not any("learning" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 7: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 8: Phase 1 excludes held-out head ==========
    print("Test 8: Phase 1 held-out exclusion")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        held_out_task = 'denoising'

        # Record held-out head state before training
        held_out_before = {k: v.clone() for k, v in model.task_heads[held_out_task].state_dict().items()}

        # Phase 1: Only backbone + feat_embed + seen heads
        params_phase1 = []
        for name, param in model.named_parameters():
            if f'task_heads.{held_out_task}' not in name:
                params_phase1.append(param)
            else:
                param.requires_grad = False

        opt = torch.optim.Adam(params_phase1, lr=0.1)

        x = torch.randn(20, 1)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_weight = torch.ones(50)

        model.train()
        model.set_task('source_localization')
        for _ in range(5):
            opt.zero_grad()
            out, _, _ = model(x, edge_index, edge_weight, x, batch=torch.zeros(20, dtype=torch.long))
            out.sum().backward()
            opt.step()

        # Verify held-out head unchanged
        for k, v in model.task_heads[held_out_task].state_dict().items():
            if not torch.equal(v, held_out_before[k]):
                failures.append(f"Phase 1: held-out head changed: {k}")

        print("  PASS" if not any("Phase 1" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 8: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 9: run_phase (zero-epoch, non-final best, all-nonfinite) ==========
    print("Test 9: run_phase behavior")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        # -- zero-epoch: return entry state unchanged --
        entry_state = copy.deepcopy(model.state_dict())
        result_state, result_loss = run_phase(
            model, [], [], torch.optim.Adam(model.parameters()),
            max_epochs=0, patience=10, phase_name='zero', device='cpu', args=None,
            process_fn=lambda a, g: g, metric_fn=None,
            train_fn=lambda *a, **k: None, validate_fn=lambda *a, **k: 0.5)
        for k in entry_state:
            if not torch.equal(result_state[k], entry_state[k]):
                failures.append("run_phase zero-epoch: state changed")
                break

        # -- non-final best: each epoch mutates a param; best val is at epoch 1 --
        marker = list(model.backbone.parameters())[0]
        snapshots = {}
        epoch_counter = [0]

        def mut_train(net, loader, optimizer, device, args, process_fn):
            with torch.no_grad():
                marker.add_(1.0)  # make each epoch's state distinct
            snapshots[epoch_counter[0]] = marker.detach().clone()
            epoch_counter[0] += 1

        val_seq = [0.5, 0.3, 0.4, 0.42, 0.44]  # best at epoch index 1

        def seq_val(net, loader, device, args, process_fn, metric_fn):
            return val_seq[min(epoch_counter[0] - 1, len(val_seq) - 1)]

        best_state, best_loss = run_phase(
            model, [1], [1], torch.optim.Adam(model.parameters()),
            max_epochs=5, patience=2, phase_name='nonfinal', device='cpu', args=None,
            process_fn=lambda a, g: g, metric_fn=None,
            train_fn=mut_train, validate_fn=seq_val)
        if abs(best_loss - 0.3) > 1e-9:
            failures.append(f"run_phase best_loss={best_loss}, expected 0.3")
        # best_state's marker (backbone.Kf) must equal the epoch-1 snapshot, not the final epoch
        if 'backbone.Kf' in best_state and not torch.equal(best_state['backbone.Kf'], snapshots[1]):
            failures.append("run_phase: best_state is not the non-final best epoch")

        # -- all-nonfinite: must raise --
        raised = False
        try:
            run_phase(model, [1], [1], torch.optim.Adam(model.parameters()),
                      max_epochs=3, patience=5, phase_name='nonfinite', device='cpu', args=None,
                      process_fn=lambda a, g: g, metric_fn=None,
                      train_fn=lambda *a, **k: None, validate_fn=lambda *a, **k: float('nan'))
        except RuntimeError as e:
            raised = 'no finite validation' in str(e)
        if not raised:
            failures.append("run_phase all-nonfinite did not raise")

        print("  PASS" if not any("run_phase" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 9: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 10: Checkpoint roundtrip ==========
    print("Test 10: Checkpoint roundtrip")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))

        model.denoising_bypass = False

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fpath = f.name

        save_foundation_checkpoint(
            model, fpath,
            held_out_flag='blurring',
            enabled_flags=['noise', 'painting', 'blurring', 'sensoring', 'pdessm'],
            denoising_bypass=False,
            split_info={'train': [0], 'val': [1], 'test': [2]},
            hid_channels=8, label_channels=1, niter=1, cgls_iter=1)

        # Save was done with model on its default active task ('source_localization').
        # Capture per-task predictions from the source model for comparison.
        x = torch.randn(20, 1)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_weight = torch.ones(50)
        batch = torch.zeros(20, dtype=torch.long)
        ref_preds = {}
        for task in model.task_heads.keys():
            model.set_task(task)
            if hasattr(model.current_forward_op, 'ind'):
                model.current_forward_op.ind = torch.arange(10)
            if hasattr(model.current_forward_op, 'sensor_indices'):
                model.current_forward_op.sensor_indices = torch.arange(10)
            with torch.no_grad():
                out, _, _ = model(x, edge_index, edge_weight, x, batch=batch)
            ref_preds[task] = out.clone()

        # Load into a fresh model that is on a DIFFERENT active task (alias-safety):
        model2 = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        model2.denoising_bypass = True
        model2.set_task('denoising')  # deliberately different active task
        ckpt = load_checkpoint(fpath, model2, 'cpu')

        if model2.denoising_bypass != False:
            failures.append("Checkpoint: denoising_bypass not restored")
        if ckpt.get('held_out_flag') != 'blurring':
            failures.append("Checkpoint: held_out_flag wrong")

        # Every head must reproduce the source predictions (no alias overwrite)
        for task in model2.task_heads.keys():
            model2.set_task(task)
            if hasattr(model2.current_forward_op, 'ind'):
                model2.current_forward_op.ind = torch.arange(10)
            if hasattr(model2.current_forward_op, 'sensor_indices'):
                model2.current_forward_op.sensor_indices = torch.arange(10)
            with torch.no_grad():
                out2, _, _ = model2(x, edge_index, edge_weight, x, batch=batch)
            if not torch.allclose(ref_preds[task], out2, rtol=1e-5, atol=1e-6):
                failures.append(f"Checkpoint: task {task} prediction mismatch after cross-task load")

        # Damaged checkpoint (unexpected key) must be rejected
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            dmg = f.name
        torch.save({'model_state_dict': {'bogus.key': torch.zeros(1)}, 'version': 2}, dmg)
        rejected = False
        try:
            load_checkpoint(dmg, model2, 'cpu')
        except (ValueError, RuntimeError):
            rejected = True
        if not rejected:
            failures.append("Checkpoint: damaged checkpoint not rejected")
        os.unlink(dmg)
        os.unlink(fpath)

        print("  PASS" if not any("Checkpoint" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 10: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 11: update_best rejects NaN ==========
    print("Test 11: update_best NaN rejection")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        best_loss, best_state, updated = update_best(float('nan'), None, None, model)
        if updated:
            failures.append("update_best: NaN accepted")

        best_loss, best_state, updated = update_best(0.5, None, None, model)
        if not updated:
            failures.append("update_best: valid loss rejected")

        print("  PASS" if not any("update_best" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 11: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 12: Data split disjointness ==========
    print("Test 12: Data splits")
    try:
        total = 521
        train_end = int(0.8 * total)
        val_start = train_end + 1
        val_end = val_start + int(0.1 * total)
        test_start = val_end + 1

        train = set(range(0, train_end))
        val = set(range(val_start, val_end))
        test = set(range(test_start, total))

        if train & val:
            failures.append("CPOX: train/val overlap")
        if train & test:
            failures.append("CPOX: train/test overlap")
        if val & test:
            failures.append("CPOX: val/test overlap")

        print("  PASS" if not any("CPOX" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 12: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 13: Full model batch independence ==========
    print("Test 13: Full model batch independence")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=2, device='cpu')
        model.eval()

        n1, n2 = 15, 25
        x1, x2 = torch.randn(n1, 1), torch.randn(n2, 1)
        edge1 = torch.randint(0, n1, (2, 30))
        edge2 = torch.randint(0, n2, (2, 50))
        obs = 5
        local_ind = torch.arange(obs)

        for task in ['source_localization', 'pde_reconstruction', 'denoising',
                     'inpainting', 'sensor_recovery']:
            model.set_task(task)

            def set_local():
                if task == 'inpainting':
                    model.current_forward_op.ind = local_ind
                elif task == 'sensor_recovery':
                    model.current_forward_op.sensor_indices = local_ind

            set_local()
            with torch.no_grad():
                out1, _, _ = model(x1, edge1, torch.ones(30), x1, batch=torch.zeros(n1, dtype=torch.long))
            set_local()
            with torch.no_grad():
                out2, _, _ = model(x2, edge2, torch.ones(50), x2, batch=torch.zeros(n2, dtype=torch.long))

            x_batch = torch.cat([x1, x2])
            edge_batch = torch.cat([edge1, edge2 + n1], dim=1)
            batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])

            # global observation indices: first `obs` nodes of each graph
            if task == 'inpainting':
                model.current_forward_op.ind = torch.cat([local_ind, n1 + local_ind])
            elif task == 'sensor_recovery':
                model.current_forward_op.sensor_indices = torch.cat([local_ind, n1 + local_ind])

            with torch.no_grad():
                out_batch, _, _ = model(x_batch, edge_batch, torch.ones(80), x_batch, batch=batch)

            if not torch.allclose(out_batch[:n1], out1, rtol=1e-4, atol=1e-5):
                failures.append(f"Full batch {task}: output graph 1")
            if not torch.allclose(out_batch[n1:], out2, rtol=1e-4, atol=1e-5):
                failures.append(f"Full batch {task}: output graph 2")

        print("  PASS" if not any("Full batch" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 13: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 14: Legacy method compatibility ==========
    print("Test 14: Legacy method (drip)")
    try:
        args = argparse.Namespace(
            method='drip', regnet='hyper', layers=2, channels=8,
            cglsIter=2, solveIter=2, rnfPE=1, task='mask', dropout=0.0,
            mu=0.01, blur_count='4', classify=0,
            noise=False, painting=False, blurring=False, sensoring=False, pdessm=False
        )

        forward_op = get_forward_op(args, 8, 1, 'cpu')
        forward_op.ind = torch.arange(10)

        net = get_network(args, forward_op, 8, 1, 1, 'cpu')

        x = torch.randn(20, 1)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_weight = torch.ones(50)

        out, _, _ = net(x, edge_index, edge_weight, x)

        if out.shape != x.shape:
            failures.append(f"Legacy: output shape {out.shape}")

        print("  PASS" if not any("Legacy" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 14: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 15: CGLS legacy operator signature (no batch=) ==========
    print("Test 15: CGLS legacy operator signature")
    try:
        # A legacy-style operator whose forward/adjoint do NOT accept a batch=
        # kwarg (like graphPath). CGLS must not pass batch= to it.
        class LegacyOp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.learnEmb = False

            def forward(self, I, edge_index=None, edge_weight=None, emb=True):
                return 2.0 * I

            def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
                return 2.0 * Ic

        op = LegacyOp()
        solver = networks.graph_CGLS(op, CGLSit=5, eps=1e-6)
        x = torch.randn(12, 1)
        edge = torch.randint(0, 12, (2, 20))
        w = torch.ones(20)
        b = op.forward(x, edge, w, emb=False)
        # single-graph and batched paths must both avoid passing batch=
        out_s, _ = solver(b, torch.zeros_like(x), edge, w, emb=False)
        batch = torch.cat([torch.zeros(6, dtype=torch.long), torch.ones(6, dtype=torch.long)])
        out_b, _ = solver(b, torch.zeros_like(x), edge, w, batch=batch, emb=False)
        # A = 2I  =>  solution of Ax=b is x (since b=2x). Recovered exactly.
        if not torch.allclose(out_s, x, atol=1e-4):
            failures.append("Legacy-signature CGLS: single-graph solve wrong")
        print("  PASS" if not any("Legacy-signature" in f for f in failures) else "  FAIL")
    except TypeError as ex:
        failures.append(f"Test 15: CGLS passed batch= to legacy operator: {ex}")
        print(f"  FAIL: {ex}")
    except Exception as ex:
        failures.append(f"Test 15: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 16: Legacy checkpoint migration (alias keys) ==========
    print("Test 16: Legacy alias-key migration")
    try:
        ref = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        with torch.no_grad():
            for p in ref.parameters():
                p.add_(torch.randn_like(p))
        ref.set_task('source_localization')

        # Build a legacy-style state_dict WITH alias keys duplicating the active head.
        base = ref.state_dict()
        legacy = dict(base)
        active = 'source_localization'
        for k, v in base.items():
            prefix = f'task_heads.{active}.'
            if k.startswith(prefix):
                suffix = k[len(prefix):]
                legacy[f'current_forward_op.{suffix}'] = v.clone()
                legacy[f'solver.forOp.{suffix}'] = v.clone()

        # migrate + load must succeed and preserve every head's predictions
        target = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        migrated = migrate_state_dict(legacy)
        miss, unexp = target.load_state_dict(migrated, strict=False)
        if set(miss) or set(unexp):
            failures.append(f"Migration: leftover keys miss={set(miss)} unexp={set(unexp)}")

        x = torch.randn(20, 1)
        ei = torch.randint(0, 20, (2, 50)); ew = torch.ones(50)
        batch = torch.zeros(20, dtype=torch.long)
        for task in ref.task_heads.keys():
            ref.set_task(task); target.set_task(task)
            if hasattr(ref.current_forward_op, 'ind'):
                ref.current_forward_op.ind = torch.arange(10)
                target.current_forward_op.ind = torch.arange(10)
            if hasattr(ref.current_forward_op, 'sensor_indices'):
                ref.current_forward_op.sensor_indices = torch.arange(10)
                target.current_forward_op.sensor_indices = torch.arange(10)
            with torch.no_grad():
                a, _, _ = ref(x, ei, ew, x, batch=batch)
                b2, _, _ = target(x, ei, ew, x, batch=batch)
            if not torch.allclose(a, b2, atol=1e-6):
                failures.append(f"Migration: head {task} prediction changed")
        print("  PASS" if not any("Migration" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 16: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 17: Normalization-stats + blur_k checkpoint roundtrip ==========
    print("Test 17: METRLA-style stats roundtrip")
    try:
        import numpy as _np
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=20,
            label_channels=1, niter=1, cgls_iter=1, device='cpu', blur_k=3)
        stats = {'means': _np.array([1.0, 2.0], dtype=_np.float32),
                 'stds': _np.array([0.5, 0.25], dtype=_np.float32)}
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            fpath = f.name
        save_foundation_checkpoint(
            model, fpath, held_out_flag='blurring',
            enabled_flags=['noise', 'painting', 'blurring', 'sensoring', 'pdessm'],
            denoising_bypass=True, split_info={}, hid_channels=8, label_channels=1,
            niter=1, cgls_iter=1, normalization_stats=stats, blur_k=3)

        # Must load even under weights_only=True (no NumPy arrays persisted)
        raw = torch.load(fpath, weights_only=True)
        if not isinstance(raw['normalization_stats']['means'], list):
            failures.append("Stats not stored as lists")
        if raw.get('blur_k') != 3:
            failures.append("blur_k not persisted")

        # Architecture validation: wrong blur_k must be rejected
        bad = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=20,
            label_channels=1, niter=1, cgls_iter=1, device='cpu', blur_k=4)
        rejected = False
        try:
            load_checkpoint(fpath, bad, 'cpu')
        except ValueError:
            rejected = True
        if not rejected:
            failures.append("blur_k mismatch not rejected")
        os.unlink(fpath)
        print("  PASS" if not any(("Stats" in f or "blur_k" in f) for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 17: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 18: evaluate_all_operators + graph-count weighting ==========
    print("Test 18: production evaluation + weighting")
    try:
        # Unit check: graph-count weighting differs from batch-mean for unequal batches
        predA = torch.tensor([[3.0]])            # 1 graph, ratio (3-1)^2/1 = 4
        tgtA = torch.tensor([[1.0]])
        batchA = torch.tensor([0])
        sA, nA = accumulate_pergraph_ratios(predA, tgtA, batchA)
        predB = torch.tensor([[1.0], [1.0], [1.0]])   # 3 graphs, ratio 0 each
        tgtB = torch.tensor([[1.0], [1.0], [1.0]])
        batchB = torch.tensor([0, 1, 2])
        sB, nB = accumulate_pergraph_ratios(predB, tgtB, batchB)
        graph_weighted = (sA + sB) / (nA + nB)        # (4 + 0)/4 = 1.0
        batch_mean = ((sA / nA) + (sB / nB)) / 2       # (4 + 0)/2 = 2.0
        if abs(graph_weighted - 1.0) > 1e-9 or abs(batch_mean - 2.0) > 1e-9:
            failures.append(f"Weighting wrong: gw={graph_weighted}, bm={batch_mean}")

        # Integration: evaluate_all_operators over a fake loader with unequal batches
        class G:
            def __init__(self, n):
                self.y = torch.randn(n, 1)
                self.x = torch.randn(n, 1)
                self.edge_index = torch.randint(0, n, (2, 3 * n))
                self.edge_weight = torch.ones(3 * n)
                self.batch = torch.zeros(n, dtype=torch.long)
            def to(self, dev):
                return self

        class FakeLoader:
            def __init__(self):
                self.graphs = [G(20), G(8)]   # unequal batch sizes
            def __iter__(self):
                return iter(self.graphs)

        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=2, device='cpu')
        eargs = argparse.Namespace(mask_per_snapshot_budget=5, cglsIter=2,
                                   channels=8, blur_count='4', held_out_op='blurring',
                                   noise=True, painting=True, blurring=True,
                                   sensoring=True, pdessm=True)
        summary = evaluate_all_operators(model, FakeLoader(), eargs, 'cpu', lambda a, g: g)
        for flag in ['noise', 'painting', 'blurring', 'sensoring', 'pdessm']:
            if flag not in summary:
                failures.append(f"eval missing {flag}")
            elif abs(summary[flag]['zero'] - 1.0) > 1e-6:
                failures.append(f"eval zero-baseline != 1.0 for {flag}: {summary[flag]['zero']}")
        if summary['blurring']['trained'] is not False:
            failures.append("held-out 'blurring' marked trained")
        if summary['noise']['trained'] is not True:
            failures.append("seen 'noise' not marked trained")
        print("  PASS" if not any(("Weighting" in f or "eval " in f or "trained" in f) for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 18: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 19: temporal splits + synthetic signal slicing ==========
    print("Test 19: temporal split construction")
    try:
        (tr, va, te) = foundation_temporal_splits(100, 0.8, 0.1, gap=1)
        # disjoint with gaps
        if not (tr == (0, 80) and va == (81, 91) and te == (92, 100)):
            failures.append(f"split ranges wrong: {tr},{va},{te}")
        # actual slicing of a synthetic temporal signal
        from torch_geometric_temporal.signal import StaticGraphTemporalSignal
        N = 100
        edges = _np_arr()
        feats = [__import__('numpy').random.randn(4, 1).astype('float32') for _ in range(N)]
        targs = [__import__('numpy').random.randn(4).astype('float32') for _ in range(N)]
        sig = StaticGraphTemporalSignal(edges, __import__('numpy').ones(edges.shape[1]), feats, targs)
        train_sig = sig[tr[0]:tr[1]]
        val_sig = sig[va[0]:va[1]]
        test_sig = sig[te[0]:te[1]]
        if train_sig.snapshot_count != 80 or val_sig.snapshot_count != 10 or test_sig.snapshot_count != 8:
            failures.append("sliced signal snapshot counts wrong")
        print("  PASS" if not any(("split" in f or "sliced" in f) for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 19: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 20: phase freezing (phases 2 and 3) ==========
    print("Test 20: phase freezing")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        held = 'denoising'

        # Phase 2: backbone+feat_embed frozen, held-out frozen, seen heads trainable
        model.freeze_backbone()
        for name, head in model.task_heads.items():
            for p in head.parameters():
                p.requires_grad = (name != held)
        if any(p.requires_grad for p in model.backbone.parameters()):
            failures.append("Phase2: backbone not frozen")
        if any(p.requires_grad for p in model.feat_embed.parameters()):
            failures.append("Phase2: feat_embed not frozen")
        if any(p.requires_grad for p in model.task_heads[held].parameters()):
            failures.append("Phase2: held-out head trainable")
        if not any(p.requires_grad for p in model.task_heads['inpainting'].parameters()):
            failures.append("Phase2: seen head frozen")

        # Phase 3: only held-out head trainable
        model.freeze_backbone()
        for name, head in model.task_heads.items():
            for p in head.parameters():
                p.requires_grad = (name == held)
        p3 = model.get_trainable_params_for_phase(3, held)
        if not all(p.requires_grad for p in model.task_heads[held].parameters()):
            failures.append("Phase3: held-out head not trainable")
        if any(p.requires_grad for p in model.task_heads['inpainting'].parameters()):
            failures.append("Phase3: seen head still trainable")
        if len(list(p3)) != len(list(model.task_heads[held].parameters())):
            failures.append("Phase3: param selection wrong")
        print("  PASS" if not any("Phase2" in f or "Phase3" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 20: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 21: DeterministicFixedPoints repeatability ==========
    print("Test 21: deterministic eval sampling")
    try:
        from torch_geometric.data import Data
        pos = torch.randn(50, 3)
        d1 = Data(pos=pos.clone(), x=torch.randn(50, 3), num_nodes=50)
        d2 = Data(pos=pos.clone(), x=d1.x.clone(), num_nodes=50)
        t = DeterministicFixedPoints(16)
        o1 = t(d1); o2 = t(d2)
        if o1.num_nodes != 16 or not torch.equal(o1.pos, o2.pos) or not torch.equal(o1.x, o2.x):
            failures.append("DeterministicFixedPoints not repeatable")
        print("  PASS" if not any("DeterministicFixedPoints" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 21: {ex}")
        print(f"  FAIL: {ex}")

    # ========== Test 22: foundation_validate weighting + empty rejection ==========
    print("Test 22: foundation_validate contracts")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        class G2:
            def __init__(self, n):
                self.y = torch.randn(n, 1); self.x = torch.randn(n, 1)
                self.edge_index = torch.randint(0, n, (2, 2 * n)); self.edge_weight = torch.ones(2 * n)
                self.batch = torch.zeros(n, dtype=torch.long)
            def to(self, d): return self

        loader = [G2(20), G2(4)]
        vargs = argparse.Namespace(mask_per_snapshot_budget=5)
        val = foundation_validate(model, loader, 'cpu', vargs, lambda a, g: g, ['noise'])
        if not (val >= 0):
            failures.append("foundation_validate returned invalid loss")
        # empty selection must raise
        raised = False
        try:
            foundation_validate(model, loader, 'cpu', vargs, lambda a, g: g, [])
        except ValueError:
            raised = True
        if not raised:
            failures.append("foundation_validate empty-selection did not raise")
        # empty loader must raise
        raised2 = False
        try:
            foundation_validate(model, [], 'cpu', vargs, lambda a, g: g, ['noise'])
        except ValueError:
            raised2 = True
        if not raised2:
            failures.append("foundation_validate empty-loader did not raise")
        print("  PASS" if not any("foundation_validate" in f for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 22: {ex}")
        print(f"  FAIL: {ex}")

    # Summary
    print(f"\n{'=' * 50}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All tests passed")
    return 0


def _np_arr():
    """Small helper: a fixed 4-node ring edge_index for synthetic temporal tests."""
    import numpy as _np
    return _np.array([[0, 1, 2, 3], [1, 2, 3, 0]])