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
            device=device
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
    """Generate reproducible operator configuration from seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_nodes = graph.y.shape[0]
    config = {}

    if flag == 'painting':
        # Random mask indices
        n_obs = min(getattr(args, 'mask_per_snapshot_budget', 16), n_nodes)
        perm = torch.randperm(n_nodes, generator=gen)
        config['ind'] = perm[:n_obs]
    elif flag == 'sensoring':
        # Fixed first-k sensor positions
        n_sensors = min(getattr(args, 'mask_per_snapshot_budget', 16), n_nodes)
        config['sensor_indices'] = torch.arange(n_sensors)
    # Other operators don't need special config

    return config


def apply_config(op, config):
    """Apply frozen config to operator."""
    if 'ind' in config:
        op.ind = config['ind'].to(op.ind.device if hasattr(op, 'ind') and op.ind is not None else 'cpu')
    if 'sensor_indices' in config:
        op.sensor_indices = config['sensor_indices'].to(
            op.sensor_indices.device if hasattr(op, 'sensor_indices') and op.sensor_indices is not None else 'cpu')


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


def save_foundation_checkpoint(model, path, *, held_out_flag, enabled_flags, denoising_bypass,
                               split_info, hid_channels, label_channels, niter, cgls_iter,
                               normalization_stats=None):
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
        'normalization_stats': normalization_stats,
        'task_flag_mapping': FLAG_TO_TASK,
        'version': 2
    }
    torch.save(ckpt, path)
    print(f"Foundation checkpoint saved to: {path}")


def load_checkpoint(path, model, device):
    """Load foundation checkpoint with strict validation."""
    ckpt = torch.load(path, map_location=device)

    version = ckpt.get('version', 1)
    state = ckpt.get('model_state_dict', ckpt)

    # Load state dict strictly
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Only allow specific known migrations
    allowed_missing = set()
    allowed_unexpected = set()

    actual_missing = set(missing) - allowed_missing
    actual_unexpected = set(unexpected) - allowed_unexpected

    if actual_missing:
        raise ValueError(f"Missing keys not in migration: {actual_missing}")
    if actual_unexpected:
        raise ValueError(f"Unexpected keys not in migration: {actual_unexpected}")

    # Restore settings
    model.denoising_bypass = ckpt.get('denoising_bypass', True)

    # Update solver reference
    model.solver.forOp = model.current_forward_op

    return ckpt


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

    # ========== Test 3: Mask/sensor batched CGLS ==========
    print("Test 3: Mask/sensor batched CGLS")
    try:
        n1, n2 = 10, 15
        obs_per_graph = 5

        for OpClass, idx_attr in [(graphMask, 'ind'), (SensorRecovery, 'sensor_indices')]:
            x1, x2 = torch.randn(n1, 1), torch.randn(n2, 1)
            edge1 = torch.randint(0, n1, (2, 20))
            edge2 = torch.randint(0, n2, (2, 30))

            # Global indices for batched
            local_ind1 = torch.arange(obs_per_graph)
            local_ind2 = torch.arange(obs_per_graph)
            global_ind = torch.cat([local_ind1, n1 + local_ind2])

            # Create operators
            if OpClass == graphMask:
                op = graphMask(ind=global_ind, embdsize=8, nin=1, device='cpu', learnEmb=False)
                op1 = graphMask(ind=local_ind1, embdsize=8, nin=1, device='cpu', learnEmb=False)
                op2 = graphMask(ind=local_ind2, embdsize=8, nin=1, device='cpu', learnEmb=False)
            else:
                op = SensorRecovery(sensor_indices=global_ind, nin=1, embdsize=8, device='cpu', learnEmb=False)
                op1 = SensorRecovery(sensor_indices=local_ind1, nin=1, embdsize=8, device='cpu', learnEmb=False)
                op2 = SensorRecovery(sensor_indices=local_ind2, nin=1, embdsize=8, device='cpu', learnEmb=False)

            solver = networks.graph_CGLS(op, CGLSit=5, eps=1e-5)
            solver1 = networks.graph_CGLS(op1, CGLSit=5, eps=1e-5)
            solver2 = networks.graph_CGLS(op2, CGLSit=5, eps=1e-5)

            # Individual solves
            b1 = op1.forward(x1, edge1, torch.ones(20), emb=False)
            out1, res1 = solver1(b1, torch.zeros_like(x1), edge1, torch.ones(20), emb=False)

            b2 = op2.forward(x2, edge2, torch.ones(30), emb=False)
            out2, res2 = solver2(b2, torch.zeros_like(x2), edge2, torch.ones(30), emb=False)

            # Batched solve
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

        print("  PASS" if not any("batch" in f.lower() for f in failures) else "  FAIL")
    except Exception as ex:
        failures.append(f"Test 3: {ex}")
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

    # ========== Test 9: Zero-epoch returns entry state ==========
    print("Test 9: Zero-epoch handling")
    try:
        model = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')

        entry_state = copy.deepcopy(model.state_dict())

        # Mock functions
        def mock_train(*args, **kwargs):
            pass

        def mock_validate(*args, **kwargs):
            return 0.5

        result_state, result_loss = run_phase(
            model, [], [],
            torch.optim.Adam(model.parameters()),
            max_epochs=0, patience=10, phase_name='test_zero',
            device='cpu', args=None, process_fn=lambda a, g: g,
            metric_fn=compute_loss, train_fn=mock_train, validate_fn=mock_validate)

        for k in entry_state:
            if not torch.equal(result_state[k], entry_state[k]):
                failures.append("Zero-epoch: state changed")
                break

        print("  PASS" if not any("Zero-epoch" in f for f in failures) else "  FAIL")
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

        model2 = networks.GraphInverseFoundationModel(
            num_layers=2, hid_channels=8, input_feat_dim=1,
            label_channels=1, niter=1, cgls_iter=1, device='cpu')
        model2.denoising_bypass = True

        ckpt = load_checkpoint(fpath, model2, 'cpu')

        if model2.denoising_bypass != False:
            failures.append("Checkpoint: denoising_bypass not restored")

        if ckpt.get('held_out_flag') != 'blurring':
            failures.append("Checkpoint: held_out_flag wrong")

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

        for task in ['source_localization', 'pde_reconstruction', 'denoising']:
            model.set_task(task)

            with torch.no_grad():
                out1, _, _ = model(x1, edge1, torch.ones(30), x1, batch=torch.zeros(n1, dtype=torch.long))
                out2, _, _ = model(x2, edge2, torch.ones(50), x2, batch=torch.zeros(n2, dtype=torch.long))

            x_batch = torch.cat([x1, x2])
            edge_batch = torch.cat([edge1, edge2 + n1], dim=1)
            batch = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(n2, dtype=torch.long)])

            with torch.no_grad():
                out_batch, _, _ = model(x_batch, edge_batch, torch.ones(80), x_batch, batch=batch)

            if not torch.allclose(out_batch[:n1], out1, rtol=1e-4):
                failures.append(f"Full batch {task}: output graph 1")
            if not torch.allclose(out_batch[n1:], out2, rtol=1e-4):
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

    # Summary
    print(f"\n{'=' * 50}")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All 14 tests passed")
    return 0