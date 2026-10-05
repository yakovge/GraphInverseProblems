import os
import torch
from torch_geometric.datasets.gnn_benchmark_dataset import GNNBenchmarkDataset
from torch_geometric_temporal.signal import temporal_signal_split

from customMETRLA import METRLADatasetLoader
from pygt_dataloader import DataLoader as BatchDataLoader
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from graphForwardOps import graph_smooth, graphMask, graphPath, graph_edgeRecovery, SensorRecovery
import networks
from customCPOX import ChickenpoxDatasetLoader
import math 
import random 
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.data import Data


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

    num_nodes = args.num_nodes 

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
        ops.append([PDESSM(nin=label_channels, embdsize=hid_channels, dim=args.num_nodes, tau=getattr(args, 'pde_tau', 20.0),
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
        ops.append([AddNoise(nin=label_channels, embdsize=hid_channels, noise_std=0.25, 
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

        net = networks.verseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
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
            label_channels=label_channels,
            niter=args.solveIter,
            cgls_iter=args.cglsIter,
            device=device,
            forward_ops=forward_op  
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


PGT_LOADERS = {'CPOX': 'ChickenpoxDatasetLoader', 'PEDALME': 'PedalMeDatasetLoader', 'WIKIMATHS': 'WikiMathsDatasetLoader',
               'MONTEVIDEO': 'MontevideoBusDatasetLoader', 'WINDMILL': 'WindmillOutputLargeDatasetLoader'}

def load_pgt_snapshots(name):
    """All snapshots of a pgt regression dataset: z-normalised target, constant feature (native ones are lagged targets)."""
    import torch_geometric_temporal.dataset as pgt
    snaps = list(getattr(pgt, PGT_LOADERS[name])().get_dataset(lags=1))
    y = torch.stack([torch.as_tensor(s.y).float().view(-1) for s in snaps])
    mu, sd = y.mean(), y.std()
    return [Data(x=torch.ones(len(t), 1), y=((t - mu) / sd).unsqueeze(-1), edge_index=s.edge_index) for t, s in zip(y, snaps)]


class MultiLoader:
    """Shuffled interleaving of per-dataset loaders, so every batch holds graphs of one size (needed by PDESSM)."""
    def __init__(self, loaders):
        self.loaders = loaders
    def __len__(self):
        return sum(len(l) for l in self.loaders)
    def __iter__(self):
        its = [iter(l) for l in self.loaders]
        order = [i for i, l in enumerate(self.loaders) for _ in range(len(l))]
        random.shuffle(order)
        return (next(its[i]) for i in order)


def get_data_and_loaders(args):
    # validation = last val_frac of the training time series (only CPOX and MULTI); used for model selection
    val_frac, val_loader = getattr(args, 'val_frac', 0.1), None
    if args.dataset == 'MULTI':
        # leave-one-dataset-out: train on every snapshot of args.train_datasets, test on the whole args.test_dataset
        train_sets = [load_pgt_snapshots(d) for d in args.train_datasets.split(',')]
        val_sets = [s[len(s) - int(val_frac * len(s)):] for s in train_sets]  # time tail of every training dataset
        train_sets = [s[:len(s) - int(val_frac * len(s))] for s in train_sets]
        val_loader = MultiLoader([DataLoader(d, batch_size=args.test_batch_size, shuffle=False) for d in val_sets])
        train_dataset = sum(train_sets, [])
        test_dataset = get_fractional_dataset(load_pgt_snapshots(args.test_dataset), args.test_frac)
        train_loader = MultiLoader([DataLoader(d, batch_size=args.train_batch_size, shuffle=True) for d in train_sets])
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        label_channels, feat_channels = 1, 1

    elif args.dataset in ['CLUSTER', 'PATTERN']:
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
        train_dataset = list(train_dataset)
        n_val = int(val_frac * len(train_dataset))  # weeks just before the test weeks
        val_loader = DataLoader(train_dataset[len(train_dataset) - n_val:], batch_size=args.test_batch_size, shuffle=False)
        train_dataset = train_dataset[:len(train_dataset) - n_val]
        train_loader = DataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        test_loader = DataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        label_channels = 1 #lags #1
        feat_channels =  2 if args.use_meta_data else 1 # sin/cos week-of-year encoding (see process_data), or a constant

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
    
    feat_channels += 1
    return train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels, val_loader

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
        # features were raw week indices 1..521 (unbounded, test weeks never seen in training -> exploding outputs);
        # encode them as bounded yearly seasonality instead
        week = 2 * math.pi * graph.x.clone()[:,1:2] / 52.0
        graph.x = torch.cat([torch.sin(week), torch.cos(week)], dim=-1)
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

    # ADD THESE TWO LINES AT THE VERY END OF process_data:
    mask = torch.ones((graph.x.shape[0], 1), dtype=torch.float32, device=graph.x.device)
    graph.x = torch.cat([graph.x, mask], dim=-1)

    return graph


def clustered_nodes(edge_index, nodes, budget):
    """Grow one BFS patch of `budget` nodes inside a snapshot (sensors cover whole neighbourhoods)."""
    budget = min(budget, len(nodes))
    obs = nodes[torch.randint(len(nodes), (1,))]
    while len(obs) < budget:
        nbr = edge_index[1, torch.isin(edge_index[0], obs)]
        new = nbr[~torch.isin(nbr, obs)].unique()
        if len(new) == 0:  # component exhausted: seed a new patch
            rest = nodes[~torch.isin(nodes, obs)]
            new = rest[torch.randint(len(rest), (1,))]
        obs = torch.cat([obs, new[torch.randperm(len(new))[:budget - len(obs)]]])
    return obs


def task_specific_modifiers(graph, args, forward_op, dataset):
    if isinstance(forward_op, PDESSM):  # graph size varies across datasets; batches are single-dataset
        forward_op.nodes_per_graph = graph.num_nodes // len(graph.batch.unique())
    # sensors first: args.task defaults to 'mask', so the painting branch would otherwise catch every op
    if isinstance(forward_op, SensorRecovery):
        # resample clustered sensor patches per snapshot every batch (was a fixed arange(25) over the whole batch)
        forward_op.sensor_indices = torch.cat([clustered_nodes(graph.edge_index, torch.where(graph.batch == c)[0],
                                               args.n_sensors) for c in graph.batch.unique()])
        graph.x[:, -1] = 0.0
        graph.x[forward_op.sensor_indices, -1] = 1.0
    # 1. Inpainting / Masking -- only for the mask operator: `args.task == 'mask' or args.painting` (default task 'mask')
    # also hit denoising/blur/PDE and wrote a random 6-node "observed" flag into graph.x[:, -1] for them
    elif isinstance(forward_op, graphMask):
        if args.classify == 1:
            n_classes = dataset.num_classes
            total_mask_budget = int(args.train_batch_size * args.mask_per_class_budget)
            mask_indices = []
            for c in range(n_classes):
                class_ind = torch.where(graph.y.argmax(dim=-1) == c)[0]
                sampled = list(class_ind[torch.randperm(len(class_ind))[:total_mask_budget]])
                mask_indices.extend(sampled)
            rand_mask = torch.tensor(mask_indices, device=graph.x.device).long()
        else:
            # regression
            mask_indices = []
            for c in graph.batch.unique():
                snapshot_ind = torch.where(graph.batch == c)[0]
                budget = getattr(args, 'mask_per_snapshot_budget', 6)
                sampled = list(snapshot_ind[torch.randperm(len(snapshot_ind))[:budget]])
                mask_indices.extend(sampled)
            rand_mask = torch.tensor(mask_indices, device=graph.x.device).long()
                 
        forward_op.ind = rand_mask
        graph.x[:, -1] = 0.0 
        graph.x[rand_mask, -1] = 1.0 # 

    # 3. Path / Random Walk
    elif args.task == 'path':
        forward_op.gen_paths(nnodes=graph.x.shape[0], edge_index=graph.edge_index)
         
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