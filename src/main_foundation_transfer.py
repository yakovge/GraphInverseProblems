"""Foundation-model study: cross-operator pretraining + transfer to held-out operators.

Protocol (per Prof. Eliasof's guidance): the signal family / dataset is FIXED and
only the forward operator A changes. One PGD-style unrolled solver
(graph_NeuralProximalGradient backbone) is pretrained on a mixture of operators
-- the operator is resampled per batch -- and then evaluated on operators it
never saw:

  pretrain mixture : denoise (identity+noise), mask (property completion),
                     smooth (A^k diffusion; GRIP's 'deblur' / inverse source
                     estimation), path (inverse graph transport)
  held out         : maskSmooth (PDE-state reconstruction: masked diffused state;
                     compositional/near holdout)
                     maxpool (nonlinear neighborhood max; genuinely new operator,
                     far holdout)

Evaluation on each holdout: zero-shot (frozen), few-shot fine-tuning on N support
graphs (fixed schedule, no test peeking), the "denoiser-only" transfer recipe
(data-fidelity step disabled -- the v1-project mechanism finding re-tested at this
scale), a from-scratch model trained on the same N graphs, and the classical
Tikhonov solver. Metrics follow the upstream script (node accuracy for
classification datasets); results go to a CSV.

Example (CPU smoke):
  python main_foundation_transfer.py --dataset CLUSTER --device cpu \
      --pretrain_epochs 1 --train_frac 0.01 --eval_batches 5 \
      --few_shot_N 5 --seeds 0 --ft_steps 20

Example (GPU, preliminary result):
  python main_foundation_transfer.py --dataset CLUSTER --device cuda:0 \
      --pretrain_epochs 10 --seeds 0,1,2
"""
import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import networks
from foundation_ops import (NUM_OP_TYPES, apply_batch_modifiers,
                            make_foundation_op, op_type_id)
from utils import count_trainable_parameters, get_data_and_loaders, process_data

##################################
########### ARGUMENTS ############
##################################
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data'))
parser.add_argument('--dataset', type=str, default='CLUSTER')
parser.add_argument('--classify', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--layers', type=int, default=12)   # c=64, l=12 -> ~155k params
parser.add_argument('--solveIter', type=int, default=5)      # unrolling steps
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--rnfPE', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=4e-6)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--train_frac', type=float, default=1.0)
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--eval_batches', type=int, default=50)   # cap on test batches
# operators
parser.add_argument('--pretrain_ops', type=str, default='denoise,mask,smooth,path')
parser.add_argument('--holdout_ops', type=str, default='maskSmooth,maxpool')
parser.add_argument('--blur_count', type=str, default='4')
parser.add_argument('--pathLength', type=int, default=32)
parser.add_argument('--mask_per_class_budget', type=int, default=4)
parser.add_argument('--mask_per_snapshot_budget', type=int, default=16)  # regression datasets
parser.add_argument('--denoise_sigma', type=float, default=0.3)
# conditioning (our addition; 0 = vanilla upstream behavior)
parser.add_argument('--op_conditioning', type=int, default=1)
parser.add_argument('--type_dropout', type=float, default=0.25)
# transfer protocol
parser.add_argument('--few_shot_N', type=str, default='5,20,100')
parser.add_argument('--n_support_draws', type=int, default=2)
parser.add_argument('--ft_steps', type=int, default=150)      # fixed schedule, all methods
parser.add_argument('--seeds', type=str, default='0,1,2')
parser.add_argument('--skip_classical', type=int, default=0)
# io
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
parser.add_argument('--wandb', type=int, default=0)
args = parser.parse_args()
args.test_batch_size = args.train_batch_size   # as in the upstream script
args.use_meta_data = 1                         # upstream default (keep node features)
args.CPOX_lags = 1
args.test_frac = 1.0

device = args.device
PRETRAIN_OPS = args.pretrain_ops.split(',')
HOLDOUT_OPS = [h for h in args.holdout_ops.split(',') if h]
FEW_SHOT_N = [int(n) for n in args.few_shot_N.split(',')]
SEEDS = [int(s) for s in args.seeds.split(',')]
os.makedirs(args.ckpt_dir, exist_ok=True)
out_csv = args.out or os.path.join(os.path.dirname(__file__), '..', 'results',
                                   f'foundation_{args.dataset}.csv')
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

if args.wandb:
    import wandb
    wandb.init(project='grip-foundation', config=vars(args))

LABEL_CHANNELS = [None]  # set in main() once the dataset is loaded


##################################
############ MODEL ###############
##################################
class FoundationPGD(networks.graph_NeuralProximalGradient):
    """graph_NeuralProximalGradient with (a) per-batch operator swapping,
    (b) optional operator-type conditioning added to the feature embedding
    (UNKNOWN token trained via type dropout, so held-out operators are not handed
    an untrained category), (c) a data_step flag implementing the denoiser-only
    transfer recipe, and (d) open/close maps lifting the iterate to hidden width
    so the regularizer carries ~100-200k parameters (upstream PGD keeps the
    iterate in label space, which caps it at a few thousand).
    """

    def __init__(self, regNet, dataProj, forOp, niter, input_feat_dim, rnfPE,
                 label_channels, hid_channels, op_conditioning=True,
                 type_dropout=0.25):
        super().__init__(regNet, dataProj, forOp, niter=niter,
                         input_feat_dim=input_feat_dim, rnfPE=rnfPE)
        self.open = nn.Linear(label_channels, hid_channels)
        self.close = nn.Linear(hid_channels, label_channels)
        self.op_conditioning = op_conditioning
        self.type_dropout = type_dropout
        if op_conditioning:
            self.type_emb = nn.Embedding(NUM_OP_TYPES, hid_channels)
        self.op_name = None
        self.data_step = True     # False = denoiser-only transfer recipe

    def set_operator(self, op, name):
        self.forOp = op
        self.dataProj.forOp = op
        self.op_name = name

    def _cond(self, f):
        if not self.op_conditioning:
            return f
        from foundation_ops import OP_TYPE_IDS
        tid = op_type_id(self.op_name)
        if self.training and self.type_dropout > 0 and torch.rand(()) < self.type_dropout:
            tid = OP_TYPE_IDS['UNKNOWN']
        e = self.type_emb(torch.tensor(tid, device=f.device))
        return f + e.unsqueeze(0)

    def forward(self, D, edge_index, edge_weights, f=None):
        if f is not None:
            f = self.feat_embed(f)
            if self.rnfPE:
                rnf = torch.randn(f.shape[0], self.net.nchannels, device=D.device)
                rnf = torch.sin(self.rnf_embed(rnf))
                f = torch.cat([f, rnf], dim=-1)
                f = self.rnf_combine(f)
            f = self._cond(f)

        # initial recovery: scaled back-projection (as upstream)
        Z = self.forOp.adjoint(D, edge_index, edge_weights, emb=False)
        Az = self.forOp.forward(Z, edge_index, edge_weights, emb=False)
        alpha = (D * Az).mean(dim=(0, 1), keepdim=True) / ((Az * Az).mean(dim=(0, 1), keepdim=True) + 1e-12)
        Z = alpha * Z

        Zall = []
        for _ in range(self.niter):
            Zh = self.open(Z)
            Zh, Zall = self.net(Zh, Zall, f, edge_index, edge_weights)
            Zref = Z + self.close(Zh)
            if self.data_step:
                R = D - self.forOp.forward(Zref, edge_index, edge_weights, emb=False)
                G = self.forOp.adjoint(R, edge_index, edge_weights, emb=False)
                Ag = self.forOp.forward(G, edge_index, edge_weights, emb=False)
                mu = (R * Ag).mean(dim=(0, 1), keepdim=True) / ((Ag * Ag).mean(dim=(0, 1), keepdim=True) + 1e-12)
                Z = Zref + mu * G
            else:
                Z = Zref
        return Z, Zref, torch.Tensor([0])


def build_model(label_channels, feat_channels):
    reg = networks.graphResNetFO(num_layers=args.layers, nopen=args.channels,
                                 nfeatures=args.channels, dropout=args.dropout)
    # dataProj is kept for interface parity with upstream pgd (cglsIter=1 there);
    # the explicit data step lives in FoundationPGD.forward.
    placeholder_op = make_foundation_op('denoise', args, label_channels, args.channels, device)
    proj = networks.graph_CGLS(placeholder_op, CGLSit=1, eps=1e-5)
    net = FoundationPGD(reg, proj, placeholder_op, niter=args.solveIter,
                        input_feat_dim=feat_channels, rnfPE=args.rnfPE,
                        label_channels=label_channels, hid_channels=args.channels,
                        op_conditioning=bool(args.op_conditioning),
                        type_dropout=args.type_dropout)
    return net.to(device)


##################################
########## DATA / LOSS ###########
##################################
def make_ops(label_channels):
    return {name: make_foundation_op(name, args, label_channels, args.channels, device)
            for name in PRETRAIN_OPS + HOLDOUT_OPS}


def observe(op, graph):
    """Generate the observation D = corrupt(A(y)) for a processed batch."""
    fn = getattr(op, 'corrupt', None)
    if fn is not None:
        return fn(graph.y, graph.edge_index, graph.edge_weight)
    return op.forward(graph.y, graph.edge_index, graph.edge_weight, emb=False)


def losses(X, D_rec, graph, D):
    """Supervised loss follows upstream (CE for classification); the data-
    consistency term is normalized MSE in measurement space for every operator
    (uniform across noisy / nonlinear observations)."""
    if args.classify:
        loss_X = F.cross_entropy(X, graph.y)
    else:
        loss_X = F.mse_loss(X, graph.y) / F.mse_loss(torch.zeros_like(graph.y), graph.y)
    loss_data = F.mse_loss(D_rec, D) / (F.mse_loss(torch.zeros_like(D), D) + 1e-12)
    return loss_X, loss_data


def accuracy(X, graph):
    if not args.classify:
        return 0.0
    pred = torch.argmax(X, dim=-1)
    return (pred == torch.argmax(graph.y, dim=-1)).float().mean().item()


def _pad_onehot(graph, label_channels):
    """process_data infers one-hot width from the classes present in the batch;
    pad so a batch that happens to miss the top class keeps a fixed width."""
    if args.classify and graph.y.shape[-1] < label_channels:
        pad = label_channels - graph.y.shape[-1]
        graph.y = F.pad(graph.y, (0, pad))
    return graph


def run_batch(net, ops, name, graph, train_mode):
    """One forward pass with operator `name`; returns (loss, loss_X, acc)."""
    op = ops[name]
    graph = process_data(args, graph)
    graph = _pad_onehot(graph, LABEL_CHANNELS[0])
    op = apply_batch_modifiers(graph, op, args)
    graph = graph.to(device)
    with torch.no_grad():
        D = observe(op, graph)
    net.set_operator(op, name)
    X, _, _ = net(D, graph.edge_index, graph.edge_weight, graph.x)
    D_rec = net.forOp.forward(X, graph.edge_index, graph.edge_weight, emb=False)
    loss_X, loss_data = losses(X, D_rec, graph, D)
    loss = loss_X + loss_data
    return loss, loss_X.item(), accuracy(X, graph)


##################################
######## TRAIN / EVAL ############
##################################
def train_steps(net, ops, op_names, loader, optimizer, n_steps=None, rng=None, label=''):
    """Train over the loader, resampling the operator per batch ("A changes from
    example to example"). If n_steps is given, stop after that many updates."""
    net.train()
    step, tot_loss, tot_acc = 0, 0.0, 0.0
    for graph in loader:
        name = op_names[int(rng.integers(len(op_names)))] if len(op_names) > 1 else op_names[0]
        optimizer.zero_grad()
        loss, _, acc = run_batch(net, ops, name, graph, train_mode=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        tot_loss += loss.item(); tot_acc += acc; step += 1
        if step % 100 == 0:
            print(f'  [{label}] step {step} loss {tot_loss/step:.4f} acc {tot_acc/step:.3f}', flush=True)
        if n_steps is not None and step >= n_steps:
            break
    return tot_loss / max(step, 1), tot_acc / max(step, 1)


@torch.no_grad()
def eval_op(net, ops, name, loader):
    net.eval()
    tot_lx, tot_acc, nb = 0.0, 0.0, 0
    for bi, graph in enumerate(loader):
        if bi >= args.eval_batches:
            break
        _, lx, acc = run_batch(net, ops, name, graph, train_mode=False)
        tot_lx += lx; tot_acc += acc; nb += 1
    return tot_lx / max(nb, 1), tot_acc / max(nb, 1)


def finetune(net, ops, name, support_graphs, steps):
    """Fixed-schedule fine-tuning on the support set (no validation peeking)."""
    from torch_geometric.loader import DataLoader
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.wd, amsgrad=True, eps=1e-3)
    loader = DataLoader(support_graphs, batch_size=args.train_batch_size, shuffle=True)
    net.train()
    step = 0
    while step < steps:
        for graph in loader:
            optimizer.zero_grad()
            loss, _, _ = run_batch(net, ops, name, graph, train_mode=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step >= steps:
                break
    return net


def classical_eval(ops, name, loader, label_channels):
    """Tikhonov baseline (upstream Laplace_noReg_Net), no learning."""
    cargs = copy.copy(args)
    cargs.method = 'tikhonov_regularization'
    cargs.task = name                 # anything but 'edgeRecovery'
    cargs.solveIter = 200             # descent iterations for the classical solve
    cargs.LapNoRegNet_tol = 0.0025
    net = networks.Laplace_noReg_Net(ops[name], cargs, reg='tikhonov_regularization',
                                     device=device)
    tot_lx, tot_acc, nb = 0.0, 0.0, 0
    for bi, graph in enumerate(loader):
        if bi >= args.eval_batches:
            break
        graph = process_data(args, graph)
        graph = _pad_onehot(graph, label_channels)
        op = apply_batch_modifiers(graph, ops[name], args)
        graph = graph.to(device)
        with torch.no_grad():
            D = observe(op, graph)
        X = net(D, graph)
        lx, _ = losses(X, D, graph, D)
        tot_lx += lx.item(); tot_acc += accuracy(X, graph); nb += 1
    return tot_lx / max(nb, 1), tot_acc / max(nb, 1)


##################################
############# MAIN ###############
##################################
def support_sets(train_dataset, N, seed, draw):
    g = torch.Generator().manual_seed(100000 + seed * 1000 + draw)
    idx = torch.randperm(len(train_dataset), generator=g)[:N].tolist()
    return [train_dataset[i] for i in idx]


def main():
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np_rng = np.random.default_rng(seed)
        train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels = \
            get_data_and_loaders(args)
        LABEL_CHANNELS[0] = label_channels
        if args.train_frac < 1.0:
            from torch_geometric.loader import DataLoader
            n_use = max(1, int(args.train_frac * len(train_dataset)))
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(train_dataset), generator=g)[:n_use].tolist()
            train_loader = DataLoader([train_dataset[i] for i in idx],
                                      batch_size=args.train_batch_size, shuffle=True)
        ops = make_ops(label_channels)

        # ---------- pretrain (cached) ----------
        net = build_model(label_channels, feat_channels)
        n_params = count_trainable_parameters(net)
        print(f'[seed {seed}] model params: {n_params}')
        tag = (f"{args.dataset}_{'+'.join(PRETRAIN_OPS)}_c{args.channels}_l{args.layers}"
               f"_cond{args.op_conditioning}_e{args.pretrain_epochs}_tf{args.train_frac}_s{seed}")
        ckpt = os.path.join(args.ckpt_dir, f'pretrain_{tag}.pt')
        if os.path.exists(ckpt):
            net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f'[seed {seed}] loaded cached pretrain: {ckpt}')
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd,
                                         amsgrad=True, eps=1e-3)
            for epoch in range(args.pretrain_epochs):
                l, a = train_steps(net, ops, PRETRAIN_OPS, train_loader, optimizer,
                                   rng=np_rng, label=f'pretrain s{seed} e{epoch}')
                print(f'[seed {seed}] pretrain epoch {epoch}: loss {l:.4f} acc {a:.3f}', flush=True)
            torch.save(net.state_dict(), ckpt)
        pretrain_state = copy.deepcopy(net.state_dict())

        # ---------- in-distribution sanity (frozen) ----------
        for name in PRETRAIN_OPS:
            lx, acc = eval_op(net, ops, name, test_loader)
            rows.append(dict(dataset=args.dataset, seed=seed, op=name, method='pretrained_indist',
                             N='', draw='', acc=acc, loss_X=lx, n_params=n_params))
            print(f'[seed {seed}] in-dist {name}: acc {acc:.3f}')

        # ---------- transfer to holdouts ----------
        for hold in HOLDOUT_OPS:
            lx, acc = eval_op(net, ops, hold, test_loader)
            rows.append(dict(dataset=args.dataset, seed=seed, op=hold, method='pretrained_zeroshot',
                             N=0, draw='', acc=acc, loss_X=lx, n_params=n_params))
            print(f'[seed {seed}] zero-shot {hold}: acc {acc:.3f}')

            net.data_step = False
            lx, acc = eval_op(net, ops, hold, test_loader)
            net.data_step = True
            rows.append(dict(dataset=args.dataset, seed=seed, op=hold, method='pretrained_zeroshot_recipe',
                             N=0, draw='', acc=acc, loss_X=lx, n_params=n_params))
            print(f'[seed {seed}] zero-shot recipe {hold}: acc {acc:.3f}')

            for N in FEW_SHOT_N:
                for draw in range(args.n_support_draws):
                    support = support_sets(train_dataset, N, seed, draw)
                    variants = [
                        ('pretrained_fewshot', True, True),
                        ('pretrained_fewshot_recipe', True, False),
                        ('fromscratch_fewshot', False, True),
                    ]
                    for method, use_pretrain, data_step in variants:
                        torch.manual_seed(seed * 7919 + draw)
                        m = build_model(label_channels, feat_channels)
                        if use_pretrain:
                            m.load_state_dict(pretrain_state)
                        m.data_step = data_step
                        m = finetune(m, ops, hold, support, args.ft_steps)
                        lx, acc = eval_op(m, ops, hold, test_loader)
                        rows.append(dict(dataset=args.dataset, seed=seed, op=hold, method=method,
                                         N=N, draw=draw, acc=acc, loss_X=lx, n_params=n_params))
                        print(f'[seed {seed}] {method} {hold} N={N} draw={draw}: acc {acc:.3f}', flush=True)

            if not args.skip_classical and seed == SEEDS[0]:
                lx, acc = classical_eval(ops, hold, test_loader, label_channels)
                rows.append(dict(dataset=args.dataset, seed=seed, op=hold, method='classical_tikhonov',
                                 N='', draw='', acc=acc, loss_X=lx, n_params=0))
                print(f'classical {hold}: acc {acc:.3f}')

        # write incrementally so partial runs are not lost
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'[seed {seed}] wrote {len(rows)} rows -> {out_csv}')

    print('done.')


if __name__ == '__main__':
    main()
