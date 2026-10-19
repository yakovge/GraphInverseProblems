"""Save/load round trip: a checkpoint must reproduce the predictions of the net that wrote it, for every task.
Run from the repo root:  python src/check_save_load.py   (CPU, ~20 s). Exits non-zero on failure."""
import sys, os, tempfile
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from plot_fixed import make_args
from utils import get_forward_op, get_network, process_data, task_specific_modifiers
from checkpoints import load_foundation_checkpoint

torch.manual_seed(0)
args = make_args(); args.layers, args.channels, args.num_nodes = 4, 8, 20
ops = get_forward_op(args, args.channels, 1, 'cpu')
net = get_network(args, ops, args.channels, 1, 3, 'cpu')  # CPOX features: sin, cos week + observation flag
net.freeze_backbone()                      # task heads in use, each perturbed so they differ
for h in list(net.task_heads.values()) + list(net.task_encoders.values()):
    for p in h.parameters():
        p.data.add_(torch.randn_like(p))
net.eval()
for bad in ('current_head.weight', 'current_forward_op.K', 'solver.forOp.K'):
    assert bad not in net.state_dict(), f"{bad} is in the state_dict: alias / physics leaked into the checkpoint"
assert not any(n.endswith('.K') or n.endswith('.r') for n, _ in net.named_parameters() if 'forward_op' in n or 'forOp' in n)

# a batch of 4 random 20-node graphs shaped like CPOX (x = [count, week])
graphs = [Data(x=torch.stack([torch.rand(20) * 100, torch.full((20,), float(w))], 1),
               edge_index=torch.randint(0, 20, (2, 60))) for w in range(4)]
from torch_geometric.loader import DataLoader
raw = next(iter(DataLoader(graphs, batch_size=4)))
net.set_task('denoising')                  # save while a NON-inpainting task is active (the failing case)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'epoch_1_noise_test.pth')
    torch.save(net.state_dict(), path)
    loaded = load_foundation_checkpoint(path, make_args(), ops, 'cpu', 1, 3)
    for p1, p2 in zip(net.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)
    worst = 0.0
    for k, (op, task) in enumerate(ops):
        torch.manual_seed(k)
        g, op = task_specific_modifiers(process_data(args, raw.clone()), args, op, None)
        b = op(g.y, g.edge_index, g.edge_weight, emb=False)
        with torch.no_grad():
            net.set_task(task); X1, _, _ = net(b, g.edge_index, g.edge_weight, g.x)
            loaded.set_task(task); X2, _, _ = loaded(b, g.edge_index, g.edge_weight, g.x)
        worst = max(worst, (X1 - X2).abs().max().item())
        print(f"  {task:20s} max |X_saved - X_loaded| = {(X1 - X2).abs().max().item():.2e}")
    assert worst == 0.0, "loaded checkpoint does not reproduce the saved model's predictions"
print("OK: save/load round trip is exact for every task")
