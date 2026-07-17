"""Correctness tests for the foundation-study forward operators.

Run:  python test_foundation_ops.py   (from src/; CPU, a few seconds)

Checks:
  1. Adjoint dot-product test <A x, r> == <x, A^T r> for every linear operator
     (new ones and the upstream ones we reuse).
  2. graphMaxPool: forward matches a brute-force dense neighborhood max, the
     VJP adjoint is finite/nonzero and works inside torch.no_grad(), and it is
     linear in r (it is a fixed matrix once the linearization point is fixed).
"""
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import erdos_renyi_graph, remove_self_loops

from foundation_ops import (graphIdentity, graphMaskSmooth, graphMaxPool,
                            make_foundation_op)

torch.manual_seed(0)
DEVICE = "cpu"


def make_graph(n=60, p=0.1, channels=6):
    edge_index = erdos_renyi_graph(n, p)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=True)
    x = torch.randn(n, channels)
    return edge_index, edge_weight, x


def adjoint_dot_test(op, name, ei, ew, x):
    y = op.forward(x, ei, ew, emb=False)
    r = torch.randn_like(y)
    lhs = (y * r).sum()
    rhs = (x * op.adjoint(r, ei, ew, emb=False)).sum()
    rel = (lhs - rhs).abs() / (lhs.abs() + 1e-12)
    assert rel < 1e-4, f"{name}: adjoint test failed, rel err {rel:.2e}"
    print(f"  adjoint ok: {name:12s} rel err {rel:.2e}")


def test_linear_adjoints():
    ei, ew, x = make_graph()

    class A:  # minimal args stand-in for make_foundation_op
        blur_count = 4
        pathLength = 8
        denoise_sigma = 0.3

    for name in ["denoise", "mask", "smooth", "maskSmooth"]:
        op = make_foundation_op(name, A, label_channels=x.shape[1],
                                hid_channels=16, device=DEVICE)
        if name in ("mask", "maskSmooth"):
            op.ind = torch.randperm(x.shape[0])[:20]
        adjoint_dot_test(op, name, ei, ew, x)

    try:  # path needs torch_cluster.random_walk
        op = make_foundation_op("path", A, label_channels=x.shape[1],
                                hid_channels=16, device=DEVICE)
        op.gen_paths(nnodes=x.shape[0], edge_index=ei)
        adjoint_dot_test(op, "path", ei, ew, x)
    except ImportError as e:
        print(f"  path skipped (torch_cluster not available: {e})")


def test_maxpool():
    ei, ew, x = make_graph()
    n = x.shape[0]
    op = graphMaxPool()
    y = op.forward(x, ei, ew, emb=False)

    # brute-force reference: max over in-neighbors ∪ self, per channel
    ref = x.clone()
    for i in range(n):
        nbrs = ei[0][ei[1] == i]
        vals = torch.cat([x[nbrs], x[i:i + 1]], dim=0)
        ref[i] = vals.max(dim=0).values
    assert torch.allclose(y, ref, atol=1e-6), "maxpool forward mismatch"
    print("  maxpool forward matches brute force")

    # VJP adjoint: finite, nonzero, works under no_grad (regression test for the
    # eval-context bug we hit in the v1 project)
    r = torch.randn_like(y)
    with torch.no_grad():
        g = op.adjoint(r, ei, ew, emb=False)
    assert torch.isfinite(g).all() and g.abs().sum() > 0, "maxpool VJP degenerate"
    print("  maxpool VJP finite/nonzero under no_grad")

    # linearity in r at a fixed linearization point
    r2 = torch.randn_like(y)
    g1 = op.adjoint(r, ei, ew, emb=False)
    g2 = op.adjoint(r2, ei, ew, emb=False)
    g12 = op.adjoint(r + 2.0 * r2, ei, ew, emb=False)
    assert torch.allclose(g12, g1 + 2.0 * g2, atol=1e-5), "VJP not linear in r"
    print("  maxpool VJP linear in r")

    # gradients flow through adjoint's r-dependence during training
    rt = torch.randn_like(y).requires_grad_(True)
    g = op.adjoint(rt, ei, ew, emb=False)
    g.sum().backward()
    assert rt.grad is not None and torch.isfinite(rt.grad).all()
    print("  maxpool VJP backprops to r")


if __name__ == "__main__":
    test_linear_adjoints()
    test_maxpool()
    print("all foundation-op tests passed")
