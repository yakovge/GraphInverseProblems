import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import Node2Vec

from torch_cluster import knn, random_walk


class graphEmbed(nn.Module):
    def __init__(self, embdsize, nin=3, learned=True, device='cuda'):
        super(graphEmbed, self).__init__()
        if learned:
            self.k = 9
            self.Emb = (nn.init.xavier_uniform_(torch.empty(embdsize, nin, device=device)))
            # id = torch.zeros((self.k, self.k))
            # id[self.k // 2, self.k // 2] = 1
            # self.Emb[0, 0, :, :] = id
            # self.Emb[1, 1, :, :] = id
            # self.Emb[2, 2, :, :] = id

            self.Emb = nn.Parameter(self.Emb)
            self.bias_back = nn.Parameter(torch.zeros(nin))
            self.bias_for = nn.Parameter(torch.zeros(embdsize))

        else:
            self.Emb = torch.eye(nin, nin, device=device)  # .unsqueeze(-1).unsqueeze(-1)
            self.bias_back = torch.zeros(embdsize)
            self.bias_for = torch.zeros(nin)

    def forward(self, I):
        Emb = self.Emb.to(I.device)
        #I = F.conv1d(I.t().unsqueeze(0), weight=Emb.unsqueeze(-1))  # , bias=self.bias_for

        #I = I.squeeze().t()
        I = I @ self.Emb
        return I  # I @ Emb#F.conv2d(I, Emb, padding=self.Emb.shape[-1] // 2)

    def backward(self, I):
        Emb = self.Emb.to(I.device)
        #I = F.conv_transpose1d(I.t().unsqueeze(0), weight=Emb.unsqueeze(-1))  # , bias=self.bias_back
        #I = I.squeeze().t()
        I = I @ Emb.t()
        return I  # I @ Emb.t()#F.conv_transpose2d(I, Emb, padding=self.Emb.shape[-1] // 2)


class maskImage(nn.Module):
    def __init__(self, ind, imsize, embdsize, nin=3, device='cuda', learnEmb=True):
        super(maskImage, self).__init__()
        ind = ind.to(device)
        self.ind = ind
        self.imsize = imsize
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
        Ic = I.reshape(I.shape[0], I.shape[1], -1)
        Ic = Ic[:, :, self.ind]
        return Ic

    def adjoint(self, Ic, emb=True):
        I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        I[:, :, self.ind] = Ic
        I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class graphMask(nn.Module):
    def __init__(self, ind, embdsize, nin=3, device='cuda', learnEmb=True):
        super(graphMask, self).__init__()
        ind = ind.to(device)
        self.ind = ind 
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        Ic = torch.zeros_like(I)
        Ic[self.ind, :] = I[self.ind, :]
        return Ic
    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        I = torch.zeros_like(Ic)
        I[self.ind, :] = Ic[self.ind, :]
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class graphPath(nn.Module):
    def __init__(self, embdsize, nin=3, device='cuda', learnEmb=True, pathLength=3):
        super(graphPath, self).__init__()
        self.pathLength = pathLength
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb
        self.node_seq = None
        self.device = device

    def gen_paths(self, nnodes, edge_index):
        L = self.pathLength
        # nodesIdx = torch.arange(nnodes).cuda() 
        # edge_index = edge_index.cuda()  # must be on the same device as nodesIdx
        nodesIdx = torch.arange(nnodes, device=self.device)
        edge_index = edge_index.to(self.device)
        # nodesIdx = torch.repeat_interleave(nodesIdx, dim=0, repeats=4)
        node_seq = random_walk(edge_index[0, :], edge_index[1, :], start=nodesIdx,
                               walk_length=L - 1, p=1, q=1).t().to(self.device)
        self.node_seq = node_seq

        T = torch.zeros(nnodes, nnodes, device=self.device)

        L = self.node_seq.shape[0]
        batch_indices = torch.arange(self.node_seq.shape[1]).to(T.device)
        batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L).flatten().to(T.device)
        batch_indices_j = self.node_seq.flatten().to(T.device)
        # for iii, vec in enumerate(self.node_seq.t()):
        #    for s in vec:
        #        T[iii, s] += 0.5
        T[batch_indices_i, batch_indices_j] += 0.5

        self.T = T
        return node_seq

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        # Ic = I.reshape(I.shape[0], I.shape[1], -1)
        # Ic = Ic[:, :, self.ind]
        # Ic = I[self.ind, :]
        if False:
            Ic = 0.5 * I[self.node_seq, :].sum(dim=0)
        else:
            # Test dense:
            # nnodes = len(edge_index.unique())

            # T = torch.zeros(nnodes, nnodes, device=I.device)
            # for iii, vec in enumerate(self.node_seq):
            #     for s in vec:
            #         T[iii, s] += 0.5

            Ic = self.T @ I #self.T @ I.to(self.T.device)

        return Ic

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        nnodes = len(edge_index.unique())
        # I = torch.zeros(nnodes, Ic.shape[1], device=Ic.device)

        # I[self.ind, :] = Ic
        # I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        if False:
            # batch_indices = torch.arange(node_seq.shape[0])
            # batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)
            # adj_for_x = 0.5 * for_x[batch_indices_i.t(), :].sum(dim=0)

            L = self.node_seq.shape[0]
            batch_indices = torch.arange(self.node_seq.shape[1])
            batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)

            # batch_indices_perm[]
            I = 0.5 * Ic[batch_indices_i.t(), :].sum(dim=0)  # P @ x , P shape nxn, x shape nxc,
            # 0 [ 1 3 4]
            # 1 [ 2 3 5]
            # 0 1 0 1 1 0 Beams x Length
            # 0 0 1 1 0 1
        else:
            # Test dense:
            # T = torch.zeros(nnodes, nnodes, device=Ic.device)
            # for iii, vec in enumerate(self.node_seq):
            #     for s in vec:
            #         T[iii, s] += 0.5

            I = self.T.t() @ Ic

        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I

    def forward2(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        # Ic = I.reshape(I.shape[0], I.shape[1], -1)
        # Ic = Ic[:, :, self.ind]
        # Ic = I[self.ind, :]
        Ic = I[self.node_seq, :]  # / self.node_seq.shape[0]
        Ic = 0.5 * (Ic[:-1, :] + Ic[1:, :])
        Ic = Ic.sum(dim=0)  # / self.node_seq.shape[0]

        return Ic

    def adjoint2(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        nnodes = len(edge_index.unique())
        # I = torch.zeros(nnodes, Ic.shape[1], device=Ic.device)

        # I[self.ind, :] = Ic
        # I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        L = self.node_seq.shape[0]
        batch_indices = torch.arange(self.node_seq.shape[1])
        batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)
        I = Ic[batch_indices_i.t(), :]
        I = 0.5 * (I[:-1, :] + I[1:, :])
        I = I.sum(dim=0)  # / self.node_seq.shape[0]  # P @ x , P shape nxn, x shape nxc,
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class blur(nn.Module):
    def __init__(self, K, embdsize, learnEmb=True, device='cuda'):
        super(blur, self).__init__()
        K = K.to(device)
        nin = 3
        self.K = K
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)
        Ic = F.conv2d(I, self.K)

        return Ic

    def adjoint(self, Ic, emb=True):
        I = F.conv_transpose2d(Ic, self.K)
        if emb:
            I = self.Emb.backward(I)
        return I


class graph_smooth(nn.Module):
    def __init__(self, nin, embdsize, learnEmb=True, device='cuda', k=3):
        super(graph_smooth, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, self.nin, learned=learnEmb, device=device)
        self.k = k
        self.learnEmb = learnEmb

    def forward(self, node_features, edge_index, edge_weights, emb=True):
        if emb and self.learnEmb:
            node_features = self.Emb(node_features)
            
        num_nodes = node_features.shape[0]
        
        # Construct PyTorch sparse adjacency matrix to prevent OOM
        A_sparse = torch.sparse_coo_tensor(
            edge_index, 
            edge_weights, 
            (num_nodes, num_nodes)
        )
        
        # Iteratively apply sparse matrix multiplication
        node_features_smooth = node_features.clone()
        for _ in range(self.k):
            node_features_smooth = torch.sparse.mm(A_sparse, node_features_smooth)
            
        return node_features_smooth

    def adjoint(self, node_features, edge_index, edge_weights, emb=True):
        num_nodes = node_features.shape[0]
        
        # The adjoint of A^k is (A^T)^k. Transpose the edge indices.
        edge_index_t = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        A_t_sparse = torch.sparse_coo_tensor(
            edge_index_t, 
            edge_weights, 
            (num_nodes, num_nodes)
        )
        
        # Iteratively apply transposed sparse matrix multiplication
        for _ in range(self.k):
            node_features = torch.sparse.mm(A_t_sparse, node_features)
            
        if emb and self.learnEmb:
            node_features = self.Emb.backward(node_features)
            
        return node_features


class graph_edgeRecovery(nn.Module):
    def __init__(self, nin, embdsize, learnEmb=True, K=1, device='cuda'):
        super(graph_edgeRecovery, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, self.nin, learned=learnEmb,device=device)
        self.K = K
        self.learnEmb = learnEmb

    def forward(self, node_features, edge_index, edge_weights, emb=True):
        # xN' = P^k(xE)xN
        # xN = node_features
        if emb:
            node_features = self.Emb(node_features)
        # node_features = F.conv2d(node_features, self.K)

        A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster
        self.A = A
        node_features_smooth = node_features.clone()
        data_out = []
        for i in range(self.K):
            node_features_smooth = A @ node_features_smooth
            data_out.append(node_features_smooth)
        data_out = torch.stack(data_out, dim=0)  # [K,N,C]
        self.data_out = data_out
        self.edge_weights = edge_weights
        return data_out

    def adjoint(self, seq, node_features, edge_index, edge_weights, emb=True):
        # call forward

        # I = F.conv_transpose2d(Ic, self.K)
        N = len(edge_index.unique())
        C = node_features.shape[-1]
        K = self.K

        # grad(f)^T @ seq
        from torch.autograd import grad
        data_out = self.forward(node_features, edge_index, edge_weights, emb=emb)
        f = (data_out * seq).sum()
        #JtR = torch.autograd.functional.jvp(self.forward, (node_features, edge_index, edge_weights), seq)
        JtR = grad(f, edge_weights, create_graph=True)[0]
        JtR = torch.stack([JtR, JtR], dim=-1)
        # A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        # A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster
        # A = self.A
        # node_features = A.t()@(A.t()@((A.t() @ node_features))) #A.t() @ node_features

        # for i in range(self.K):
        #    node_features = A.t() @ node_features

        if emb:
            JtR = self.Emb.backward(JtR)
        return JtR


class contactMap(nn.Module):
    def __init__(self, embdsize, sigma=1.0, device='cuda'):
        super(contactMap, self).__init__()
        self.sigma = sigma

    def forward(self, X, emb=True):
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        XX = Xsq + Xsq.transpose(1, 2)

        XTX = torch.bmm(X.transpose(2, 1), X)
        D = torch.relu(XX - 2 * XTX)

        return D

    def adjoint(self, X, dV):
        n1 = X.shape[-1]
        e2 = torch.ones(3, 1)
        e1 = torch.ones(n1, 1)
        E12 = e1 @ e2.t()
        E12 = E12.unsqueeze(0)
        E12 = torch.repeat_interleave(E12, X.shape[0], dim=0)

        P1 = 2 * X * (torch.bmm(dV, E12).transpose(-1, -2) + torch.bmm(dV.transpose(-1, -2), E12).transpose(-1, -2))
        P2 = 2 * torch.bmm(dV.transpose(-2, -1) + dV, X.transpose(-2, -1)).transpose(-2, -1)
        dX = P1 - P2

        return dX

    def jacMatVec(self, X, dX):
        XdX = torch.sum(X * dX, dim=-2, keepdim=True)
        XdXT = torch.bmm(X.transpose(-1, -2), dX)
        dXXT = torch.bmm(dX.transpose(-1, -2), X)
        V = 2 * XdX + 2 * XdX.transpose(-1, -2) - 2 * XdXT - 2 * dXXT
        return V


class blurFFT(nn.Module):
    def __init__(self, embdsize, nin, learnEmb=True, dim=256, device='cuda'):
        super(blurFFT, self).__init__()
        self.nin = nin
        self.Emb = Embed(embdsize, nin, learned=learnEmb)
        self.dim = dim
        self.device = device

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)
        P, center = self.psfGauss(self.dim)

        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0, 1])).unsqueeze(0).unsqueeze(0)
        B = torch.real(torch.fft.ifft2(S * torch.fft.fft2(I)))

        return B

    def adjoint(self, Ic, emb=True):
        I = self.forward(Ic, emb=False)
        if emb:
            I = self.Emb.backward(I)
        return I

    def psfGauss(self, dim, s=[2.0, 2.0]):
        m = dim
        n = dim

        x = torch.arange(-n // 2 + 1, n // 2 + 1, device=self.device)
        y = torch.arange(-n // 2 + 1, n // 2 + 1, device=self.device)
        X, Y = torch.meshgrid(x, y)

        PSF = torch.exp(-(X ** 2) / (2 * s[0] ** 2) - (Y ** 2) / (2 * s[1] ** 2))
        PSF = PSF / torch.sum(PSF)

        # Get center ready for output.
        center = [1 - m // 2, 1 - n // 2]

        return PSF, center


class radonTransform(nn.Module):
    def __init__(self, embdsize, nin, learnEmb=True, device='cuda'):
        super(radonTransform, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb)
        self.device = device

        A = io.loadmat('radonMat18.mat')
        A = A['A']
        A = torch.tensor(A, device=device)
        A = A.type(torch.cuda.FloatTensor)
        A = A.to_sparse()
        self.A = A

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)

        T = I.view(I.shape[0], I.shape[1], -1)
        Tt = T.transpose(1, 2)
        Ttt = Tt.transpose(0, 1)
        Tttt = Ttt.reshape(Ttt.shape[0], -1)
        Yttt = torch.matmul(self.A, Tttt)
        Ytt = Yttt.reshape(Yttt.shape[0], -1, 3)
        Yt = Ytt.transpose(0, 1)
        Y = Yt.transpose(1, 2)
        Y = Y.reshape(Y.shape[0], 3, 18, 139)

        return Y

    def adjoint(self, Ic, emb=True):
        T = Ic.view(Ic.shape[0], Ic.shape[1], -1)
        Tt = T.reshape(-1, T.shape[2]).t()
        Yt = self.A.t() @ Tt
        Y = Yt.t()
        I = Y.reshape(-1, 3, 96, 96)
        if emb:
            I = self.Emb.backward(I)
        return I


class AddNoise(nn.Module):
    """Adds Gaussian noise to the input tensor."""
    def __init__(self, nin, embdsize, noise_std=0.1, device='cuda', learnEmb=True):
        super(AddNoise, self).__init__()
        self.noise_std = noise_std
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def corrupt(self, I):
        """Use this ONLY once to generate the noisy measurement D."""
        return I + self.noise_std * torch.randn_like(I)

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        """The mathematical operator for denoising is just the Identity."""
        if emb and self.learnEmb:
            I = self.Emb(I)
        return I

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        """The adjoint of the Identity operator is also the Identity."""
        if emb and self.learnEmb:
            Ic = self.Emb.backward(Ic)
        return Ic

class SensorRecovery(nn.Module):
    """Masks non-sensor nodes while retaining tensor shape."""
    def __init__(self, sensor_indices, nin, embdsize, device='cuda', learnEmb=True):
        super(SensorRecovery, self).__init__()
        self.sensor_indices = sensor_indices.to(device)
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
        Ic = torch.zeros_like(I)
        Ic[self.sensor_indices] = I[self.sensor_indices]
        return Ic

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        I = torch.zeros_like(Ic)
        I[self.sensor_indices] = Ic[self.sensor_indices]
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I

class PDESSM(nn.Module):
    """
    Applies PDE spatial mixing using the true topology of the graph via the Graph Fourier Transform.
    Automatically vectorizes over batched PyG graphs.
    """
    def __init__(self, nin, embdsize, dim, tau=1.0, device='cuda', learnEmb=True):
        super(PDESSM, self).__init__()
        # 'dim' is the exact number of nodes per graph (e.g., 20 for CPOX, 207 for METRLA)
        self.nodes_per_graph = dim
        
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb
        self.device = device
        self.tau = tau
        
        # PDE parameters in the graph spectral domain
        self.K = nn.Parameter(torch.ones(1, device=device))  # Diffusion (low-pass over eigenvalues)
        self.r = nn.Parameter(torch.zeros(1, device=device)) # Reaction (global amplification/suppression)
        self._eig_cache = {}

    def _get_spectral_components(self, edge_index, edge_weight, batch_size):
        """
        Computes the batched Normalized Graph Laplacian and its eigendecomposition.
        """
        # Create a batch vector to map edges back to isolated dense graphs
        batch_vec = torch.arange(batch_size, device=self.device).repeat_interleave(self.nodes_per_graph)
        
        # Convert edge_index to batched dense adjacency matrix: [Batch, N, N]
        from torch_geometric.utils import to_dense_adj
        A = to_dense_adj(edge_index, batch_vec, edge_attr=edge_weight, max_num_nodes=self.nodes_per_graph)
        A = 0.5 * (A + A.transpose(1, 2))  # directed graphs (WikiMaths, Montevideo): eigh/self-adjointness need a symmetric L
        
        # Compute Degree matrix D^{-1/2}
        deg = A.sum(dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        I_mat = torch.eye(self.nodes_per_graph, device=self.device).unsqueeze(0)
        L_sym = I_mat - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)
        
        # Eigendecomposition (Graph Fourier Basis)
        # evals: [Batch, N] (frequencies), evecs: [Batch, N, N] (Fourier basis)
        evals, evecs = torch.linalg.eigh(L_sym)
        
        return evals, evecs

    def _apply_graph_pde(self, I, edge_index, edge_weight):
        """
        Applies the PDE-SSM filter in the Graph Spectral Domain.
        """
        channels = I.shape[-1]
        batch_size = I.shape[0] // self.nodes_per_graph
        
        if I.shape[0] % self.nodes_per_graph != 0:
            raise ValueError(f"PDESSM expects fixed graph sizes of {self.nodes_per_graph} nodes.")
            
        # 1. Get graph frequencies (evals) and spatial basis (evecs)
        # the Laplacian has no learned parameters and the graphs are static: eigendecompose once per graph/batch size
        key = (self.nodes_per_graph, batch_size, edge_index.shape[1], int(edge_index.sum()))
        if key not in self._eig_cache:
            with torch.no_grad():
                self._eig_cache[key] = self._get_spectral_components(edge_index, edge_weight, batch_size)
        evals, evecs = self._eig_cache[key]
        
        # 2. Compute the Green's function symbol over the graph eigenvalues
        # G(\lambda) = exp(tau * (-K * \lambda + r))
        Lambda_k = -self.K * evals + self.r
        G_k = torch.exp(self.tau * Lambda_k) # Shape: [Batch, N]
        
        # 3. Reshape input to [Batch, N, Channels]
        I_grid = I.view(batch_size, self.nodes_per_graph, channels)
        
        # 4. Graph Fourier Transform (GFT): \hat{I} = U^T I
        I_hat = torch.bmm(evecs.transpose(1, 2), I_grid)
        
        # 5. Apply the PDE-SSM spectral filter
        I_filtered = I_hat * G_k.unsqueeze(-1)
        
        # 6. Inverse Graph Fourier Transform (IGFT): I_{out} = U \hat{I}_{filtered}
        I_out = torch.bmm(evecs, I_filtered)
        
        # 7. Flatten back to PyG structure [Batch*Nodes, Channels]
        return I_out.view(-1, channels)

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
            
        return self._apply_graph_pde(I, edge_index, edge_weight)

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # For undirected graphs, the normalized Laplacian yields strictly real eigenvalues.
        # Therefore, the Green's function symbol is entirely real, and the operator is self-adjoint.
        I = self._apply_graph_pde(Ic, edge_index, edge_weight)
        
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I