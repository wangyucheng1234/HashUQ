import torch
import numpy as np
from lap import lapjv
from lap import lapmod
import random

# def wasserstein1d(x, y):
#     """Compute wasserstein loss in 1D"""
#     x1, _ = torch.sort(x, dim=0)
#     y1, _ = torch.sort(y, dim=0)
#     z = (x1-y1).view(-1)
#     n = x.size(0)
#     return torch.dot(z, z)/n

def wasserstein1d(x, y, aggregate=True):
    """Compute wasserstein loss in 1D"""
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    n = x.size(0)
    if aggregate:
        z = (x1-y1).view(-1)
        return torch.dot(z, z)/n
    else:
        return (x1-y1).square().sum(0)/n

def compute_g_sliced_wasserstein_loss(nprojections, real_features, gen_features, device='cuda'):
    bsize, dim = gen_features.size()

    theta = torch.randn((dim, nprojections),
                            requires_grad=False,
                            device=device)
    theta = theta/torch.norm(theta, dim=0)[None, :]
    xgen_1d = gen_features.view(-1, dim)@theta
    xreal_1d = real_features.view(-1, dim)@theta
    
    gloss = wasserstein1d(xreal_1d, xgen_1d)
    
    return gloss

EPSILON = 1e-12
def dense_wasserstein_distance(cost_matrix, device):
    num_pts = len(cost_matrix);
    C_cpu = cost_matrix.detach().cpu().numpy();
    C_cpu *= 100000 / (C_cpu.max() + EPSILON)
    lowest_cost, col_ind_lapjv, row_ind_lapjv = lapjv(C_cpu);
    
    loss = 0.0;
    for i in range(num_pts):
        loss += cost_matrix[i,col_ind_lapjv[i]];
                
    return loss/num_pts;

def compute_frobenius_pairwise_distances_torch(X, Y, device, p=1, normalized=True):
    """Compute pairwise distances between 2 sets of points"""
    assert X.shape[1] == Y.shape[1]
    
    d = X.shape[1]
    dists = torch.zeros(X.shape[0], Y.shape[0], device=device)

    for i in range(X.shape[0]):
        if p == 1:
            dists[i, :] = torch.sum(torch.abs(X[i, :] - Y), dim=1)
        elif p == 2:
            dists[i, :] = torch.sum((X[i, :] - Y) ** 2, dim=1)
        else:
            raise Exception('Distance type not supported: p={}'.format(p))
        
        if normalized:
            dists[i, :] = dists[i, :] / d

    return dists

def compute_g_primal_loss_v2(real_features, gen_features, device='cuda'):
    b_size = real_features.size(0)
    real_features = real_features.view(b_size, -1)
    gen_features = gen_features.view(b_size, -1)
    
    C = compute_frobenius_pairwise_distances_torch(real_features, gen_features, device, p=2, normalized=True)
    gloss = dense_wasserstein_distance(C, device)
    
    return gloss

def quantization_ot_loss(b, device='cuda'):
    real_b = torch.randn(b.shape, device=device).sign()
    return compute_g_primal_loss_v2(real_b, b, device=device)


def quantization_swd_loss(b, nprojections=10000, device='cuda', aggregate=True):
    dim = b.size(1)
    real_b = torch.randn(b.shape, device=device).sign()
    
    theta = torch.randn((dim, nprojections), requires_grad=False, device=device)
    theta = theta/torch.norm(theta, dim=0)[None, :]
    
    xgen_1d = b.view(-1, dim)@theta
    xreal_1d = real_b.view(-1, dim)@theta
    
    if aggregate:
        gloss = wasserstein1d(xreal_1d, xgen_1d) / nprojections
    else:
        gloss = wasserstein1d(xreal_1d, xgen_1d, aggregate=False)
    
    return gloss

def quantization_swdc_loss(b, device='cuda', aggregate=True):
    real_b = torch.randn(b.shape, device=device).sign()
    bsize, dim = b.size()

    if aggregate:
        gloss = wasserstein1d(real_b, b) / dim
    else:
        gloss = wasserstein1d(real_b, b, aggregate=False)

    return gloss

# def quantization_swd_loss(b, nprojections=10000, device='cuda'):
#     real_b = torch.randn(b.shape, device=device).sign()
#     return compute_g_sliced_wasserstein_loss(nprojections, real_b, b, device=device)/nprojections

# def quantization_swdc_loss(b, device='cuda'):
#     real_b = torch.randn(b.shape, device=device).sign()
#     bsize, dim = b.size()

#     gloss = wasserstein1d(real_b, b) / dim

#     return gloss

def quantization_swdc_signed_loss(b, device='cuda'):
    real_b = real_b = torch.sign(b).detach()
    bsize, dim = b.size()

    gloss = wasserstein1d(real_b, b) / dim

    return gloss

def quantization_swd_signed_loss(b, device='cuda'):
    real_b = torch.sign(b).detach()
    bsize, dim = b.size()

    gloss = wasserstein1d(real_b, b) / dim

    return gloss


class conditional_transport(torch.nn.Module):
    def __init__(self, rho = 0.5, phi = 0, device = 'cuda') -> None:
        torch.nn.Module.__init__(self)
        self.softmax_forward = torch.nn.Softmax(dim = 1)
        self.softmax_backward = torch.nn.Softmax(dim = 0)
        self.rho = rho
        #Set phi to be fixed
        self.phi = phi*torch.ones(1).to(device)

    def compute_distance(self, X, Y):
        #d(X, Y): Nx X Ny
        distance = (torch.exp(-self.phi))*torch.sum((X.unsqueeze(1) - Y.unsqueeze(0))**2, dim = 2)
        return distance
    
    def compute_cost(self, X, Y):
        #c(X, Y): Nx X Ny
        cost = torch.sum((X.unsqueeze(1) - Y.unsqueeze(0))**2, dim = 2)
        return cost
    
    def compute_forward_plan(self, distance):
        #pi(Y|X): Ny X Nx
        forward_plan = torch.transpose(self.softmax_forward(distance), 0, 1)
        return forward_plan
    
    def compute_backward_plan(self, distance):
        #pi(X|Y): Nx X Ny
        backward_plan = self.softmax_backward(distance)
        return backward_plan
    
    def compute_ct(self, X, Y):
        assert X.size()[1] == Y.size()[1]
        distance = -self.compute_distance(X, Y)
        f_plan = self.compute_forward_plan(distance)
        b_plan = self.compute_backward_plan(distance)
        cost_matrix = self.compute_cost(X, Y)
        ct_cost = self.rho*torch.sum(torch.transpose(f_plan, 0, 1)*cost_matrix)/(X.size()[0]) + (1 - self.rho)*torch.sum(b_plan*cost_matrix)/(Y.size()[0])
        return ct_cost
    
    def compute_entry_ct(self, X, Y):
        assert X.size()[1] == Y.size()[1]
        ct_cost = []
        for i in range(X.size()[1]):
            distance = -self.compute_distance(X[:, i].unsqueeze(1), Y[:, i].unsqueeze(1))
            f_plan = self.compute_forward_plan(distance)
            b_plan = self.compute_backward_plan(distance)
            cost_matrix = self.compute_cost(X[:, i].unsqueeze(1), Y[:, i].unsqueeze(1))
            ct_cost.append(self.rho*torch.sum(torch.transpose(f_plan, 0, 1)*cost_matrix)/(X.size()[0]) + (1 - self.rho)*torch.sum(b_plan*cost_matrix)/(Y.size()[0]))
        ct_cost = sum(ct_cost)
        return ct_cost
    
    def compute_entry_ct_v2(self, X, Y):
        #Non-for loop version of compute_entry_ct
        assert X.size()[1] == Y.size()[1]
        distance = -(torch.exp(-self.phi))*(X.unsqueeze(1) - Y.unsqueeze(0))**2
        f_plan = self.softmax_forward(distance)
        b_plan = self.softmax_backward(distance)
        cost_matrix = (X.unsqueeze(1) - Y.unsqueeze(0))**2
        ct_cost = self.rho*torch.sum(f_plan*cost_matrix)/(X.size()[0]) + (1 - self.rho)*torch.sum(b_plan*cost_matrix)/(Y.size()[0])
        return ct_cost
    