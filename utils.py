import torch
import numpy as np

def _make_index(row, cols):
    return [(row, c) for c in cols]


def _apply_index(params, index):
    if index == slice(None):
        return params
    if torch.is_tensor(params) and torch.is_tensor(index):
        return params[index[:, 0], index[:, 1]]
    return [params[r][c] for (r, c) in index]


def normalize_phi(phi, lb, ub):
    """
    Normalize phi to the range [-1, 1] based on the provided lower and upper bounds.
    
    Args:
        phi (torch.Tensor): The input tensor to be normalized.
        lb (torch.Tensor): The lower bound tensor.
        ub (torch.Tensor): The upper bound tensor.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    return (phi - lb) / (ub - lb) * 2 - 1

def denormalize_phi(normalized_phi, lb, ub):
    """
    Denormalize phi from the range [-1, 1] back to the original range based on the provided lower and upper bounds.
    
    Args:
        normalized_phi (torch.Tensor): The normalized tensor to be denormalized.
        lb (torch.Tensor): The lower bound tensor.
        ub (torch.Tensor): The upper bound tensor.
    Returns:
        torch.Tensor: The denormalized tensor.
    """
    return (normalized_phi + 1) / 2 * (ub - lb) + lb

def normalize_muons(X, mean=None, std=None):
    """Standardize the kinematic columns and map the pdg column to -sign(pdg).

    The reductions are forced to a float64 accumulator. X is float32 and
    X[..., :-1] is a strided view, so numpy accumulates the sum in float32 and
    over a training array (n_phi x n_muons ~ 1e9 values) the running sum reaches
    ~1e10, where the float32 spacing (~4096) is far larger than the ~50 being
    added: the sum stalls and the returned mean/std are badly wrong -- and wrong
    by an amount that depends on how many muons were passed in. A model trained
    on those statistics is self-consistent only with them, so it looks perfect
    in-sample and cannot be reproduced by any other script.
    """
    if mean is None:
        mean = X[..., :-1].mean(axis=(0, 1), dtype=np.float64, keepdims=True).astype(X.dtype)
    if std is None:
        std = (X[..., :-1].std(axis=(0, 1), dtype=np.float64, keepdims=True)
               + 1e-12).astype(X.dtype)
    X_norm = X.copy()
    X_norm[..., :-1] = (X[..., :-1] - mean) / std
    X_norm[..., -1] = -np.sign(X[..., -1])
    return X_norm, mean, std

def denormalize_muons(X_norm, mean, std):
    X = X_norm.copy()
    X[..., :-1] = X_norm[..., :-1] * std + mean
    X[..., -1] = -13 * X_norm[..., -1]
    return X

def compute_distance(phi1, phi2, norm = 'l2'):
    """
    Compute the Euclidean distance between two tensors phi1 and phi2.
    
    Args:
        phi1 (torch.Tensor): The first tensor.
        phi2 (torch.Tensor): The second tensor.
        norm (str): The type of norm to use ('l2' or 'l1').
    Returns:
        torch.Tensor: The computed distance.
    """
    diff = phi1 - phi2
    if norm == 'l2':
        return torch.norm(diff, p=2, dim=-1)
    elif norm == 'l1':
        return torch.norm(diff, p=1, dim=-1)
    else:
        raise ValueError(f"Unknown norm: {norm}")

def model_predict_batches(model, phis, muons, batch_size:int=2**20, device='cuda'):
    """Generate model predictions in muon batches to avoid OOM."""
    model.eval()
    outputs = []

    phis_batch = torch.from_numpy(phis).float().to(device)
    n_muons = muons.shape[1]

    with torch.no_grad():
        for i in range(0, n_muons, batch_size):
            end_idx = min(i + batch_size, n_muons)
            muons_batch = torch.from_numpy(muons[:, i:end_idx]).float().to(device)
            output = model(phis_batch, muons_batch)
            outputs.append(output.cpu())

            del muons_batch

    return torch.cat(outputs, dim=1)

from scipy.stats import qmc
def sample_phi(lb, ub, n_samples=100, dim = 43, sampling='sobol'):
        if sampling == 'normal':
            samples_phi_n = torch.randn(n_samples, dim)
        elif sampling == 'uniform':
            samples_phi_n = torch.rand(n_samples, dim) * 2 - 1
        elif sampling == 'LHS':
            samples_phi_n = lhs(dim, samples=n_samples) * 2 - 1
        elif sampling == 'sobol':
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples_phi_n = sampler.random(n_samples) * 2 - 1
            samples_phi_n = torch.from_numpy(samples_phi_n)
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")
        if samples_phi_n.dim() == 2:
            lb = lb.unsqueeze(0)
            ub = ub.unsqueeze(0)
        samples_phi = denormalize_phi(samples_phi_n, lb, ub)
        return samples_phi