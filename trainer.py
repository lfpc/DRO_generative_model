import torch
from tqdm import trange


def train_hits_classifier(model, phi, x, y, epochs=2000, lr=3e-3, batch_size=8192, device=None):
    """Fit model(phi, x) -> hit logits with BCE, Adam and a cosine schedule.

    phi (n_phi, d), x (n_phi, n_muons, x_dim), y (n_phi, n_muons). Each step takes the same
    batch_size // n_phi muon slots from every design. Returns the per-epoch mean loss.
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_phi, n_samples = phi.shape[0], x.shape[1]
    per_phi_batch = max(1, batch_size // n_phi)
    phi = phi.to(device)

    losses = []
    model.train()
    for _ in trange(epochs, dynamic_ncols=True):
        total, total_elements = 0.0, 0
        shuffled = torch.randperm(n_samples)
        for start in range(0, n_samples, per_phi_batch):
            idx = shuffled[start:start + per_phi_batch]
            x_batch = x[:, idx].to(device, non_blocking=True)
            y_batch = y[:, idx].float().to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(phi, x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            total += loss.item() * logits.numel()
            total_elements += logits.numel()
        scheduler.step()
        losses.append(total / total_elements)
    return losses
