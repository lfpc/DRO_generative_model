import torch
from tqdm import trange


def train_hits_classifier(
    model,
    phi,
    x,
    y,
    epochs=2000,
    lr=3e-3,
    batch_size=8192,
    device=None,
    l2_reg=None,
    scheduler='cosine',
    progress_desc=None,
    progress_position=0,
    progress_leave=True,
    progress_disable=False,
    epoch_callback=None,
    pos_weight=None,
    phi_val=None,
    x_val=None,
    y_val=None,
):
    device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.5), int(epochs * 0.8)], gamma=0.2
        )
    elif scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) 
    pw = None if pos_weight is None else torch.as_tensor(
        float(pos_weight), dtype=torch.get_default_dtype(), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    n_phi = phi.shape[0]
    n_samples = x.shape[1]
    per_phi_batch = max(1, batch_size // max(n_phi, 1))
    indices = torch.arange(n_samples)
    phi_dev = phi.to(device)

    validate = phi_val is not None and x_val is not None and y_val is not None
    if validate:
        phi_val_dev = phi_val.to(device)
        n_val_samples = x_val.shape[0]

    losses = []
    val_losses = []
    model.train()
    for _epoch in trange(
        epochs,
        desc=progress_desc,
        position=progress_position,
        leave=progress_leave,
        dynamic_ncols=True,
        disable=progress_disable,
    ):
        total = 0.0
        total_elements = 0
        shuffled = indices[torch.randperm(n_samples)]
        for start in range(0, n_samples, per_phi_batch):
            end = min(start + per_phi_batch, n_samples)
            batch_idx = shuffled[start:end]
            x_batch = x[:, batch_idx].to(device, non_blocking=True)
            y_batch = y[:, batch_idx].float().to(device, non_blocking=True)

            optimizer.zero_grad()
            if True:# isinstance(model, DeepONetClassifier) or isinstance(model, StochasticTaylor) or isinstance(model,StochasticReducedTaylor):
                logits = model(phi_dev, x_batch)
            else:
                phi_rep = phi_dev[:, None, :].expand(n_phi, x_batch.shape[1], phi_dev.shape[1]).reshape(-1, phi_dev.shape[1])
                x_flat = x_batch.reshape(-1, x_batch.shape[-1])
                inp = torch.cat([phi_rep, x_flat], dim=1)
                logits = model(inp).view(n_phi, -1)

            loss = loss_fn(logits, y_batch)
            if l2_reg is not None:
                l2_penalty = sum((p ** 2).sum() for p in model.parameters())
                loss = loss + l2_reg * l2_penalty
            loss.backward()
            optimizer.step()

            elems = logits.numel()
            total += loss.item() * elems
            total_elements += elems
        scheduler.step()
        losses.append(total / max(total_elements, 1))

        if validate:
            model.eval()
            val_total = 0.0
            val_elements = 0
            with torch.no_grad():
                for start in range(0, n_val_samples, per_phi_batch):
                    end = min(start + per_phi_batch, n_val_samples)
                    x_batch = x_val[start:end].to(device, non_blocking=True).unsqueeze(0)
                    for j in range(phi_val_dev.shape[0]):
                        phi_j = phi_val_dev[j:j + 1]
                        y_batch = y_val[j:j + 1, start:end].float().to(device, non_blocking=True)
                        logits = model(phi_j, x_batch)
                        loss = loss_fn(logits, y_batch)
                        elems = logits.numel()
                        val_total += loss.item() * elems
                        val_elements += elems
            val_losses.append(val_total / max(val_elements, 1))
            model.train()

        if epoch_callback is not None:
            model.eval()
            with torch.no_grad():
                epoch_callback(model, _epoch, losses[-1])
            model.train()
    return losses, val_losses

from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
def fit_gp(model):
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model