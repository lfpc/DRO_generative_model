import argparse
import copy
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_surrogate, save_surrogate
from problems import ShipMuonShieldCuda

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_type', default='deeponet', choices=['deeponet', 'taylor'])
parser.add_argument('--rank', type=int, default=8, help='taylor: shared curvature directions')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--p', type=int, default=8, help='latent dim in deeponet')
parser.add_argument('--muons', default='temp', choices=['data', 'uniform', 'temp', 'mixed'])
parser.add_argument('--n_phi', type=int, default=64,
                    help='Designs in the buffer (slot 0 is phi_0, pinned)')
parser.add_argument('--n_muons', type=int, default=int(5e6),
                    help='Muons per design. The buffer holds n_phi * n_muons * 28 bytes')
parser.add_argument('--refresh_every', type=int, default=5,
                    help='Epochs between refreshes; each re-simulates half the buffer '
                         '(0 disables, which reduces this to lcso.py)')
parser.add_argument('--n_phi_val', type=int, default=64, help='Held-out designs (fixed)')
parser.add_argument('--n_muons_val', type=int, default=int(10e6))
parser.add_argument('--epochs', type=int, default=200, help='Hard cap: passes over the buffer')
parser.add_argument('--patience', type=int, default=20, help='Epochs without improvement')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2 ** 20)
parser.add_argument('--load_val', action='store_true')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
VAL_H5 = os.path.join(args.out_dir, 'resample_val_data.h5')
MODEL = os.path.join(args.out_dir, f'lcso_resample_{args.model_type}.pt')

# The envelope the transport kernel is exercised on: outside it the CUDA kernel dies with
# an illegal memory access, and the tempered flow does reach there. Floors on the sampler,
# not tuning knobs.
ShipMuonShieldCuda.PZ_MIN = 1.0
ShipMuonShieldCuda.PT_MAX = 13.0

# uniform_fields=True disables the field-map simulation, as in every other script here.
shield = ShipMuonShieldCuda(n_samples=args.n_muons, uniform_fields=True)
phi_0 = shield.initial_phi
model = build_surrogate(args.model_type, phi_dim=phi_0.shape[-1], x_dim=7, phi_0=phi_0,
                        hidden=args.hidden_dim, p=args.p, rank=args.rank).to(DEVICE)
W, M, D = args.n_phi, args.n_muons, model.dim
PER_PHI = max(1, args.batch_size // W)
STEPS = max(1, M // PER_PHI)
print(f'{type(model).__name__}: {sum(q.numel() for q in model.parameters()):,} params; '
      f'{W} designs x {M:,} muons, {STEPS} steps/epoch, refresh every '
      f'{args.refresh_every} epochs')


def sample_muons(n, temperature=None):
    if temperature is not None:
        return shield.sample_gen(n_samples=n, temperature=temperature)
    if args.muons == 'uniform':
        return shield.sample_uniform(n)
    if args.muons == 'temp':
        return shield.sample_gen(n_samples=n)              # temperature 1.5
    if args.muons == 'mixed':
        n_u = int(round(0.4 * n))
        return torch.cat([shield.sample_uniform(n_u), shield.sample_gen(n_samples=n - n_u)])
    shield.n_samples = n
    return shield.sample_x()


def simulate(phi, n, temperature=None):
    """Fresh muons for this design, transported. Returns (raw muons[:, :7], hits)."""
    mu = sample_muons(n, temperature)
    hits = shield(phi, mu)
    return mu[:, :7].contiguous().float(), hits.to(torch.int8)


def sample_designs(k):
    """PHYSICAL designs: box samples pulled towards phi_0 by a random factor, so their
    radii spread from 0 outwards instead of piling up in a shell at ||u|| ~ 3.8."""
    u = model.sample_phi(k) * torch.rand(k, 1)
    return model.denormalize_phi(u.to(DEVICE)).cpu()


# --- validation set: simulated once, never refreshed ------------------------
if args.load_val and os.path.exists(VAL_H5) and os.path.getsize(VAL_H5):
    with h5py.File(VAL_H5, 'r') as f:
        val_phis, val_muons = torch.from_numpy(f['phis'][:]), torch.from_numpy(f['muons'][:])
        val_hits = torch.from_numpy(f['hits'][:])
else:
    val_phis = sample_designs(args.n_phi_val)
    val_muons = torch.empty(args.n_phi_val, args.n_muons_val, 7, dtype=torch.float32)
    val_hits = torch.empty(args.n_phi_val, args.n_muons_val, dtype=torch.int8)
    for i in range(args.n_phi_val):
        print(f'  simulating val {i + 1}/{args.n_phi_val}')
        # temperature 1: validation sits on the real flux, not the tempered one
        val_muons[i], val_hits[i] = simulate(val_phis[i], args.n_muons_val, temperature=1.0)
    with h5py.File(VAL_H5, 'w') as f:
        f.attrs['uniform_fields'] = True
        for k, v in (('phis', val_phis), ('muons', val_muons), ('hits', val_hits)):
            f.create_dataset(k, data=v.numpy())
val_true = val_hits.sum(1).double().numpy()
val_phis_d = val_phis.to(DEVICE)
print(f'validation: {tuple(val_muons.shape)}, {int(val_true.sum()):,} hits')

# --- the buffer -------------------------------------------------------------
phis = torch.empty(W, D, dtype=torch.float32)
muons = torch.empty(W, M, 7, dtype=torch.float32)
hits = torch.empty(W, M, dtype=torch.int8)

phis[0] = phi_0                                 # pinned
phis[1:] = sample_designs(W - 1)
for i in range(W):
    print(f'  simulating initial {i + 1}/{W}')
    muons[i], hits[i] = simulate(phis[i], M)
print(f'buffer: {tuple(muons.shape)}, hit rate {hits.float().mean():.3e}')

# Muon statistics are computed ONCE and frozen: they are baked into every weight learned,
# so recomputing them at a refresh would silently invalidate the training so far. The
# float64 accumulator matters on its own -- in float32 a sum over ~1e9 values stalls and
# the statistics come out quietly wrong.
sub = muons.numpy()[..., :6]                    # .numpy() on a CPU tensor is zero-copy
model.set_muon_norm(sub.mean(axis=(0, 1), dtype=np.float64),
                    sub.std(axis=(0, 1), dtype=np.float64) + 1e-12)
del sub


def refresh(start):
    """Redraw phi_0's muons, then re-simulate half the buffer, rotating from `start`."""
    muons[0], hits[0] = simulate(phis[0], M)
    k = max(1, (W - 1) // 2)
    new = sample_designs(k)
    for j in range(k):
        s = 1 + (start + j) % (W - 1)
        print(f'  refreshing slot {s}')
        phis[s] = new[j]
        muons[s], hits[s] = simulate(new[j], M)
    return start + k


loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)


@torch.no_grad()
def validate(batch=2 ** 19):
    """BCE and the hit-count error on the held-out designs. The count error is the one
    that decides anything -- see the docstring."""
    model.eval()
    tot, n, pred = 0.0, 0, np.zeros(len(val_phis))
    for i in range(len(val_phis)):
        for s in range(0, val_muons.shape[1], batch):
            xb = val_muons[i, s:s + batch].to(DEVICE).unsqueeze(0)
            yb = val_hits[i, s:s + batch].float().to(DEVICE).unsqueeze(0)
            logits = model(val_phis_d[i:i + 1], xb)
            tot += loss_fn(logits, yb).item() * yb.numel()
            n += yb.numel()
            pred[i] += float(torch.sigmoid(logits).double().sum())
    model.train()
    ok = val_true > 0
    err = float(np.abs(np.log(np.maximum(pred[ok], 1e-12) / val_true[ok])).mean())
    return tot / n, err, pred


# --- train ------------------------------------------------------------------
curves = {'train': [], 'val_bce': [], 'val_count': []}
refreshes, cursor = [], 0
best, best_state, bad = float('inf'), None, 0
model.train()
phis_d = phis.to(DEVICE)

for epoch in range(1, args.epochs + 1):
    run = 0.0
    for _ in range(STEPS):
        idx = torch.randint(0, M, (PER_PHI,)).sort().values      # sorted: a cheaper gather
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(phis_d, muons[:, idx].to(DEVICE)),
                       hits[:, idx].float().to(DEVICE))
        loss.backward()
        opt.step()
        run += loss.item()
    sched.step()

    v_bce, v_count, _ = validate()
    for k, v in (('train', run / STEPS), ('val_bce', v_bce), ('val_count', v_count)):
        curves[k].append(v)
    if v_count < best:
        best, bad = v_count, 0
        best_state = copy.deepcopy({k: t.cpu() for k, t in model.state_dict().items()})
    else:
        bad += 1
    print(f'epoch {epoch:4d}/{args.epochs}  bce {run / STEPS:.6g}  val_bce {v_bce:.6g}  '
          f'val_count {v_count:.4f}  best {best:.4f}  no-improve {bad}/{args.patience}')
    if bad >= args.patience:
        print(f'converged: no improvement in the validation count for {args.patience} epochs')
        break
    if args.refresh_every and epoch % args.refresh_every == 0:
        cursor = refresh(cursor)                # at an epoch boundary, never mid-epoch
        phis_d = phis.to(DEVICE)
        refreshes.append(epoch)
else:
    print(f'stopped at the --epochs cap ({args.epochs}) without converging')

if best_state is not None:
    model.load_state_dict(best_state)           # ship the best-validating weights, not the last
model.to(DEVICE)
save_surrogate(model, MODEL)
v_bce, v_count, val_pred = validate()
print(f'saved {MODEL}  (val count error {v_count:.4f}, {len(refreshes)} refreshes)')

# --- plots ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
ep = range(1, len(curves['train']) + 1)
ax[0].plot(ep, curves['train'], color='#0072B2', lw=2, label='train (buffer)')
ax[0].plot(ep, curves['val_bce'], color='#D55E00', lw=2, label='validation')
ax[0].set(xlabel='epoch', ylabel='BCE', yscale='log', title='Loss')
ax[0].legend()
ax[1].plot(ep, curves['val_count'], color='#D55E00', lw=2)
ax[1].set(xlabel='epoch', ylabel='$|\\log(\\mathrm{pred}/\\mathrm{true})|$', yscale='log',
          title='Validation hit count (what training stops on)')
for a in ax:
    for e in refreshes:
        a.axvline(e, color='0.85', lw=0.5, zorder=0)
fig.savefig(os.path.join(FIGS, 'resample_loss.png'), dpi=150)
plt.close(fig)

ok = (val_true > 0) & (val_pred > 0)
if ok.any():
    d = np.array([float((val_phis[i] - phi_0).norm()) for i in range(len(val_phis))])
    fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    sc = ax.scatter(val_true[ok], val_pred[ok], c=d[ok], cmap='viridis', edgecolors='k',
                    linewidths=0.5)
    lim = [min(val_true[ok].min(), val_pred[ok].min()),
           max(val_true[ok].max(), val_pred[ok].max())]
    ax.plot(lim, lim, 'k--', label='identity')
    fig.colorbar(sc, ax=ax, label='distance from phi_0')
    ax.set(xscale='log', yscale='log', xlabel='True hits', ylabel='Predicted hits',
           title='Predicted vs. true hits (validation)')
    ax.legend()
    fig.savefig(os.path.join(FIGS, 'resample_true_vs_pred_hits.png'), dpi=150)
    plt.close(fig)
print(f'saved {FIGS}/resample_{{loss,true_vs_pred_hits}}.png')
