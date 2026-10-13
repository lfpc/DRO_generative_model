"""Train the LCSO surrogate of the muon shield on a fixed set of simulated designs (GPU).

Simulates n_phi_train designs around the nominal one (plus test designs at temperature 1),
trains, saves outputs/lcso_model.pt (or lcso_model_taylor.pt), and plots predicted vs. true hits.

    python lcso.py --muons temp --n_phi_train 256 --n_phi_test 30 --pz_min 2.8
    python lcso.py --model_type taylor --load_data     # reuse outputs/{training,test}_data.h5
"""
import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import build_surrogate, save_surrogate
from problems import ShipMuonShieldCuda
from trainer import train_hits_classifier

class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgFormatter)
parser.add_argument('--model_type', default='deeponet', choices=['deeponet', 'taylor'])
parser.add_argument('--rank', type=int, default=8, help='taylor: curvature directions')
parser.add_argument('--muons', default='temp', choices=['data', 'uniform', 'temp', 'mixed'],
                    help='training muons (temp = flow at temperature 1.5)')
parser.add_argument('--n_phi_train', type=int, default=256)
parser.add_argument('--n_phi_test', type=int, default=30)
parser.add_argument('--n_muons_train', type=int, default=int(5e6))
parser.add_argument('--n_muons_test', type=int, default=int(5e7))
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--p', type=int, default=64, help='latent dim in deeponet')
# below p_z ~ 2.8 GeV (the edge of the real sample) the CUDA kernel can crash
parser.add_argument('--pz_min', type=float, default=1.0, help='flow muons: min p_z [GeV]')
parser.add_argument('--pt_max', type=float, default=14.0, help='flow muons: max p_t [GeV]')
parser.add_argument('--load_data', action='store_true', help='reuse the stored designs')
parser.add_argument('--out_dir', default='outputs')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

DEVICE = torch.device(args.device)
FIGS = os.path.join(args.out_dir, 'figs')
os.makedirs(FIGS, exist_ok=True)
TRAIN_H5 = os.path.join(args.out_dir, 'training_data.h5')
TEST_H5 = os.path.join(args.out_dir, 'test_data.h5')
MODEL = os.path.join(args.out_dir,
                     'lcso_model.pt' if args.model_type == 'deeponet' else 'lcso_model_taylor.pt')

ShipMuonShieldCuda.PZ_MIN = args.pz_min
ShipMuonShieldCuda.PT_MAX = args.pt_max

# uniform fields, as in every other script (the class default is the field map)
shield = ShipMuonShieldCuda(n_samples=args.n_muons_train, uniform_fields=True)
phi_0 = shield.initial_phi
model = build_surrogate(args.model_type, phi_dim=phi_0.shape[-1], x_dim=7, phi_0=phi_0,
                        hidden=args.hidden_dim, p=args.p, rank=args.rank).to(DEVICE)
print(f'{type(model).__name__}: {sum(q.numel() for q in model.parameters()):,} params')


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


def simulate_designs(path, n_phi, n_muons, temperature=None, tag=''):
    """Simulate the nominal design plus n_phi box samples into an HDF5 store
    (phis PHYSICAL, muons RAW: what the model takes)."""
    phis = model.denormalize_phi(
        torch.cat([torch.zeros(1, model.dim), model.sample_phi(n_phi)]).to(DEVICE)).cpu()
    with h5py.File(path, 'w') as f:
        f.attrs['uniform_fields'] = True
        f.create_dataset('phis', data=phis.numpy().astype(np.float32))
        mu_ds = f.create_dataset('muons', (len(phis), n_muons, 7), dtype=np.float32,
                                 chunks=(1, n_muons, 7))
        hi_ds = f.create_dataset('hits', (len(phis), n_muons), dtype=np.int8,
                                 chunks=(1, n_muons))
        for i, phi in enumerate(phis):
            print(f'  simulating {tag}{i + 1}/{len(phis)}')
            muons = sample_muons(n_muons, temperature)
            hits = shield(phi, muons)
            mu_ds[i] = muons.numpy().astype(np.float32, copy=False)[:, :7]
            hi_ds[i] = hits.numpy().astype(np.int8, copy=False)
    print(f'  saved {path}')


# --- training designs -------------------------------------------------------
if not (args.load_data and os.path.exists(TRAIN_H5) and os.path.getsize(TRAIN_H5)):
    simulate_designs(TRAIN_H5, args.n_phi_train, args.n_muons_train, tag='train ')
with h5py.File(TRAIN_H5, 'r') as f:
    phis_train = torch.from_numpy(f['phis'][:])          # PHYSICAL
    muons = torch.from_numpy(f['muons'][:])              # RAW
    hits = torch.from_numpy(f['hits'][:])
print(f'training set: {tuple(muons.shape)}, hit rate {hits.float().mean():.3e}')

# Muon statistics, stored in the model. float64 accumulator (a float32 sum over ~1e9 values
# stalls) on a zero-copy view (a copy of the ~36 GB array does not fit in RAM).
sub = muons.numpy()[..., :6]
model.set_muon_norm(sub.mean(axis=(0, 1), dtype=np.float64),
                    sub.std(axis=(0, 1), dtype=np.float64) + 1e-12)
del sub

# --- train ------------------------------------------------------------------
losses = train_hits_classifier(model, phis_train, muons, hits, epochs=args.epochs,
                               lr=args.lr, batch_size=2 ** 22, device=DEVICE)
save_surrogate(model, MODEL)
print(f'saved {MODEL}  (final train loss {losses[-1]:.6g})')

plt.figure()
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('BCE')
plt.yscale('log')
plt.title('Training loss')
plt.savefig(os.path.join(FIGS, 'training_loss.png'), dpi=150)
plt.close()

# --- test designs -----------------------------------------------------------
if not (args.load_data and os.path.exists(TEST_H5) and os.path.getsize(TEST_H5)):
    simulate_designs(TEST_H5, args.n_phi_test, args.n_muons_test, temperature=1.0,
                     tag='test ')


# --- evaluate ---------------------------------------------------------------
@torch.no_grad()
def evaluate(phis, get_muons, get_hits, n):
    model.eval()
    out = []
    for i in range(n):
        pred, _ = model.predict_hits(phis[i].to(DEVICE), get_muons(i).to(DEVICE),
                                     batch_size=2 ** 20)
        true = int(get_hits(i).sum())
        dist = float((phis[i] - phi_0).norm())
        out.append((true, float(pred), dist))
        print(f'  {i + 1:3d}/{n}  true {true:>9,}  pred {float(pred):>11,.1f}  '
              f'|phi-phi_0| {dist:6.2f}')
    return out


def plot(rows, name, title):
    t = np.array([r[0] for r in rows], float)
    p = np.array([r[1] for r in rows], float)
    d = np.array([r[2] for r in rows], float)
    ok = (t > 0) & (p > 0)
    if not ok.any():
        print(f'  nothing to plot for {name}')
        return
    fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    sc = ax.scatter(t[ok], p[ok], c=d[ok], cmap='viridis', edgecolors='k', linewidths=0.5)
    lim = [min(t[ok].min(), p[ok].min()), max(t[ok].max(), p[ok].max())]
    ax.plot(lim, lim, 'k--', label='identity')
    fig.colorbar(sc, ax=ax, label='distance from phi_0')
    ax.set(xscale='log', yscale='log', xlabel='True hits', ylabel='Predicted hits',
           title=title)
    ax.legend()
    fig.savefig(os.path.join(FIGS, name), dpi=150)
    plt.close(fig)
    print(f'  saved {os.path.join(FIGS, name)}')


print('\ntraining designs (in-sample):')
plot(evaluate(phis_train, lambda i: muons[i], lambda i: hits[i], len(phis_train)),
     'train_insample_true_vs_pred_hits.png', 'Predicted vs. true hits (in-sample)')

print('\ntest designs:')
with h5py.File(TEST_H5, 'r') as f:
    phis_test = torch.from_numpy(f['phis'][:])
    rows = evaluate(phis_test,
                    lambda i: torch.from_numpy(f['muons'][i]),   # one design at a time
                    lambda i: np.asarray(f['hits'][i]), len(phis_test))
plot(rows, 'true_vs_pred_hits.png', 'Predicted vs. true hits (test)')
