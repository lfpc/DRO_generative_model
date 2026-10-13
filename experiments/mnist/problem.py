"""MNIST classification as a distributionally robust learning problem.

    min_theta  sup_{Q in U(P)}  E_{(x,y) ~ Q} [ CE(f_theta(x), y) ]

An MLP on all 784 pixels (no PCA: the off-manifold directions must exist for a transport
adversary to use). Pixels are dequantised, logit-mapped and lightly smoothed so the data
have a density on R^784. There is no true density, so plausibility is judged by a flow
fitted on the test set (see train_generator.py). Raw data are downloaded to data/ on first use.
"""
import gzip
import os
import struct
import urllib.request

import torch

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MIRROR = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
CLASSES = tuple(range(10))


def _idx(name):
    """Read an IDX file (the raw MNIST format) into a tensor, downloading it if absent."""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'  fetching {name} ...')
        with urllib.request.urlopen(MIRROR + name + '.gz') as r:
            open(path, 'wb').write(gzip.decompress(r.read()))
    with open(path, 'rb') as f:
        _, _, ndim = struct.unpack('>HBB', f.read(4))
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).reshape(shape)


class MNIST:
    def __init__(self, hidden=128, ridge=1e-5, alpha=0.05, smooth=0.25, seed=0,
                 dtype=torch.float32):
        self.hidden, self.ridge, self.alpha, self.dtype = hidden, ridge, alpha, dtype
        self.smooth = smooth
        self.input_dim = 784
        self.design_dim = 784 * hidden + hidden + hidden * 10 + 10

        raw = _idx('train-images-idx3-ubyte').reshape(-1, 784)
        self.labels = _idx('train-labels-idx1-ubyte').long()
        test_raw = _idx('t10k-images-idx3-ubyte').reshape(-1, 784)
        self.test_labels = _idx('t10k-labels-idx1-ubyte').long()
        g = torch.Generator().manual_seed(seed)
        self.features = self.dequantise(raw, g)
        self.test_features = self.dequantise(test_raw, g)

    def dequantise(self, raw, generator=None):
        """Discrete pixels -> a continuous law on R^784.

        Uniform dequantisation, logit after shrinking into [alpha, 1 - alpha], then Gaussian
        noise of width `smooth`. The smoothing matters: without it the always-black pixels
        have sd 0.02 (vs 2.5) and the flow's held-out likelihood diverges.
        """
        u = torch.rand(raw.shape, generator=generator, dtype=self.dtype)
        p = self.alpha + (1 - 2 * self.alpha) * (raw.to(self.dtype) + u) / 256.0
        x = torch.log(p) - torch.log1p(-p)
        return x + self.smooth * torch.randn(x.shape, generator=generator, dtype=self.dtype)

    def to_image(self, x):
        """Back to a picture. The inverse of `dequantise` up to the noise."""
        return ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha)).clamp(0, 1)

    def sample(self, n, generator=None):
        """Draw with replacement from the training set (the empirical law is the truth)."""
        i = torch.randint(self.features.shape[0], (n,), generator=generator)
        return self.features[i], self.labels[i]

    def unpack(self, phi):
        h, d = self.hidden, self.input_dim
        i = 0
        W1 = phi[i:i + d * h].reshape(d, h); i += d * h
        b1 = phi[i:i + h]; i += h
        W2 = phi[i:i + h * 10].reshape(h, 10); i += h * 10
        return W1, b1, W2, phi[i:i + 10]

    def logits(self, phi, x):
        W1, b1, W2, b2 = self.unpack(phi)
        return torch.relu(x @ W1 + b1) @ W2 + b2

    def loss(self, phi, x, y):
        """Per-sample cross-entropy plus a ridge penalty (keeps the optimum finite)."""
        ce = torch.nn.functional.cross_entropy(self.logits(phi, x), y.long(),
                                               reduction='none')
        return ce + self.ridge * (phi ** 2).sum()

    def risk(self, phi, x, y):
        return self.loss(phi, x, y).mean()

    def accuracy(self, phi, x, y):
        return float((self.logits(phi, x).argmax(1) == y.long()).to(self.dtype).mean())

    def init_design(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        h, d = self.hidden, self.input_dim
        return torch.cat([
            torch.randn(d * h, generator=g, dtype=self.dtype) * (2.0 / d) ** 0.5,
            torch.zeros(h, dtype=self.dtype),
            torch.randn(h * 10, generator=g, dtype=self.dtype) * (2.0 / h) ** 0.5,
            torch.zeros(10, dtype=self.dtype)])

    def fit(self, steps=4000, batch=256, lr=2e-3, seed=0):
        """Nominal (non-robust) classifier, by Adam on the true gradient."""
        g = torch.Generator().manual_seed(seed)
        phi = self.init_design(seed).requires_grad_(True)
        opt = torch.optim.Adam([phi], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            x, y = self.sample(batch, g)
            opt.zero_grad()
            self.risk(phi, x, y).backward()
            opt.step()
            sched.step()
        return phi.detach()
