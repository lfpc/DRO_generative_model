"""MNIST classification as a distributionally robust learning problem.

    min_theta  sup_{Q in U(P)}  E_{(x,y) ~ Q} [ CE(f_theta(x), y) ]

The usual benchmark, set up so that every quantity the DRO layer needs is available. Two
choices deserve their reasons.

An MLP, not a convolutional network. A multilayer perceptron already reaches ~98% on MNIST,
so the convolution buys accuracy this study does not spend, and it costs the thing the study
does need: a classifier whose input is a plain vector, in the same space the generative
model lives in.

All 784 dimensions, no projection. An earlier version of this file reduced the images to
their leading principal components, which made the flow easy to fit and destroyed the
experiment: the projection deletes the directions off the data manifold, so a transport ball
confined to the retained subspace has nowhere off-manifold to go and the comparison the
benchmark exists to make becomes vacuous. The off-manifold directions have to be present for
the question to mean anything.

Keeping them forces the standard preprocessing, because raw MNIST is not a continuous law at
all: pixels take 256 discrete values and roughly four in five are exactly zero, so the
distribution on [0,1]^784 is supported on a finite set and no density model of it exists.
Dequantisation followed by a logit map is the usual remedy (Dinh et al., RealNVP; Papamakarios
et al.): add uniform noise within each quantisation bin, rescale into (0,1) away from the
boundary, and take a logit. The result is a genuine density on R^784, the map is monotone per
pixel and exactly invertible, so `to_image` always turns a score back into a picture, and the
near-degenerate background pixels survive as directions of tiny but nonzero variance -- which
is precisely where an off-manifold adversary will go.

What real data costs us: there is no true density, so KL(P || p_theta) and the plausibility
axis used elsewhere in this suite are unavailable. `held_out_log_prob` substitutes a second
flow fitted on data the adversary never sees. That is weaker -- the judge is then the same
model class as the defendant -- and it is stated wherever the number is used.
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
    """Read an IDX file (the raw MNIST format) into a tensor, fetching it if absent.

    The 53 MB of raw data is gitignored, so this is what makes the folder self-contained:
    the repository carries the code that gets MNIST rather than the bytes.
    """
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print(f'  fetching {name} ...')
        with urllib.request.urlopen(MIRROR + name + '.gz') as r:
            open(path, 'wb').write(gzip.decompress(r.read()))
    with open(path, 'rb') as f:
        _, kind, ndim = struct.unpack('>HBB', f.read(4))
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

    # -- the data ---------------------------------------------------------
    def dequantise(self, raw, generator=None):
        """Discrete pixels -> a continuous law on R^784, invertibly.

        Uniform noise inside each of the 256 quantisation bins, then a logit after shrinking
        away from the boundary by `alpha`. Without the shrink the background pixels map to
        minus infinity; with alpha = 0.05 the coordinates live in about [-2.94, 2.94] and the
        flow has a well-scaled target.

        Then Gaussian smoothing of width `smooth`, which is not cosmetic. Dequantisation
        alone leaves the 153 always-black pixels with a standard deviation of 0.02 against
        2.5 for the active ones, and a flow trained on that does not fit the digits at all:
        it drives the density to infinity on the training noise in those flat directions,
        and the held-out likelihood diverges to 10^5 while the training loss falls. Smoothing
        gives every pixel a comparable, genuinely continuous spread. The modelled law is then
        MNIST convolved with a small Gaussian -- which is what any norm-bounded threat model
        implicitly assumes anyway -- and, crucially for this benchmark, all 784 directions
        survive, including the background ones an off-manifold adversary needs.
        """
        u = torch.rand(raw.shape, generator=generator, dtype=self.dtype)
        p = self.alpha + (1 - 2 * self.alpha) * (raw.to(self.dtype) + u) / 256.0
        x = torch.log(p) - torch.log1p(-p)
        return x + self.smooth * torch.randn(x.shape, generator=generator, dtype=self.dtype)

    def to_image(self, x):
        """Back to a picture. The inverse of `dequantise` up to the noise."""
        return ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha)).clamp(0, 1)

    def sample(self, n, generator=None):
        """Draw with replacement from the training set: the empirical law is the truth here,
        there being no generative process behind MNIST to sample from."""
        i = torch.randint(self.features.shape[0], (n,), generator=generator)
        return self.features[i], self.labels[i]

    # -- the classifier ---------------------------------------------------
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
        """Per-sample cross-entropy, with a ridge penalty for the reason two_moons has one:
        without it the risk can be driven down by scaling the weights and the optimum runs
        off to infinity."""
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
        """Nominal (non-robust) classifier, by Adam on the true gradient.

        No surrogate here: the point of this benchmark is the ambiguity set, and a cheap
        exact gradient removes one confound from that comparison.
        """
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
