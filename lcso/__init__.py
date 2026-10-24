"""LCSO: a surrogate of the per-muon hit probability p(hit | design, muon), trained once on
simulated pools of designs and then used for the robustness analysis.

  models.py           the architectures (DeepONet, Taylor, MLP, ensembles) and their derivatives
  surrogate.py        the analysis' view of a trained surrogate: tolerance box, save, load, stack
  training.py         the training loop (case-control BCE, early stopping on held-out designs)
  data.py             the training pools: their designs and their rows
  simulate_pools.py   simulate a pool of designs (simulate_pools.sh: all of them, on several GPUs)
  adaptive_designs.py designs for an adaptive pool, where the conclusions rest
  train.py            train a surrogate on pools, or stack seeds into one (python -m lcso.train)
"""
from .surrogate import LCSONet, ModelSurrogate, load_members, load_surrogate, muon_inputs, save_surrogate, stack

__all__ = ['LCSONet', 'ModelSurrogate', 'load_members', 'load_surrogate', 'muon_inputs', 'save_surrogate', 'stack']
