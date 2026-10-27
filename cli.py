"""Command-line helpers shared by the scripts."""
import argparse

import torch

from paths import MODEL


class ArgFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """--help with the module docstring as written and every flag's default."""


def parser(doc):
    return argparse.ArgumentParser(description=doc, formatter_class=ArgFormatter)


def add_common(p, model=True):
    """--model (the surrogate), --muons (muons.make_muons) and --device."""
    if model:
        p.add_argument('--model', default=MODEL, help='surrogate checkpoint (lcso/train.py)')
    p.add_argument('--muons', default='flow', help="muon source: 'flow', 'flow:<checkpoint>' or 'file:<muons.npy>'")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p
