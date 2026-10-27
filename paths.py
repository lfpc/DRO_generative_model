"""Where everything is read and written. Override with environment variables:

  RMS_OUTPUTS    results, models, pools and references (default: <repo>/outputs)
  PROJECTS_DIR   the folder holding generator_muons and MuonsAndMatter (default: ~/projects)
  MUON_FLOW      the generative flow of the muons (default: generator_muons/outputs/muon_flow.pt)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('RMS_OUTPUTS', os.path.join(ROOT, 'outputs'))
PROJECTS_DIR = os.path.expanduser(os.environ.get('PROJECTS_DIR', '~/projects'))
GENERATOR_DIR = os.path.join(PROJECTS_DIR, 'generator_muons')
MUON_FLOW = os.environ.get('MUON_FLOW', os.path.join(GENERATOR_DIR, 'outputs', 'muon_flow.pt'))

MODEL = os.path.join(OUTPUTS, 'lcso.pt')                  # the surrogate the analysis uses
RUN = os.path.join(OUTPUTS, 'robustness')                 # the analysis' results and figures
POOLS = os.path.join(OUTPUTS, 'pools')                    # training pools: <name>/shard_<worker>.pt
REFERENCES = os.path.join(OUTPUTS, 'references')          # validation designs on stored muons: <name>/
VALIDATION = os.path.join(OUTPUTS, 'validation')          # the surrogate scored on the references
EXPERIMENTS = os.path.join(OUTPUTS, 'experiments')        # experiments/: results and figures


def output(path):
    """A path relative to OUTPUTS (absolute paths are kept)."""
    return os.path.join(OUTPUTS, path)
