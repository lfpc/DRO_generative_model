"""Where everything is read and written. Override with environment variables:

  RMS_OUTPUTS    everything produced (default: <repo>/outputs), laid out as
                   models/    the surrogates (lcso.pt, its seeds lcso_s<k>.pt, ...)
                   data/      simulations: training pools, reference designs, adaptive designs
                   results/   numbers and text reports, one folder per stage
                   figures/   every figure, one folder per stage
                   archive/   earlier analyses
  PROJECTS_DIR   the folder holding generator_muons and MuonsAndMatter (default: ~/projects)
  MUON_FLOW      the generative flow of the muons (default: generator_muons/outputs/muon_flow.pt)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.environ.get('RMS_OUTPUTS', os.path.join(ROOT, 'outputs'))
PROJECTS_DIR = os.path.expanduser(os.environ.get('PROJECTS_DIR', '~/projects'))
GENERATOR_DIR = os.path.join(PROJECTS_DIR, 'generator_muons')
MUON_FLOW = os.environ.get('MUON_FLOW', os.path.join(GENERATOR_DIR, 'outputs', 'muon_flow.pt'))

MODELS = os.path.join(OUTPUTS, 'models')
DATA = os.path.join(OUTPUTS, 'data')
RESULTS = os.path.join(OUTPUTS, 'results')
FIGURES = os.path.join(OUTPUTS, 'figures')

MODEL = os.path.join(MODELS, 'lcso.pt')                   # the surrogate the analysis uses
POOLS = os.path.join(DATA, 'pools')                       # training pools: <name>/shard_<worker>.pt
REFERENCES = os.path.join(DATA, 'references')             # validation designs on stored muons: <name>/
REPORT = os.path.join(ROOT, 'report', 'muon_shield_robustness.tex')
REPORT_FIGS = os.path.join(ROOT, 'report', 'figs')        # the report's figures, as PDF (tracked)

RUN = os.path.join(RESULTS, 'robustness')                 # the analysis' results (figures: figures(RUN))
VALIDATION = os.path.join(RESULTS, 'validation')          # the surrogate scored on the references
EXPERIMENTS = os.path.join(RESULTS, 'experiments')        # experiments/


def figures(results):
    """The figure folder of a results folder (or of a name): results/<x> -> figures/<x>."""
    return os.path.join(FIGURES, os.path.basename(os.path.normpath(results)))


def save_figure(fig, folder, name, report_name=None, dpi=150):
    """fig as <folder>/<name>.png, to look at; with report_name, also as report/figs/<report_name>.pdf
    when the report includes that file (vector; layers drawn with rasterized=True stay pixels)."""
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f'{name}.png'), dpi=dpi)
    if report_name and os.path.exists(REPORT) and f'{{{report_name}.pdf}}' in open(REPORT).read():
        fig.savefig(os.path.join(REPORT_FIGS, f'{report_name}.pdf'), dpi=2 * dpi)
        print(f'  -> {REPORT_FIGS}/{report_name}.pdf')


def checkpoint(name):
    """A checkpoint by name ('lcso' -> models/lcso.pt); paths ending in .pt are kept."""
    return name if name.endswith('.pt') else os.path.join(MODELS, f'{name}.pt')
