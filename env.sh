# Source before running anything:  source env.sh
# The muonshield conda env, this repository and cuda_muons_ship (MuonsAndMatter) on the path.
export PROJECTS_DIR=${PROJECTS_DIR:-$HOME/projects}
_u=$-; set +u
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate muonshield
[[ $_u == *u* ]] && set -u
MM=$PROJECTS_DIR/MuonsAndMatter
_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=$_root:$MM/muons_and_matter:$MM/cpp/build:$MM/cuda_muons${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
