# Source before running study code (bash or zsh): single-threaded kernels as in
# the campaign, and this folder on the import path for the `study` package.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
_study_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")" && pwd)"
export PYTHONPATH="${_study_dir}${PYTHONPATH:+:$PYTHONPATH}"
unset _study_dir
