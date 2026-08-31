#!/bin/bash
# Shared SLURM setup for ReasonFlow jobs.
#
# Source this at the top of any sbatch script via:
#   source "<REPO_ROOT>/slurm/_common.sh"
#
# Replace these placeholders before submitting:
#   <REPO_ROOT>     - absolute path to the repo
#   <CONDA_ROOT>    - conda install (e.g. ${HOME}/miniconda3)
#
# If your cluster requires a SLURM QoS, add `#SBATCH --qos=...` to
# each sbatch script in slurm/ before submitting.
#

set -euo pipefail

# Initialise conda for non-interactive shells (avoids "conda: command not found"
# or "Run 'conda init' before 'conda activate'" errors under sbatch/srun).
source "<CONDA_ROOT>/etc/profile.d/conda.sh"

conda activate reasonflow

# Ensure conda's libstdc++ is picked up (newer than system's; scipy/rdkit need
# CXXABI_1.3.15 which the system libstdc++.so.6 on this cluster doesn't have).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "<REPO_ROOT>"

# Useful env info for job logs
echo "== SLURM =="
echo "  Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "  Node:      ${SLURM_JOB_NODELIST:-$(hostname)}"
echo "  GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "  Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "  QoS:       ${SLURM_JOB_QOS:-N/A}"
echo "  Env:       $(which python)"
echo "  CWD:       $(pwd)"
echo "==========="
