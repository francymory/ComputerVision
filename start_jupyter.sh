#!/bin/bash
# Richiedere una GPU e avviare Jupyter Notebook
srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --account=cvcs2024 --time=60:00 --pty bash -c '
  source /homes/fmorandi/.bashrc &&
  jupyter notebook --ip=0.0.0.0 --no-browser --port=8888
'