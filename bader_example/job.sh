#!/bin/bash

export PYTHONPATH=/path/to/scalar_codes
export VTST_SCRIPTS=/path/to/vtstscripts
export VTST_BADER=/path/to/bader/executable
export VASP_PP_PATH=/path/to/vasp/potentials
export OMP_NUM_THREAD=1
export VASP_COMMAND="mpirun -np ${SLURM_NTASKS} /path/to/vasp/executable"

python run.py
