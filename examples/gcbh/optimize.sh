#!/bin/bash

export PYTHONPATH=/path/to/scalar_codes
export VASP_PP_PATH=/path/to/vasp/potentials
export VASP_COMMAND="mpirun -np ${SLURM_NTASKS} /path/to/vasp/executable"

python geo_opt.py
