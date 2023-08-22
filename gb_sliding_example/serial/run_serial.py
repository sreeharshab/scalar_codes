
from pipelines import *
import ase
import numpy as np
from ase.io import read, write
from ase.calculators.vasp import Vasp

if __name__=='__main__':
    atoms = read("POSCAR")
    opt_levels = {
        1: {"kpts": [1,3,3], "ismear": 0, "sigma": 0.02, "amin": 0.01},
        2: {"kpts": [1,7,5], "amin": 0.01},
        3: {"kpts": [1,15,9], "amin": 0.01},
    }
    n = 11
    n_steps = 20
    disp = (3.8663853050000001/5)
    theta = (0*pi)/180
    system = slide_sigma3_gb(n_steps)
    
    # Running simulation
    system.run_serial(atoms, opt_levels, disp, theta, scheme="step", restart=False)
    # # If scheme="linear"
    # system.run_serial(atoms, opt_levels, disp, theta, scheme="linear", n=n, restart=False)