from pipelines import *
from ase.io import read, write

if __name__ == "__main__":
    atoms = read("input.traj")
    opt_levels = {
        1: {"kpts": [1,3,3], "amin": 0.01, "ncore": 8, "encut": 300, "ismear": 0, "sigma": 0.02},
        2: {"kpts": [1,7,5], "amin": 0.01, "ncore": 8, "encut": 300},
        3: {"kpts": [1,15,9], "amin": 0.01, "ncore": 8, "encut": 300},
    }
    opt_atoms = geo_opt(atoms, opt_levels = opt_levels)
    write("optimized.traj",opt_atoms)