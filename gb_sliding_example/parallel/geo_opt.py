from ase.io import read, write
from pipelines import *

if __name__=="__main__":
    atoms = read("POSCAR")
    opt_levels = {
        1: {"kpts": [1,3,3], "ismear": 0, "sigma": 0.02, "amin": 0.01},
        2: {"kpts": [1,7,5], "amin": 0.01},
        3: {"kpts": [1,15,9], "amin": 0.01},
    }
    geo_opt(atoms, mode="vasp", opt_levels=opt_levels)