from pipelines import *

if __name__=="__main__":
    atoms = read('POSCAR')

    opt_levels = {
        1: {"kpts": [1,1,1], "ismear": 0, "sigma": 0.02},
        2: {"kpts": [5,5,1]},
        3: {"kpts": [9,9,1]},
    }

    geo_opt(atoms,mode='vasp',opt_levels=opt_levels)