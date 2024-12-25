from pipelines import geo_opt
from ase.io import read

if __name__=="__main__":
    atoms = read("POSCAR")

    opt_levels = {
        1: {"kpts":[1,3,2], "ncore": 8, "encut": 600, "isif": 0, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1], "ldauu": [3.5,0], "lmaxmix": 4},
        2: {"kpts":[1,5,3], "ncore": 8, "encut": 600, "isif": 0, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1], "ldauu": [3.5,0], "lmaxmix": 4},
        3: {"kpts":[3,7,3], "ncore": 8, "encut": 600, "isif": 0, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1], "ldauu": [3.5,0], "lmaxmix": 4},
    }

    geo_opt(atoms,mode="vasp",opt_levels=opt_levels,restart=True)