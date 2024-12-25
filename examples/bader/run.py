from pipelines import bader
from ase.io import read, write

if __name__ == "__main__":
    atoms = read("POSCAR")
    addnl_settings = {"ncore": 8, "encut": 600, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1,-1], "ldauu": [3.5,0,0], "lmaxmix": 4, "prec": "High"}
    atoms_with_charge = bader(atoms, kpts=[3,7,3], addnl_settings=addnl_settings, restart=True)
    write("with_charges.traj", atoms_with_charge)