from ase.io import read
from pipelines import cell_opt

if __name__=="__main__":
    atoms = read("POSCAR")
    addnl_settings = {"lreal": False, "ncore": 4}
    cell_opt(atoms, kpts=[5,5,5], npoints=6, eps=0.05, addnl_settings=addnl_settings)