from pipelines import *
import ase
from ase.io import read, write

if __name__ == "__main__":
    atoms = read("POSCAR")
    atoms_with_charge = bader(atoms)
    write("with_charges.traj", atoms_with_charge)