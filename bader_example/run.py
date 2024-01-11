from pipelines import *
import ase
from ase.io import read, write

if __name__ == "__main__":
    atoms = read("POSCAR")
    valence_electrons = {
        "Si": 4,
        "Al": 3,
    }
    atoms_with_charge = bader(atoms, kpts=[5,5,5], valence_electrons=valence_electrons)
    write("with_charges.traj", atoms_with_charge)