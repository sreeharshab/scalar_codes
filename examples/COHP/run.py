from pipelines import COHP
import ase
from ase.io import read

if __name__=="__main__":
    atoms = read("POSCAR")
    bonds = [[127, 129]]
    valence_electrons = {
        "Si": 4,
        "Al": 3,
    }
    cohp = COHP(atoms, bonds=bonds)
    cohp.run_vasp(kpts=[1,7,5], valence_electrons=valence_electrons)
    cohp.write_lobsterin()
    cohp.run_lobster()
    cohp.plot([-2.6, 1],[-11,8.5],[-0.01,1.5],[-11, 8.5])