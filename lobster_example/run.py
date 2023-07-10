from pipelines import *
import ase
from ase.io import read, write

if __name__=="__main__":
    atoms = read("POSCAR")
    bonds = [[127, 129]]
    cohp = COHP(atoms, bonds=bonds)
    cohp.run_vasp(kpts=[1,7,5])
    cohp.write_lobsterin()
    cohp.run_lobster()
    cohp.plot([-2.6, 1],[-11,8.5],[-0.01,1.5],[-11, 8.5])