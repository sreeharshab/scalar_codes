from ase.io import read
from pipelines import *

if __name__=="__main__":
    atoms = read("POSCAR")
    freq = frequency(atoms)
    freq.run(mode="ase", scheme="parallel")
    freq.analysis(mode="Harmonic", potentialenergy=0, temperature=300, copy_json_files=True)