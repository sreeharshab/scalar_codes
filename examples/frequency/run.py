from ase.io import read
from pipelines import frequency

if __name__=="__main__":
    atoms = read("POSCAR")
    freq = frequency(atoms)
    freq.run(mode="ase", scheme="parallel")