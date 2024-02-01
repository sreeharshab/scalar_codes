from ase.io import read, write
from pipelines import create_sigma3_gb

if __name__=="__main__":
    top_layers = read("top_grain.vasp")
    bottom_layers = read("bottom_grain.vasp")
    atoms = create_sigma3_gb(5, top_layers, bottom_layers)
    write("POSCAR", atoms)