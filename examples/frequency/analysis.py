from ase.io import read, write
from pipelines import frequency
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=300)
    args = parser.parse_args()
    temperature = args.temperature

    atoms = read("POSCAR")
    potentialenergy = 0
    freq = frequency(atoms)
    freq.check_vib_files()
    S, F, U = freq.analysis(mode="ase", thermo_style="Harmonic", potentialenergy=potentialenergy, temperature=temperature, copy_json_files=True)
    with open("Gibbs.txt", "w") as f:
        f.write(f"{F}")