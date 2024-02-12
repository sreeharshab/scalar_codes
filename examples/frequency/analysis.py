from ase.io import read, write
from pipelines import frequency
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--pressure", type=float, default=101325)
    args = parser.parse_args()
    temperature = args.temperature
    pressure = args.pressure

    atoms = read("POSCAR")
    potentialenergy = 0
    freq = frequency(atoms)
    H, S, G = freq.analysis(mode="IdealGas", potentialenergy=potentialenergy, temperature=temperature, pressure=pressure, copy_json_files=True, geometry="linear", symmetrynumber=1, spin=0)
    with open("Gibbs.txt", "w") as f:
        f.write(f"{G}")