from pipelines import DOS
import ase
from ase.io import read

if __name__=="__main__":
    atoms = read("POSCAR")
    dos = DOS()
    dos.run(atoms,kpts=[5,5,5])
    dos.plot()
    print(dos.get_band_centers())