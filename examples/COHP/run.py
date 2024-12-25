from pipelines import COHP
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

if __name__=="__main__":
    atoms = read("POSCAR")
    bonds = [[24,2]]
    lobsterin_template = [
        "COHPstartEnergy  -22\n",
        "COHPendEnergy     18\n",
        "basisSet          pbeVaspFit2015\n",
        "includeOrbitals   spd\n",
    ]
    addnl_settings = {"encut": 400, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1,-1], "ldauu": [3.5,0,0], "lmaxmix": 4}
    cohp = COHP(atoms, bonds=bonds, lobsterin_template=lobsterin_template)
    cohp.run_vasp(kpts=[3,7,3], addnl_settings=addnl_settings)
    cohp.write_lobsterin()
    cohp.run_lobster()
    cohp.plot([-0.7, 0.5],[-8,18],[-0.01,1.5],[-8,18])