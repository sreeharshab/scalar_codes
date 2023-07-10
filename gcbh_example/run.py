from ase.io import read, write
import numpy as np
import random
from gcbh import GrandCanonicalBasinHopping
  

def si_indices_in_gb(atoms, layer_size):
    n = 22
    count = 0
    Si_indices = np.array([])
    for atom in atoms:
        if atom.x > layer_size*(n/2-1) and atom.x < layer_size*(n/2+1) and atom.symbol == "Si":
            Si_indices = np.append(Si_indices, count)
        count+=1
    return Si_indices

# Modifier
def si_to_al(atoms_in):
    cell = atoms_in.get_cell()
    n = 22   # n is the number of layers in the GB structure.
    n_Al = 4    # n_Al is the number of Al in the GB structure.
    layer_size = cell[0][0]/n
    Al_indices = np.argwhere(np.array(atoms_in.get_chemical_symbols()) == "Al")[:,0]    # [:,0] is done because np.argwhere() returns a 2D array. For more info, see working of argwhere().
    Al_indices = list(Al_indices)
    change_nAl = random.randint(1,n_Al)    # Number of Al to be switched back to Si in atoms_in.
    change_indices = random.sample(Al_indices, change_nAl)  # Picking random indices to be switched from from Al to Si.
    for index in change_indices:
        atoms_in.symbols[index] = "Si"
    Si_indices = si_indices_in_gb(atoms_in, layer_size)
    atoms_out = atoms_in.copy()
    # Picking change_nAl number of new Al indices randomly.
    for i in range(change_nAl):
        mutate_index = int(random.choice(Si_indices))
        atoms_out.symbols[mutate_index] = "Al"
        Si_indices = si_indices_in_gb(atoms_out, layer_size)
    return atoms_out

if __name__=="__main__":
    atoms = read("POSCAR")
    filescopied = ["POSCAR", "geo_opt.py"]
    bh_run = GrandCanonicalBasinHopping(atoms=atoms, bash_script="optimize.sh", files_to_copied = filescopied, restart=False, chemical_potential="chemical_potentials.dat")
    bh_run.add_modifier(si_to_al, name = "SitoAl")
    bh_run.run(1000)