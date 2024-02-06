from pipelines import get_base_calc, set_vasp_key, get_selective_dynamics
import os
import shutil
import subprocess
import numpy as np
from numpy import pi
from ase import Atom
from ase.io import read, write, Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import FixAtoms

def pbc_correction(atoms):
    cell = atoms.get_cell()
    for atom in atoms:
        if atom.y < 0:
            atom.y = atom.y + cell[1][1]
        if atom.y > cell[1][1]:
            atom.y = atom.y - cell[1][1]
        if atom.z < 0:
            atom.z = atom.z + cell[2][2]
        if atom.z > cell[2][2]:
            atom.z = atom.z - cell[2][2]
    return atoms

def create_sigma3_gb(n, top_layers, bottom_layers):
    top_layers = top_layers*(n,1,1) # Repeating the layers.
    for i in range(0,n*16,16):
        for j in range(i,i+16,1):
            zdisp = -2.2325*(i/16)  # Moving the layer to the left by 2.2325 Angstroms.
            top_layer = top_layers[j]
            top_layer.z = top_layer.z + zdisp
    top_layers = pbc_correction(top_layers)
    # Shifting the top layers
    for i in range(n*16):
        top_layers[i].x = top_layers[i].x + (n-1)*3.157 # Moving all the top layers by (n-1)*3.157 Angstroms.
    # Getting the required number of bottom layers.
    bottom_layers = bottom_layers*(n,1,1)
    for i in range(0,n*16,16):
        for j in range(i,i+16,1):
            zdisp = -2.2325*(((n-1)*16 - i)/16)
            bottom_layer = bottom_layers[j]
            bottom_layer.z = bottom_layer.z + zdisp
    bottom_layers = pbc_correction(bottom_layers)
    cell = bottom_layers.cell   # Getting the cell size of the bottom grain.
    # Generating the grain boundary.
    grain_boundary = top_layers + bottom_layers
    grain_boundary.set_cell([2*cell[0], cell[1], cell[2]]) # Setting the grain boundary cell size to double the bottom grain cell size.
    grain_boundary = pbc_correction(grain_boundary)
    grain_boundary.center()
    return grain_boundary

class slide_sigma3_gb:
    def __init__(self, n_steps):
        self.n_steps = n_steps
 
    def neighbor_distance(self, atoms, index, neighbor_position):
        dist = ((atoms[index].x - neighbor_position[0])**2 + (atoms[index].y - neighbor_position[1])**2 + (atoms[index].z - neighbor_position[2])**2)**(1/2)
        return dist
  
    def structure_corrections(self, atoms, theta, disp, scheme=None, n=None):
        cell = atoms.get_cell()
        atoms = pbc_correction(atoms)
        nat_cut = natural_cutoffs(atoms, mult=0.95)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(atoms)
        c = 0
        close_atoms = 0
        for atom in atoms:
            d = np.array([])
            indices, offsets = nl.get_neighbors(c)
            for i, offset in zip(indices, offsets):
                pos = atoms.positions[i] + offset @ cell    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and 
                                                            # offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
                d = np.append(d, self.neighbor_distance(atoms, c, pos))
            c = c + 1
            for i in d:
                if i < 1.4:
                    close_atoms = close_atoms + 1
        if close_atoms == 0:           
            return atoms
        else:   # Recursive call
            self.slide(atoms, theta, disp, scheme=scheme, n=n)
    
    # Testing done, working when n is odd for linear scheme and working for all n for step scheme!
    def slide(self, atoms, theta, disp, scheme=None, n=None): # disp is displacement of the second fixed layer per step. n is the number of layers per grain.
        if scheme=="linear" or None:
            layer_size = (atoms.get_cell()[0][0])/(2*n)
            n_free = n-1
            first_fixed_layer = int(n_free/2)    # Index of first fixed layer. Counting of layers start from 0.
            second_fixed_layer = first_fixed_layer + n  # Index of second fixed layer.
            disp_per_layer = disp/n
            count1 = 0
            for layer in range(first_fixed_layer, -1, -1):
                for atom in atoms:
                    if atom.x > layer*layer_size and atom.x < (layer+1)*layer_size:
                            atom.y = atom.y + disp_per_layer*count1*np.cos(theta)
                            atom.z = atom.z + disp_per_layer*count1*np.sin(theta)
                count1 += 1
            count2 = 0
            for layer in range(first_fixed_layer, second_fixed_layer+1,1):
                for atom in atoms:
                    if atom.x > layer*layer_size and atom.x < (layer+1)*layer_size:
                            atom.y = atom.y + disp_per_layer*count2*np.cos(theta)
                            atom.z = atom.z + disp_per_layer*count2*np.sin(theta)
                count2 += 1
            count3 = 5
            for layer in range(second_fixed_layer+first_fixed_layer, second_fixed_layer, -1):
                for atom in atoms:
                    if atom.x > layer*layer_size and atom.x < (layer+1)*layer_size:
                            atom.y = atom.y + disp_per_layer*count3*np.cos(theta)
                            atom.z = atom.z + disp_per_layer*count3*np.sin(theta)
                count3 += 1
            atoms = self.structure_corrections(atoms, theta, disp, scheme=scheme, n=n)
            return atoms
        if scheme=="step":
            cell = atoms.get_cell()
            for atom in atoms:
                if atom.x > cell[0][0]/2:
                    atom.y = atom.y + disp*np.cos(theta)
                    atom.z = atom.z + disp*np.sin(theta)
            atoms = self.structure_corrections(atoms, theta, disp, scheme=scheme, n=n)
            return atoms
    
    def level_opt(self, atoms, step, opt_levels, level):
        calc = get_base_calc()
        set_vasp_key(calc, 'encut', 300)
        set_vasp_key(calc, 'ibrion', 2)
        set_vasp_key(calc, 'ediffg', -0.01)
        set_vasp_key(calc, 'nsw', 200)
        set_vasp_key(calc, 'potim', 0.5)
        level_settings = opt_levels[level]
        for key in level_settings.keys():
            set_vasp_key(calc, key, level_settings[key])
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms = read("OUTCAR", index=-1)
        shutil.copyfile("OUTCAR", f"{step}/level{level}_step{step}.OUTCAR")
        shutil.copyfile("CONTCAR", f"{step}/level{level}_step{step}.vasp")
        shutil.copyfile("vasp.out", f"{step}/level{level}_step{step}.out")
        return atoms
    
    def run_serial(self, atoms, opt_levels, disp, theta, scheme=None, n=None, restart=None):
        cwd = os.getcwd()
        try:
            os.mkdir(f"{int((theta/pi)*180 + 0.1)}")
        except FileExistsError:
            pass
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        if restart==None or restart==False:
            for j in range(self.n_steps):
                try:
                    os.mkdir(f"{j+1}")
                except FileExistsError:
                    pass
                atoms = self.slide(atoms, theta, disp, scheme=scheme, n=n)
                # Editing the calc object and starting the VASP simulation.
                levels = opt_levels.keys()
                for level in levels:
                    atoms = self.level_opt(atoms, j+1, opt_levels, level)
            os.chdir(cwd)
        elif restart==True:
            cwd = os.getcwd()
            last_step = 0
            for step in range(self.n_steps):
                if os.path.exists(cwd + f"/{step+1}") and os.listdir(cwd + f"/{step+1}")!=[]:
                    last_step = step+1
                else:
                    break
            levels = opt_levels.keys()
            for level in levels:
                if os.path.exists(cwd + f"/{last_step}/level{level}_step{last_step}.vasp"):
                    last_level = level
                else:
                    break
            try:
                tmp_atoms = read(cwd + f"/{last_step}/level{last_level}_step{last_step}.vasp")
            except UnboundLocalError:
                print("NO RESTART FILES FOUND! STARTING A FRESH CALCULATION...")
                last_step = 1
                try:
                    os.mkdir(f"{last_step}")
                except FileExistsError:
                    pass
                tmp_atoms = read("../POSCAR")
                tmp_atoms = self.slide(tmp_atoms, theta, disp, scheme=scheme, n=n)
                last_level = 0
            largest_level=max(levels)
            if last_level!=largest_level:
                for level in levels:
                    if level > last_level:
                        tmp_atoms = self.level_opt(tmp_atoms, last_step, opt_levels, level)
                last_level=largest_level
            if last_level==largest_level:
                tmp_atoms = self.slide(tmp_atoms, theta, disp, scheme=scheme, n=n)
                last_step = last_step+1
            for j in range(last_step,self.n_steps+1,1):
                try:
                    os.mkdir(f'{j}')
                except FileExistsError:
                    pass
                levels = opt_levels.keys()
                for level in levels:
                    tmp_atoms = self.level_opt(tmp_atoms, j, opt_levels, level)
                tmp_atoms = self.slide(tmp_atoms, theta, disp, scheme=scheme, n=n)
            os.chdir(cwd)

    # Restart option not coded!
    def run_parallel(self, atoms, disp, theta, scheme=None, n=None, restart=None, largest_level=None):
        cwd = os.getcwd()
        try:
            os.mkdir(f"{int((theta/pi)*180 + 0.1)}")
        except FileExistsError:
            pass
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        traj = Trajectory(f"Trajectory_{int((theta/pi)*180 + 0.1)}_in.traj","w")
        if restart==None or restart==False:
            for j in range(self.n_steps):
                atoms = self.slide(atoms, n, theta, disp, scheme=scheme)
                try:
                    os.mkdir(f"{j+1}")
                except FileExistsError:
                    pass
                shutil.copy(cwd + "/job.sh", f"./{j+1}")
                shutil.copy(cwd + "/geo_opt.py", f"./{j+1}")
                os.chdir(f"./{j+1}")
                write("POSCAR", atoms)
                traj.write(atoms)
                subprocess.run(["sbatch", "job.sh"])
                os.chdir("../")
        if restart==True:
            for j in range(self.n_steps):
                if os.path.exists(f"{j+1}/opt{largest_level}.OUTCAR"):
                    pass
                else:
                    os.chdir(f"./{j+1}")
                    if os.path.exists("CONTCAR"):
                        os.rename("CONTCAR", "POSCAR")
                    subprocess.run(["sbatch", "job.sh"])
                    os.chdir("../")
        os.chdir(cwd)
    
    def get_output_Trajectory(self, atoms, theta, calc_type=None):
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        # Automation code to find max level in the simulation
        level = 1
        os.chdir("./1")
        while level!=0:
            if os.path.exists(f"level{level}_step1.OUTCAR"):
                level+=1
            else:
                break
        max_level=level-1
        os.chdir("../")
        traj = Trajectory(f"Trajectory_{int((theta/pi)*180 + 0.1)}_out.traj","w")
        traj.write(atoms)
        for i in range(self.n_steps):
            os.chdir(f"./{i+1}")
            if calc_type=='parallel':
                tmp_atoms = read("OUTCAR", index=-1)
            if calc_type=='serial':
                tmp_atoms = read(f'level{max_level}_step{i+1}.OUTCAR')
            traj.write(tmp_atoms)
            os.chdir("../")
        os.chdir(cwd)

    # Testing done for calc_type="Energy" and "Force", working!
    def analysis(self, theta, property=None):
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        traj = f"Trajectory_{int((theta/pi)*180 + 0.1)}_out.traj"
        
        # Automation code to find max level in the simulation
        level = 1
        os.chdir("./1")
        while level!=0:
            if os.path.exists(f"level{level}_step1.OUTCAR"):
                level+=1
            else:
                break
        max_level=level-1
        os.chdir("../")
        
        if property == "Energy":
            E = np.array([])
            for i in range(self.n_steps):
                traj_atoms = read(traj+f"@{i+1}")
                E = np.append(E, traj_atoms.get_potential_energy())
            os.chdir(cwd)
            return E
        elif property == "Stress":
            S = np.array([])
            for i in range(self.n_steps):
                traj_atoms = read(traj+f"@{i+1}")
                S = np.append(S, traj_atoms.get_stress()[2])
            os.chdir(cwd)
            return S
        elif property == "Force":
            F = np.array([])
            for i in range(self.n_steps):
                force_list = np.array([])
                traj_atoms = read(traj+f"@{i+1}")
                for atom in traj_atoms:
                    if get_selective_dynamics(f"./{i+1}/level{max_level}_step{i+1}.vasp", atom.index)==False:
                        force_xyz = traj_atoms.get_forces()[atom.index]
                        force_xyz = np.array(force_xyz)
                        force = np.linalg.norm(force_xyz)
                        force_list = np.append(force_list, force)
                F = np.append(F, np.mean(force_list))
            os.chdir(cwd)
            return F
     
    # Layer counting starts from 0. The grain boundary is 10th and 11th layers.
    def get_layer_movement(self, n, theta, nth_layer=None, index=None):
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        traj = f"Trajectory_{int((theta/pi)*180 + 0.1)}_out.traj"
        atoms = read(traj+"@0")
        cell = atoms.get_cell()
        layer_size = (cell[0][0])/(2*n)
        coord = []
        for i in range(self.n_steps+1):
            traj_atoms = read(traj+f"@{i}")
            for atom in traj_atoms:
                if nth_layer!=None:
                    if atom.x > layer_size*nth_layer and atom.x < layer_size*(nth_layer+1):
                        coord.append([atom.y, atom.z])
                        break
                if index!=None:
                    if atom.index==index:
                        coord.append([atom.y, atom.z])
        coord = np.array(coord)
        os.chdir(cwd)
        return coord

class intercalate_Li:
    def __init__(self):
        pass
    
    def neighbor_distance(self, atoms, index, neighbor_position):
            dist = ((atoms[index].x - neighbor_position[0])**2 + (atoms[index].y - neighbor_position[1])**2 + (atoms[index].z - neighbor_position[2])**2)**(1/2)
            return dist

    def structure_check(self, atoms):
            cell = atoms.get_cell()
            atoms = pbc_correction(atoms)
            nat_cut = natural_cutoffs(atoms, mult=0.95)
            nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
            nl.update(atoms)
            c = 0
            for atom in atoms:
                    d = np.array([])
                    indices, offsets = nl.get_neighbors(c)
                    for i, offset in zip(indices, offsets):
                        pos = atoms.positions[i] + offset @ cell    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
                        d = np.append(d, self.neighbor_distance(atoms, c, pos))
                    c = c + 1
            close_atoms = 0
            for i in d:
                if i < 1.4:
                    close_atoms = close_atoms + 1
            if close_atoms == 0:           
                return atoms
            else:
                class StructureError(Exception):
                    pass
                raise StructureError("Your structure contains too close atoms!")

    def intercalate_GB_with_Li(self, atoms, n):
        cell = atoms.get_cell()
        layer_size = (cell[0][0])/(2*n)
        cell_1by1 = [cell[0][0], 3.86638530, 6.6967757]
        z_offset_Li = 1.14
        # Obtaining x coordinates
        x = np.array([])
        for layer in range(2*n):
            x = np.append(x, layer*layer_size)
        y_size = int(cell[1][1]/cell_1by1[1])*2
        z_size = int(cell[2][2]/cell_1by1[2])*2
        xyz = np.zeros((2*n*y_size*z_size,3))
        values = np.tile(x, int(xyz.shape[0]/x.size))[:xyz.shape[0]] # See more information on np.tile in numpy documentation.
        count = 0
        for i in values:
            values[count] = i+layer_size
            count+=1
        xyz[:, 0] = values
        # Obtaining z coordinates
        for i in range(x.size):
            count = 0
            for atom in atoms:
                if i < x.size-1:
                    if (atom.symbol=="Si" or atom.symbol=="Al") and atom.x > x[i] and atom.x < x[i+1] and atom.x > x[i] + 0.49*(layer_size) and atom.y < cell_1by1[1]:
                        xyz[i+2*n*count, 2] = atom.z + z_offset_Li
                        count+=1
                elif i == x.size-1:
                    if (atom.symbol=="Si" or atom.symbol=="Al") and atom.x > x[i] and atom.x < cell[0][0] and atom.x > x[i] + 0.49*(layer_size) and atom.y < cell_1by1[1]:
                        xyz[i+2*n*count, 2] = atom.z + z_offset_Li
                        count+=1
        tmp_n = int(xyz.shape[0]/y_size)
        values = xyz[:tmp_n, 2]
        for i in range(y_size-1):
            xyz[(i+1)*tmp_n:(i+2)*tmp_n,2] = values
        # Obtaining y coordinates
        tmp_y = np.array([])
        for atom in atoms:
            tmp_y = np.append(tmp_y, atom.y)
        y_max = np.max(tmp_y)
        for i in range(y_size):
            y = y_max - (i*cell[1][1])/y_size
            xyz[(i)*tmp_n:(i+1)*tmp_n,1] = y
        for i in range(xyz.shape[0]):
            Li_atom = Atom("Li", xyz[i])
            atoms.append(Li_atom)
        # Performing checks on the generated structure
        atoms = self.structure_check(atoms)
        return atoms

def symmetrize_Si100_surface(atoms, n_fixed_layers=4, n_delete_layers=None, vacuum=15):
    cell = atoms.get_cell()
    b = cell[1,1]
    c_per_layer = 1.1878125+0.1875  # Per layer of the Si surface
    atoms.center(axis=2)
    base_z = np.array([atom.z for atom in atoms]).min()
    if n_delete_layers!=None:
        n_delete_layers = n_delete_layers/2
        delete_length = n_delete_layers*c_per_layer + base_z - 0.1    # 0.1 is the tolerance for delete_length
        delete_indices = np.array([])
        for atom in atoms:
            if atom.z>=base_z and atom.z<delete_length:
                delete_indices = np.append(delete_indices, atom.index)
        delete_indices = np.array([int(i) for i in delete_indices])
        del atoms[delete_indices]
    base_z = np.array([atom.z for atom in atoms]).min()
    base_layer_index = [atom.index for atom in atoms if abs(base_z - atom.z) < 0.1]

    # Symmetrizing the surface around the base layer center
    base_layer_center = np.array([atom.position for atom in atoms[base_layer_index]]).mean(axis=0)
    inverted_atoms = atoms.copy()

    # Symmetrizing atoms around the base layer center and manipulating to match middle layers
    for atom in inverted_atoms:
        atom.position = 2 * base_layer_center - atom.position
        atom.z = atom.z - c_per_layer
        atom.y = atom.y - b/4
    atoms += inverted_atoms

    # Applying contraints to atoms in middle layer to preserve bulk
    del atoms.constraints
    atoms.center()
    n_fixed_layers = n_fixed_layers/2
    constraint_indices = [atom.index for atom in atoms if atom.z < cell[2,2]/2 + n_fixed_layers*c_per_layer and atom.z > cell[2,2]/2 - n_fixed_layers*c_per_layer]
    constraints = FixAtoms(indices=constraint_indices)
    atoms.set_constraint(constraints)

    # Final corrections
    pos = atoms.get_positions()
    z_max = max(pos[:,2])
    z_min = min(pos[:,2])
    init_vacuum = z_min + (cell[2,2] - z_max)
    vacuum_change = vacuum*2 - init_vacuum
    cell[2,2] = cell[2,2] + vacuum_change
    atoms.set_cell(cell)
    atoms.center()

    return atoms

def cure_Si_surface_with_H(atoms, upper=None):
    nat_cut = natural_cutoffs(atoms, mult=1)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    cell_z = atoms.get_cell()[2,2]
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        if len(indices) < 4:
            if upper==True and atoms[i].z > cell_z/2:
                coord_1 = (atoms[i].x, atoms[i].y + 1.06, atoms[i].z + 1.06)
                coord_2 = (atoms[i].x, atoms[i].y - 1.06, atoms[i].z + 1.06)
                H_atom_1 = Atom("H", coord_1)
                H_atom_2 = Atom("H", coord_2)
                atoms.append(H_atom_1)
                atoms.append(H_atom_2)
            if atoms[i].z < cell_z/2:
                coord_1 = (atoms[i].x, atoms[i].y + 1.06, atoms[i].z - 1.06)
                coord_2 = (atoms[i].x, atoms[i].y - 1.06, atoms[i].z - 1.06)
                H_atom_1 = Atom("H", coord_1)
                H_atom_2 = Atom("H", coord_2)
                atoms.append(H_atom_1)
                atoms.append(H_atom_2)
    return atoms