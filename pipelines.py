import os
import shutil
import subprocess
import numpy as np
from numpy import arctan, sin, cos, tan, pi
from matplotlib import pyplot as plt
import ase
from ase import Atom
from ase.io import read, write, Trajectory
from ase.calculators.vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.calculators.vasp.create_input import float_keys, exp_keys, string_keys, int_keys, bool_keys, list_int_keys, list_bool_keys, list_float_keys, special_keys, dict_keys
from ase.optimize import BFGS
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.neb import NEB
from ase.neb import NEBTools
from ase.eos import calculate_eos
from ase.eos import EquationOfState as EOS
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
import pyfiglet
import pandas as pd

# Generate the ASCII art logo
logo = pyfiglet.figlet_format("Pipelines")
print(logo)
print("Version 2\n")
print("Sree Harsha Bhimineni\nYantao Xia\nUniversity of California, Los Angeles\nbsreeharsha@ucla.edu")

"""VASP related codes using ASE"""

# Testing done, working!
def get_base_calc():
    base_calc = Vasp(
        gga="PE",
        lreal="Auto",
        lplane=True,
        lwave=False,
        lcharg=False,
        ncore=16,
        prec="Normal",
        encut=400,
        ediff=1e-6,
        algo="VeryFast",
        ismear=-5,
        gamma=True,
    )
    return base_calc

# Testing done, working!
# See ASE documentation for more information on why this is necessary.
def set_vasp_key(calc, key, value):
    if key in float_keys:
        calc.float_params[key] = value
    elif key in exp_keys:
        calc.exp_params[key] = value
    elif key in string_keys:
        calc.string_params[key] = value
    elif key in int_keys:
        calc.int_params[key] = value
    elif key in bool_keys:
        calc.bool_params[key] = value
    elif key in list_int_keys:
        calc.list_int_params[key] = value
    elif key in list_bool_keys:
        calc.list_bool_params[key] = value
    elif key in list_float_keys:
        calc.list_float_params[key] = value
    elif key in list_float_keys:
        calc.list_float_params[key] = value
    elif key in special_keys:
        calc.special_params[key] = value
    elif key in dict_keys:
        calc.dict_params[key] = value
        
    # some keys need special treatment
    # including kpts, gamma, xc
    if key in calc.input_params.keys():
        calc.input_params[key] = value

"""Yantao's code"""
def cell_opt(atoms, npoints=5, eps=0.04):
    calc = get_base_calc()

    calc.set(ibrion=-1, nsw=0, kpts=atoms.info["kpts"])
    atoms.calc = calc

    eos = calculate_eos(atoms, npoints=npoints, eps=eps, trajectory="eos.traj")

    v, e, B = eos.fit()
    eos.plot(filename="eos.png")
    opt_factor = v / atoms.get_volume()
    atoms.cell = atoms.cell * opt_factor
    write("opted_cell.vasp", atoms)
    return atoms


def axis_opt(atoms, axis, npoints=5, eps=0.04):
    """relax one vector of the cell"""
    kpts = atoms.info["kpts"]
    ens = np.zeros(npoints)
    vols = np.zeros(npoints)

    # by defualt, shrink/expand axis by 4%
    factors = np.linspace(1 - eps, 1 + eps, npoints)

    for ifactor, factor in np.ndenumerate(factors):
        atoms_tmp = atoms.copy()
        atoms_tmp.cell[axis] = atoms.cell[axis] * factor
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, kpts=kpts, directory=f"{factor:.2f}")
        atoms_tmp.calc = calc
        ens[ifactor] = atoms_tmp.get_potential_energy()
        vols[ifactor] = atoms_tmp.get_volume()

    eos = EOS(volumes=vols, energies=ens, eos="sj")
    v0, e0, B = eos.fit()
    opt_factor = v0 / atoms.get_volume()
    atoms.cell[axis] = atoms.cell[axis] * opt_factor
    write("opted_axis.vasp", atoms)
    return atoms


def geo_opt(atoms, mode="vasp", opt_levels=None, fmax=0.02):
    calc = get_base_calc()
    if not opt_levels:
        # for bulks.
        # other systems: pass in argument
        opt_levels = {
            1: {"kpts": [3, 3, 3]},
            2: {"kpts": [5, 5, 5]},
            3: {"kpts": [7, 7, 7]},
        }

    levels = opt_levels.keys()
    if mode == 'vasp':
        write("CONTCAR", atoms)
        for level in levels:
            level_settings = opt_levels[level]
            # default settings when using built-in optimizer
            set_vasp_key(calc, 'ibrion', 2)
            set_vasp_key(calc, 'ediffg', -1e-2)
            set_vasp_key(calc, 'nsw', 200)
            set_vasp_key(calc, 'nelm', 200)
            # user-supplied overrides
            for key in level_settings.keys():
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp = read("CONTCAR")
            atoms_tmp.calc = calc
            atoms_tmp.get_potential_energy()
            calc.reset()
            atoms_tmp = read("OUTCAR", index=-1)
            shutil.copyfile("CONTCAR", f"opt{level}.vasp")
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
    elif mode == 'ase':
        atoms_tmp = atoms.copy()
        from ase.optimize import BFGS
        # this atoms_tmp is updated when optimizer runs
        for level in levels:
            # default settings when using ase optimizer
            set_vasp_key(calc, 'ibrion', -1)
            set_vasp_key(calc, 'nsw', 0)
            # user-supplied overrides
            level_settings = opt_levels[level]
            for key in level_settings.keys():
                if key in ['nsw', 'ibrion', 'ediffg']:
                    continue
                set_vasp_key(calc, key, level_settings[key])

            atoms_tmp.calc = calc
            opt = BFGS(atoms_tmp,
                       trajectory = f"opt{level}.traj",
                       logfile = f"opt{level}.log")
            opt.run(fmax=fmax)
            calc.reset()
            shutil.copyfile("vasprun.xml", f"opt{level}.xml")
            shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
            
    return atoms_tmp


def freq(atoms, mode="vasp"):
    calc = get_base_calc()
    if "kpts" in atoms.info.keys():
        kpts = atoms.info["kpts"]
    else:
        kpts = [1, 7, 5]
    calc.set(kpts=kpts)

    if mode == "vasp":
        # avoid this on large structures
        # ncore/npar unusable, leads to kpoint errors
        # isym must be switched off, leading to large memory usage
        calc.set(
            ibrion=5,
            potim=0.015,
            nsw=500,  # as many dofs as needed
            ncore=None,  # avoids error of 'changing kpoints'
            npar=None,
            isym=0,
        )  # turn off symmetry

        atoms.calc = calc
        atoms.get_potential_energy()
        # todo: parse OUTCAR frequencies and modes
    elif mode == "ase":
        calc.set(lwave=True, isym=-1)  # according to michael
        atoms.calc = calc
        constr = atoms.constraints
        constr = [c for c in constr if isinstance(c, FixAtoms)]
        vib_index = [a.index for a in atoms if a.index not in constr[0].index]
        vib = Vibrations(atoms, indices=vib_index)
        vib.run()  # this will save json files
        vib.summary()


def bader(atoms):
    def run_vasp(atoms):
        calc = get_base_calc()

        if "kpts" in atoms.info.keys():
            kpts = atoms.info["kpts"]
        else:
            kpts = [1, 7, 5]

        calc.set(
            ibrion=-1, nsw=0, lorbit=12, lcharg=True, laechg=True, kpts=kpts
        )

        atoms.calc = calc
        atoms.get_potential_energy()
        assert os.path.exists("AECCAR0"), "chgsum.pl: AECCAR0 not found"
        assert os.path.exists("AECCAR2"), "chgsum.pl: AECCAR2 not found"

    def run_bader():
        # add charges
        chgsum = os.getenv("VTST_SCRIPTS") + "/chgsum.pl"
        assert os.path.exists(chgsum), "chgsum not found"
        subprocess.run([chgsum, "AECCAR0", "AECCAR2"], capture_output=True)
        assert os.path.exists("CHGCAR_sum"), "chgsum.pl: CHGCAR_sum not found"

        # run bader
        bader = os.getenv("VTST_BADER")
        assert os.path.exists(bader), "bader not found"
        subprocess.run(
            [bader, "CHGCAR", "-ref", "CHGCAR_sum"], capture_output=True
        )
        assert os.path.exists("ACF.dat"), "bader: ACF.dat not found"

    def read_bader(atoms):
        latoms = len(atoms)
        df = pd.read_table(
            "ACF.dat",
            delim_whitespace=True,
            header=0,
            skiprows=[1, latoms + 2, latoms + 3, latoms + 4, latoms + 5],
        )
        charges = df["CHARGE"].to_numpy()
        n_si = len([a for a in atoms if a.symbol == "Si"])
        n_o = len([a for a in atoms if a.symbol == "O"])
        n_al = len([a for a in atoms if a.symbol == "Al"])

        ocharges = np.array([4] * n_si + [3] * n_al + [6] * n_o)
        dcharges = -charges + ocharges
        atoms.set_initial_charges(np.round(dcharges, 2))

        return atoms

    run_vasp(atoms)

    run_bader()

    atoms_with_charge = read_bader(atoms)
    return atoms_with_charge


class COHP:
    def __init__(self, atoms, bonds, lobsterin_template=None):
        self.atoms = atoms
        self.bonds = bonds

        if lobsterin_template:
            with open(lobsterin_template) as fhandle:
                template = fhandle.readlines()
        else:
            template = [
                "COHPstartEnergy  -22\n",
                "COHPendEnergy     18\n",
                "basisSet          pbeVaspFit2015\n",
                "includeOrbitals   sp\n",
            ]

        self.lobsterin_template = template

    def run_vasp(self, kpts):
        atoms = self.atoms
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, isym=-1, prec="Accurate", kpts=kpts, lwave=True, lcharg=True)

        n_si = len([a for a in atoms if a.symbol == "Si"])
        n_o = len([a for a in atoms if a.symbol == "O"])
        n_al = len([a for a in atoms if a.symbol == "Al"])
        n_li = len([a for a in atoms if a.symbol == "Li"])

        nelect = n_si * 4 + n_o * 6 + n_al * 3 + n_li * 1
        calc.set(nbands=nelect + 20)  # giving 20 empty bands. May require more since the number of bands can be > number of basis functions, but not less!

        atoms.calc = calc
        atoms.get_potential_energy()

    def write_lobsterin(self):
        lobsterin = "lobsterin"

        with open(f"{lobsterin}", "w+") as fhandle:
            for line in self.lobsterin_template:
                fhandle.write(line)
            for b in self.bonds:
                fhandle.write(f"cohpBetween atom {b[0]+1} and atom {b[1]+1}\n")

    def run_lobster(self):
        lobster = os.getenv("LOBSTER")

        # lobster_env = os.environ.copy()
        # typically we avoid using OpenMP, this is an exception
        # lobster_env["OMP_NUM_THREADS"] = os.getenv("NSLOTS")

        # subprocess.run([lobster], capture_output=True, env=lobster_env)
        subprocess.run([lobster], capture_output=True)

    def plot(self, cohp_xlim, cohp_ylim, icohp_xlim, icohp_ylim):
        # modded from https://zhuanlan.zhihu.com/p/470592188
        # lots of magic numbers, keep until it breaks down

        def read_COHP(fn):
            raw = open(fn).readlines()
            raw = [line for line in raw if "No" not in line][3:]
            raw = [[eval(i) for i in line.split()] for line in raw]
            return np.array(raw)

        data_cohp = read_COHP("./COHPCAR.lobster")
        symbols = self.atoms.get_chemical_symbols()
        labels_cohp = [
            f"{symbols[b[0]]}[{b[0]}]-{symbols[b[1]]}[{b[1]}]"
            for b in self.bonds
        ]
        icohp_ef = [
            eval(line.split()[-1])
            for line in open("./ICOHPLIST.lobster").readlines()[1:]
        ]

        data_len = (data_cohp.shape[1] - 3) // 2
        assert (
            len(labels_cohp) == data_len
        ), "Inconsistent bonds definition and COHPCAR.lobster"
        for i in range(data_len):
            fig, ax1 = plt.subplots(figsize=[2.4, 4.8])
            ax1.plot(
                -data_cohp[:, i * 2 + 3],
                data_cohp[:, 0],
                color="k",
                label=labels_cohp[i],
            )
            ax1.fill_betweenx(
                data_cohp[:, 0],
                -data_cohp[:, i * 2 + 3],
                0,
                where=-data_cohp[:, i * 2 + 3] >= 0,
                facecolor="green",
                alpha=0.2,
            )
            ax1.fill_betweenx(
                data_cohp[:, 0],
                -data_cohp[:, i * 2 + 3],
                0,
                where=-data_cohp[:, i * 2 + 3] <= 0,
                facecolor="red",
                alpha=0.2,
            )

            ax1.set_ylim(cohp_ylim)
            ax1.set_xlim(cohp_xlim)
            ax1.set_xlabel("-COHP (eV)", color="k", fontsize="large")
            ax1.set_ylabel("$E-E_\mathrm{F}$ (eV)", fontsize="large")
            ax1.tick_params(axis="x", colors="k")
            # ICOHP
            ax2 = ax1.twiny()
            ax2.plot(-data_cohp[:, i * 2 + 4], data_cohp[:, 0], color="grey")
            ax2.set_ylim(icohp_ylim)  # [-10, 6]
            ax2.set_xlim(icohp_xlim)  # [-0.01, 1.5]
            ax2.set_xlabel("-ICOHP (eV)", color="grey", fontsize="large")
            ax2.xaxis.tick_top()
            ax2.xaxis.set_label_position("top")
            ax2.tick_params(axis="x", colors="grey")

            # legends
            ax1.axvline(0, color="k", linestyle=":", alpha=0.5)
            ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
            labelx = max(icohp_xlim) - 0.05
            labely = max(icohp_ylim) - 0.5

            ax2.annotate(
                labels_cohp[i],
                xy=(labelx, labely),
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="w", alpha=0.5),
            )
            ax2.annotate(
                f"{-icohp_ef[i]:.3f}",
                xy=(labelx, -0.05),
                ha="right",
                va="top",
                color="grey",
            )
            fig.savefig(
                f"cohp-{i+1}.png",
                dpi=500,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()


class NEB_VASP:
    def __init__(self, initial, final):
        self.initial = initial
        self.final = final
        self.images = None
        self.kpts = initial.info["kpts"]
        self.fmin = 1e-2

    def interpolate(self, method="linear", nimage=8):
        if method == "linear":
            images = [self.initial]
            images += [self.initial.copy() for i in range(nimage - 2)]
            images += [self.final]

            neb = NEB(images)
            neb.interpolate()
            self.images = images

        elif method == "optnpath":
            types = list(set(self.initial.get_chemical_symbols()))
            template = {
                "nat": len(self.initial),  # Number of atoms
                "ngeomi": 2,  # Number of initial geometries
                "ngeomf": nimage,  # Number of geometries along the path
                "OptReac": False,  # Don't optimize the reactants ?
                "OptProd": False,  # Don't optimize the products
                "PathOnly": True,  # stop after generating the first path
                "AtTypes": types,  # Type of the atoms
                "coord": "mixed",
                "maxcyc": 10,  # Launch 10 iterations
                "IReparam": 1,  # re-distribution of points along the path every 1 iteration
                "SMax": 0.1,  # Max displacement will be 0.1 a.u.
                "ISpline": 5,  # Start using spline interpolation at iteration
                "prog": "VaSP",
            }  # optnpath refuse to work w/o prog tag
            path_fname = "tmp_neb.path"
            with open(path_fname, "w") as fhandle:
                fhandle.write("&path\n")
                for k in template.keys():
                    val = template[k]
                    if isinstance(val, bool):
                        val = str(val)[0]
                    elif isinstance(val, list):
                        val = " ".join(['"' + str(line) + '"' for line in val])
                    elif isinstance(val, str):
                        val = "'" + val + "'"

                    fhandle.write(f"  {k:s}={val},\n")
                fhandle.write("/\n")

            # a hack around the bug in optnpath:
            # if selective dynamics not used optnpath
            # will repeat 'Cartesian' in the output POSCARs
            self.initial.set_constraint(FixAtoms([]))
            self.final.set_constraint(FixAtoms([]))

            # another hack around optnpath not recognizing
            # POSCAR format with atom counts in VASP5
            is_fname = "tmp_init.vasp"
            fs_fname = "tmp_final.vasp"

            write(is_fname, self.initial, vasp5=False, label="IS")
            write(fs_fname, self.final, vasp5=False, label="FS")

            os.system(f"cat {is_fname} {fs_fname} >> {path_fname}")
            os.remove(is_fname)
            os.remove(fs_fname)
            optnpath = os.getenv("OPTNPATH")

            subprocess.run([optnpath, path_fname], capture_output=True)
            os.remove(path_fname)
            os.remove("Path_cart.Ini")
            os.remove("Path.Ini")
            images = []
            for iimage in range(nimage):
                poscar_fname = f"POSCAR_{iimage:02d}"
                images.append(read(poscar_fname))
                os.remove(poscar_fname)

            self.images = images

    def write_input(self, backend):
        self.backend = backend
        if backend == "ase":
            for image in self.images[1:-2]:
                calc = get_base_calc()
                calc.set(ibrion=-1, nsw=0, kpts=self.kpts)
                image.calc = calc
            print("no input needs to be written for ase backend")

        elif backend == "vtst":
            calc = get_base_calc()

            calc.set(
                ibrion=3, images=len(self.images), lclimb=True, ncore=4, kpar=2
            )

            calc.write_input(self.initial)

            os.remove("POSCAR")
            for iimage in range(len(self.images)):
                workdir = f"{iimage:02d}"
                if not os.path.exists(workdir):
                    os.mkdir(workdir)
                write(f"{workdir}/POSCAR", self.images[iimage])

    def run(self):
        if self.backend == "ase":

            neb = NEB(self.images)
            optimizer = BFGS(neb, trajectory="I2F.traj")
            # todo: print out warnings about number of cpus and serial execution
            optimizer.run(fmax=self.fmin)

        elif self.backend == "vtst":
            command = os.getenv("VASP_COMMAND")
            # todo: check number of cpus makes sense
            subprocess.run(command, capture_output=True, shell=True)

    def monitor(self):
        # read the OUTCARs and get their energies.

        runs = []

        # inefficient: reads long long OUTCARs twice
        for iimage in range(1, len(self.images) - 1):
            with open(f"{iimage:02d}/OUTCAR", "r") as fhandle:
                lines = fhandle.readlines()
                run = 0
                for line in lines:
                    if "  without" in line:
                        run += 1
                runs.append(run)

        runs = min(runs)
        nimages = len(self.images)
        energies = np.zeros((runs, nimages))
        for iimage in range(1, len(self.images) - 1):
            run = 0
            with open(f"{iimage:02d}/OUTCAR", "r") as fhandle:
                lines = fhandle.readlines()
                for line in lines:
                    if "  without" in line:
                        energies[run][iimage] = float(line.split()[-1])
                        run += 1
                        if run >= runs:
                            break
        energies[:, 0] = self.initial.get_potential_energy()
        energies[:, -1] = self.final.get_potential_energy()

        for ien, en in enumerate(energies):
            plt.plot(en, label=str(ien))

        plt.legend(loc="best")
        plt.savefig("neb_progress.png")

class Dimer:
    def __init__(self, atoms):
        calc = get_base_calc()
        calc.set(
            ibrion=3,
            ediffg=-2e-2,
            ediff=1e-8,
            nsw=500,
            ichain=2,
            potim=0,
            iopt=2,
            kpar=4,
            kpts=atoms.info["kpts"],
        )
        atoms.calc = calc
        self.atoms = atoms

    def run(self):
        self.atoms.get_potential_energy()

"""End of Yantao's code"""

# Testing done, working!
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

# Testing done, working!
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

    # Testing done, working!
    def neighbor_distance(self, atoms, index, neighbor_position):
        dist = ((atoms[index].x - neighbor_position[0])**2 + (atoms[index].y - neighbor_position[1])**2 + (atoms[index].z - neighbor_position[2])**2)**(1/2)
        return dist

    # Testing done, working!
    def structure_corrections(self, atoms, disp):
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
            self.slide(atoms,disp)
    
    # Testing done, working when n is odd for linear scheme and working for all n for step scheme!
    def slide(self, atoms, n, theta, disp, scheme=None): # disp is displacement of the second fixed layer per step. n is the number of layers per grain.
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
            atoms = self.structure_corrections(atoms, disp)
            return atoms
        if scheme=="step":
            cell = atoms.get_cell()
            for atom in atoms:
                if atom.x > cell[0][0]/2:
                    atom.y = atom.y + disp*np.cos(theta)
                    atom.z = atom.z + disp*np.sin(theta)
            atoms = self.structure_corrections(atoms, disp)
            return atoms
    
    # Testing done, working!
    def run_serial(self, atoms, opt_levels, n, disp, theta, scheme=None, restart=None):
        calc = get_base_calc()
        set_vasp_key(calc, 'encut', 300)
        set_vasp_key(calc, 'ibrion', 2)
        set_vasp_key(calc, 'ediffg', -0.01)
        set_vasp_key(calc, 'nsw', 200)
        set_vasp_key(calc, 'potim', 0.5)
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
                atoms = self.slide(atoms, n, theta, disp, scheme=scheme)
                # Editing the calc object and starting the VASP simulation.
                levels = opt_levels.keys()
                for level in levels:
                    level_settings = opt_levels[level]
                    for key in level_settings.keys():
                        set_vasp_key(calc, key, level_settings[key])
                    atoms.calc = calc
                    atoms.get_potential_energy()
                    calc = get_base_calc()
                    set_vasp_key(calc, 'encut', 300)
                    set_vasp_key(calc, 'ibrion', 2)
                    set_vasp_key(calc, 'ediffg', -0.01)
                    set_vasp_key(calc, 'nsw', 200)
                    set_vasp_key(calc, 'potim', 0.5)
                    atoms = read("CONTCAR")
                    shutil.copyfile("OUTCAR", f"{j+1}/level{level}_step{j+1}.OUTCAR")
                    shutil.copyfile("CONTCAR", f"{j+1}/level{level}_step{j+1}.vasp")
        elif restart==True:
            cwd = os.getcwd()
            last_step = 0
            for step in range(self.n_steps):
                if os.path.exists(cwd + f"/{step+1}"):
                    last_step = step+1
                else:
                    break
            levels = opt_levels.keys()
            for level in levels:
                if os.path.exists(cwd + f"/{last_step}/level{level}_step{last_step}.vasp"):
                    last_level = level
                else:
                    break
            tmp_atoms = read(cwd + f"/{last_step}/level{last_level}_step{last_step}.vasp")
            # This is to prevent calculating all the levels for the last step if they are already computed.
            largest_level=0
            for level in levels:
                largest_level = level
            if last_level==largest_level:
                tmp_atoms = self.slide(tmp_atoms, n, theta, disp, scheme=scheme)
                last_step = last_step+1
                try:
                    os.mkdir(f'{last_step}')
                except FileExistsError:
                    pass
            for j in range(last_step,self.n_steps+1,1):
                try:
                    os.mkdir(f'{j}')
                except FileExistsError:
                    pass
                levels = opt_levels.keys()
                for level in levels:
                    level_settings = opt_levels[level]
                    for key in level_settings.keys():
                        set_vasp_key(calc, key, level_settings[key])
                    tmp_atoms.calc = calc
                    tmp_atoms.get_potential_energy()
                    calc = get_base_calc()
                    set_vasp_key(calc, 'encut', 300)
                    set_vasp_key(calc, 'ibrion', 2)
                    set_vasp_key(calc, 'ediffg', -0.01)
                    set_vasp_key(calc, 'nsw', 200)
                    set_vasp_key(calc, 'potim', 0.5)
                    tmp_atoms = read("CONTCAR")
                    shutil.copyfile("OUTCAR", f"{j}/level{level}_step{j}.OUTCAR")
                    shutil.copyfile("CONTCAR", f"{j}/level{level}_step{j}.vasp")
                tmp_atoms = self.slide(tmp_atoms, n, theta, disp, scheme=scheme)
                

    # Testing done, working! However, restart option not coded.
    def run_parallel(self, atoms, n, disp, theta, scheme=None):
        cwd = os.getcwd()
        try:
            os.mkdir(f"{int((theta/pi)*180 + 0.1)}")
        except FileExistsError:
            pass
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        traj = Trajectory(f"Trajectory_{int((theta/pi)*180 + 0.1)}_in.traj","w")
        for j in range(self.n_steps):
            atoms = self.slide(atoms, n, theta, disp, scheme=scheme)
            try:
                os.mkdir(f"{j+1}")
            except FileExistsError:
                pass
            shutil.copy(cwd + "/job.sh", f"./{j+1}")
            shutil.copy(cwd + "/run.py", f"./{j+1}")
            os.chdir(f"./{j+1}")
            write("POSCAR", atoms)
            traj.write(atoms)
            subprocess.run(["sbatch", "job.sh"])
            os.chdir("../")
        os.chdir(cwd)
    
    # Testing done, working!
    def get_output_Trajectory(self, atoms, theta, calc=None):
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        traj = Trajectory(f"Trajectory_{int((theta/pi)*180 + 0.1)}_out.traj","w")
        traj.write(atoms)
        for i in range(self.n_steps):
            os.chdir(f"./{i+1}")
            if calc=='parallel':
                tmp_atoms = read("OUTCAR", index=-1)
            if calc=='serial':
                tmp_atoms = read(f'level{3}_step{i+1}.OUTCAR')
            traj.write(tmp_atoms)
            os.chdir("../")
        os.chdir(cwd)
    
    # Testing done, working! **Note: This does not give the coordinates with PBC correction.
    def get_layer_movement(self, atoms, n, disp, theta, traj, nth_layer=None, index=None):    # Layer counting starts from 0. The grain boundary is 10th and 11th layers.
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
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

    # Testing done, working! However, calc_type is not coded for property=Stress.
    def analysis(self, theta, calc_type=None, property=None):
        cwd = os.getcwd()
        os.chdir(cwd + f"/{int((theta/pi)*180 + 0.1)}")
        if property == "Energy":
            E = np.array([])
            for i in range(self.n_steps):
                os.chdir(f"./{i+1}/")
                if calc_type == "parallel":
                    atoms = read("OUTCAR")
                elif calc_type == "serial":
                    atoms = read(f"level{3}_step{i+1}.OUTCAR")
                E = np.append(E, atoms.get_potential_energy()/len(atoms))
                os.chdir("./../")
            os.chdir(cwd)
            return E
        elif property == "Stress":
            S = np.array([])
            for i in range(self.n_steps):
                os.chdir(f"./{i+1}/")
                atoms = read("OUTCAR")
                S = np.append(S, atoms.get_stress()[2])
                os.chdir("./../")
            os.chdir(cwd)
            return S

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


# Testing done, working for n=1!
def symmetrize_sigma3_gb(atoms, layer, n):    # For surface charging, n is the number of fixed layers.
    cell = atoms.get_cell()
    layer_size = layer.get_cell()[1][1]
    y_size = cell[1][1]
    for i in range(n):
        for atom in atoms:
            atom.y = atom.y + layer_size
        y_size = y_size + layer_size
        atoms.set_cell([cell[0][0], y_size, cell[2][2]])
        atoms = atoms + layer
    tmp_atoms = atoms.copy()
    tmp_cell = atoms.get_cell()
    for atom in tmp_atoms:
        if atom.y > n*layer_size:
            atom.y = n*layer_size - atom.y
    del tmp_atoms[[atom.index for atom in tmp_atoms if atom.y < n*layer_size and atom.y > 0]]
    new_atoms = atoms + tmp_atoms
    y_total = tmp_cell[1][1] + cell[1][1]
    new_atoms.set_cell([cell[0][0], y_total, cell[2][2]])
    new_atoms.center()
    c = FixAtoms(indices=[atom.index for atom in new_atoms if atom.y>cell[1][1] and atom.y<(cell[1][1]+layer_size)])
    new_atoms.set_constraint(c)
    new_atoms.center(vacuum = 15, axis = (1))
    return new_atoms

# Testing done, working!
def cure_Si_surface_with_H(atoms):
    nat_cut = natural_cutoffs(atoms, mult=1)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    cell = atoms.get_cell()[1][1]
    for i in range(len(atoms)):
        indices, _ = nl.get_neighbors(i)
        if len(indices) == 3:
            if atoms[i].y > cell/2:
                coord = (atoms[i].x, atoms[i].y + 1.5, atoms[i].z)
            if atoms[i].y < cell/2:
                coord = (atoms[i].x, atoms[i].y - 1.5, atoms[i].z)
            H_atom = Atom("H", coord)
            atoms.append(H_atom)
    return atoms

class NEB_ASE:
    def run_neb_ase(self, n_images, kpts, initial=None, final=None, restart=None):
        # Provide initial and final only if restart=None/False.
        # n_images is the number of images in between the initial and final image.
        if restart==True:
            images = read(f"I2F1.traj@-{n_images+2}:")  # Use this if the NEB is being restarted. .traj@-x: is used where x is the number of images in the band.
            neb = NEB(images)
        else:
            images = [initial]
            images += [initial.copy() for i in range(n_images)]
            images += [final]
            neb = NEB(images)
            # Interpolating linearly the positions of the six middle images:
            neb.interpolate()
        for image in images[1:n_images+1]:
            calc = get_base_calc()
            set_vasp_key(calc, 'ibrion', -1)
            set_vasp_key(calc, 'nsw', 0)
            set_vasp_key(calc, 'kpts', kpts)
            image.calc = calc
        optimizer = BFGS(neb, Trajectory="I2F.traj")
        optimizer.run(fmax=0.01)
        return images

    def get_neb_plot(self, n_images, images, initial_outcar_path, final_outcar_path):
        # This is to get the energies of the intial and final images from the geo-opt folder as their energies are not included in NEB.
        for i in range(0,len(images),(n_images+2)):
            en_i = read(initial_outcar_path).get_potential_energy()
            images[i].calc = SPC(atoms=images[0], energy=en_i, forces=np.zeros((len(images[0]),3)))
            en_f = read(final_outcar_path).get_potential_energy()
            images[i+n_images+1].calc = SPC(atoms=images[0], energy=en_i, forces=np.zeros((len(images[0]),3)))
        for i in range(0,len(images),(n_images+2)):
            nebtools = NEBTools(images[i:i+n_images+2])
            fig = nebtools.plot_band()
        return fig

def dos(atoms, dense_k_points):
    calc = get_base_calc()
    set_vasp_key(calc, 'ismear', -5)
    set_vasp_key(calc, 'icharg', 11)
    set_vasp_key(calc, 'lorbit', 11)
    set_vasp_key(calc, 'nedos', 1000)   # It should be between 1000 to 3000 based on the accuracy required.
    set_vasp_key(calc, 'kpts', dense_k_points)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    return atoms

# Testing done, working!
def get_neighbor_list(atoms):
    nat_cut = natural_cutoffs(atoms, mult=1)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    f = open("Neighbor_List.txt", "w")
    # Printing neighbor list for all atoms
    for r in range(len(atoms)):
        indices, offsets = nl.get_neighbors(r)
        f.write(f"The neighbors for {r} atom are: " + str(indices) + "\n")
        pos = []
        f.write("Position                                Distance\n")
        for i, offset in zip(indices, offsets):
            pos = atoms.positions[i] + offset @ atoms.get_cell()    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
            dist = ((atoms[r].x - pos[0])**2 + (atoms[r].y - pos[1])**2 + (atoms[r].z - pos[2])**2)**(1/2)
            f.write(str(pos) + " " + str(dist) + "\n")
    # Printing coordination number for all atoms
    f.write("\nCoordination numbers for all the atoms: \n")
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        f.write(str(i) + " " + str(len(indices)) + "\n")

# Testing done, working!
def check_run_completion(location):
    cwd = os.getcwd()
    os.chdir(location)
    with open("OUTCAR", "r"):
        content = f.readlines()
        c = 0
        for line in content:
            if line == " General timing and accounting informations for this job:\n":
                c = c + 1
        if c == 0:
            print("Simulation not completed in " + location)
    os.chdir(cwd)

# Testing done, working!
def get_plot_settings(fig, x_label=None, y_label=None, fig_name=None, show=None):
    ax = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "x", direction = "in")
    ax.tick_params(axis = "y", direction = "in")
    ax.ticklabel_format(useOffset=False)    # To prevent the power scaling of the axes
    if x_label!=None:
        plt.xlabel(x_label)
    if y_label!=None:
        plt.ylabel(y_label)
    plt.legend(frameon = False, loc = "upper left", fontsize = "7", bbox_to_anchor = (0.2, 1))
    if fig_name!=None:
        plt.savefig(fig_name, bbox_inches='tight')
    if show == True:
        plt.show()

"""LAMMPS related codes"""

# Testing done, working!
def fixed_layer_coord(atoms, n):    # For LAMMPS.
    cell = atoms.get_cell()
    layer_size = (atoms.get_cell()[0][0])/(2*n)
    xlo1 = layer_size*int(n/2)
    xhi1 = layer_size*(int(n/2)+1)
    xlo2 = layer_size*(n+int(n/2))
    xhi2 = layer_size*(n+int(n/2)+1)
    return [xlo1, xhi1, xlo2, xhi2]

# Testing done, working!
def edit_lammps_script(file_path, x, Tdamp, dt, N):
    with open(file_path, 'r') as f:
        content = f.read()
    content = content.replace("xlo1", str(x[0]))
    content = content.replace("xhi1", str(x[1]))
    content = content.replace("xlo2", str(x[2]))
    content = content.replace("xhi2", str(x[3]))
    content = content.replace("Tdamp", str(Tdamp))
    content = content.replace("deltat", str(dt))
    content = content.replace("NNN", str(N))
    with open(file_path, 'w') as f:
        f.write(content)