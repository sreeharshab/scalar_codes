import os
import shutil
import subprocess
import numpy as np
from numpy import arctan, sin, cos, tan, pi
from matplotlib import pyplot as plt
from matplotlib import gridspec
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
import pandas as pd
import re

logo = r""" ____  _            _ _                 
|  _ \(_)_ __   ___| (_)_ __   ___  ___ 
| |_) | | '_ \ / _ \ | | '_ \ / _ \/ __|
|  __/| | |_) |  __/ | | | | |  __/\__ \
|_|   |_| .__/ \___|_|_|_| |_|\___||___/
        |_|                             
        """
print(logo)
print("Sree Harsha Bhimineni\nYantao Xia\nUniversity of California, Los Angeles\nbsreeharsha@ucla.edu")

"""VASP related codes using ASE"""

def get_base_calc():
    base_calc = Vasp(
        gga="PE",
        lreal="Auto",
        lplane=True,
        lwave=False,
        lcharg=False,
        ncore=16,
        prec="Normal",
        encut=300,
        ediff=1e-6,
        algo="VeryFast",
        ismear=-5,
        gamma=True,
    )
    return base_calc

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
def cell_opt(atoms, kpts, npoints=5, eps=0.04, addnl_settings=None):
    calc = get_base_calc()

    calc.set(ibrion=-1, nsw=0, kpts=kpts)
    if addnl_settings!=None:
        keys = addnl_settings.keys()
        for key in keys:
            set_vasp_key(calc, key, addnl_settings[key])
    atoms.calc = calc

    eos = calculate_eos(atoms, npoints=npoints, eps=eps, trajectory="eos.traj")

    v, e, B = eos.fit()
    eos.plot(filename="eos.png")
    opt_factor = v / atoms.get_volume()
    atoms.cell = atoms.cell * opt_factor
    write("opted_cell.vasp", atoms)
    return atoms


def axis_opt(atoms, kpts, axis, npoints=5, eps=0.04, addnl_settings=None):
    """relax one vector of the cell"""
    ens = np.zeros(npoints)
    vols = np.zeros(npoints)

    # by defualt, shrink/expand axis by 4%
    factors = np.linspace(1 - eps, 1 + eps, npoints)

    for ifactor, factor in np.ndenumerate(factors):
        atoms_tmp = atoms.copy()
        atoms_tmp.cell[axis] = atoms.cell[axis] * factor
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, kpts=kpts, directory=f"{factor:.2f}")
        if addnl_settings!=None:
            keys = addnl_settings.keys()
            for key in keys:
                set_vasp_key(calc, key, addnl_settings[key])
        atoms_tmp.calc = calc
        ens[ifactor] = atoms_tmp.get_potential_energy()
        vols[ifactor] = atoms_tmp.get_volume()

    eos = EOS(volumes=vols, energies=ens, eos="sj")
    v0, e0, B = eos.fit()
    opt_factor = v0 / atoms.get_volume()
    atoms.cell[axis] = atoms.cell[axis] * opt_factor
    write("opted_axis.vasp", atoms)
    return atoms


# todo: Program restart option!
def geo_opt(atoms, mode="vasp", opt_levels=None, restart=None, fmax=0.02):
    def opt_by_vasp(calc, opt_levels, level):
        level_settings = opt_levels[level]
        # default settings when using built-in optimizer
        set_vasp_key(calc, 'ibrion', 2)
        set_vasp_key(calc, 'ediffg', -1e-2)
        set_vasp_key(calc, 'nsw', 500)
        set_vasp_key(calc, 'nelm', 500)
        # user-supplied overrides
        for key in level_settings.keys():
            set_vasp_key(calc, key, level_settings[key])

        atoms = read("CONTCAR")
        atoms.calc = calc
        atoms.get_potential_energy()
        calc = get_base_calc()
        atoms = read("OUTCAR", index=-1)
        shutil.copyfile("CONTCAR", f"opt{level}.vasp")
        shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
        shutil.copyfile("vasp.out", f"opt{level}.out")
        return atoms
    
    def opt_by_ase(calc, opt_levels, level):
        # default settings when using ase optimizer
        set_vasp_key(calc, 'ibrion', -1)
        set_vasp_key(calc, 'nsw', 0)
        # user-supplied overrides
        level_settings = opt_levels[level]
        for key in level_settings.keys():
            if key in ['nsw', 'ibrion', 'ediffg']:
                continue
            set_vasp_key(calc, key, level_settings[key])

        atoms = read("CONTCAR")
        atoms.calc = calc
        opt = BFGS(atoms,
                    trajectory = f"opt{level}.traj",
                    logfile = f"opt{level}.log")
        opt.run(fmax=fmax)
        calc = get_base_calc()
        shutil.copyfile("CONTCAR", f"opt{level}.vasp")
        shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
        shutil.copyfile("vasp.out", f"opt{level}.out")
        return atoms
    
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
    if mode == 'vasp' and restart != True:
        write("CONTCAR", atoms)
        for level in levels:
            if mode=="vasp":
                atoms = opt_by_vasp(calc, opt_levels, level)
            elif mode=="ase":
                atoms = opt_by_ase(calc, opt_levels, level)
    
    if restart == True:
        last_level = 0
        for level in levels:
            if os.path.exists(f"opt{level}.out"):
                last_level = level
        levels = list(levels)
        for level in range(last_level+1,levels[-1]+1):
            if mode=="vasp":
                atoms = opt_by_vasp(calc, opt_levels, level)
            elif mode=="ase":
                atoms = opt_by_ase(calc, opt_levels, level)
            
    return atoms

def bader(atoms, kpts, valence_electrons, addnl_settings=None):
    def run_vasp(atoms):
        calc = get_base_calc()
        
        calc.set(
            ibrion=-1, nsw=0, lorbit=12, lcharg=True, laechg=True, kpts=kpts
        )
        if addnl_settings!=None:
            for key in addnl_settings.keys():
                set_vasp_key(calc, key, addnl_settings[key])
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
        ocharges = np.array([])
        for atom in atoms:
            ocharges = np.append(ocharges, valence_electrons[atom.symbol])
        ocharges = np.array([int(i) for i in ocharges])
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

    def run_vasp(self, kpts, valence_electrons, addnl_settings=None):
        atoms = self.atoms
        calc = get_base_calc()
        calc.set(ibrion=-1, nsw=0, isym=-1, prec="Accurate", kpts=kpts, lwave=True, lcharg=True)
        if addnl_settings!=None:
            for key in addnl_settings.keys():
                set_vasp_key(calc, key, addnl_settings[key])

        nelect = 0
        for atom in atoms:
            nelect = nelect + valence_electrons[atom.symbol]
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
            ax2.plot(-data_cohp[:, i * 2 + 4], data_cohp[:, 0], color="grey", linestyle="dashdot")
            ax2.set_ylim(icohp_ylim)  # [-10, 6]
            ax2.set_xlim(icohp_xlim)  # [-0.01, 1.5]
            ax2.set_xlabel("-ICOHP (eV)", color="grey", fontsize="large")
            ax2.xaxis.tick_top()
            ax2.xaxis.set_label_position("top")
            ax2.tick_params(axis="x", colors="grey")

            # legends
            ax1.axvline(0, color="k", linestyle=":", alpha=0.5)
            ax1.axhline(0, color="k", linestyle="--", alpha=0.5)
            labelx = min(icohp_xlim) + 0.2
            labely = min(icohp_ylim) + 0.5

            ax2.annotate(
                labels_cohp[i],
                xy=(labelx, labely),
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", fc="w", alpha=0.5),
            )
            ax2.annotate(
                f"{-icohp_ef[i]:.3f}",
                xy=(labelx, -0.05),
                ha="left",
                va="bottom",
                color="black",
            )
            fig.savefig(
                f"cohp-{i+1}.png",
                dpi=500,
                bbox_inches="tight",
                transparent=True,
            )
            plt.close()

class NEB:
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

def frequency(atoms, mode="vasp", scheme="serial", settings=None):
    calc = get_base_calc()
    if settings["kpts"]==None:
        settings["kpts"]=atoms.info["kpts"]
    keys = settings.keys()
    for key in keys:
        set_vasp_key(calc, key, settings[key])

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
        if constr!=[]:
            vib_indices = [a.index for a in atoms if a.index not in constr[0].index]
        elif constr==[]:
            vib_indices = [a.index for a in atoms]
        if scheme=="serial":
            vib = Vibrations(atoms, indices=vib_indices)
            vib.run()  # this will save json files
            vib.summary(log="vibrations.txt")
            return vib.get_energies()
        elif scheme=="parallel":
            for index in vib_indices:
                try:
                    os.mkdir(f"{index}")
                except FileExistsError:
                    pass
            os.chdir(f"{index}")
            # todo: complete parallel scheme

class surface_charging:
    def __init__(self) -> None:
        pass

    # Parsing the OUTCAR to get NELECT.
    def get_PZC_nelect(self):
        os.chdir("PZC_calc")
        with open('OUTCAR', 'r') as f:
            for line in f:
                match = re.search(r'NELECT\s+=\s+(\d+\.\d+)', line)
                if match:
                    PZC_nelect = match.group(1)
                    break
        os.chdir("../")
        return float(PZC_nelect)
    
    """
    n_nelect are the number of nelects you want to run surface charging for.
    width_nelect is the difference between each nelect.
    Example:
    If PZC_nelect = 40, n_nelect = 4 and width_nelect = 0.25, the values of nelect will be [39.50, 39.75, 40.25, 40.50].
    **kwargs are used to get the arguments of the symmetrize function.
    """
    def run(self, atoms, opt_levels, n_nelect, width_nelect, symmetrize_function=None, **kwargs):
        if symmetrize_function!=None:
            atoms = symmetrize_function(atoms, **kwargs)
            write("POSCAR_sym", atoms)

        # Running a neutral solvation (PZC) calculation.
        PZC_calc = False
        last_level = len(opt_levels)
        if os.path.exists(f"./POSCAR_solvated"):
            PZC_calc = True
        if PZC_calc==False:
            try:
                os.mkdir(f"PZC_calc")
            except FileExistsError:
                pass
            os.chdir("PZC_calc")
            geo_opt(atoms, mode="vasp", opt_levels=opt_levels)
            shutil.copyfile("CONTCAR", "../POSCAR_solvated")
            os.chdir("../")
        
        PZC_nelect = self.get_PZC_nelect()
        
        n_nelect = int(n_nelect/2)
        nelect = np.arange(PZC_nelect-n_nelect*width_nelect, PZC_nelect+n_nelect*width_nelect+width_nelect, width_nelect)
        nelect = np.delete(nelect,n_nelect) # Excluding PZC_nelect.
        
        for i in nelect:
            try:
                os.mkdir(f"{i}")
            except FileExistsError:
                pass
            shutil.copyfile("POSCAR_solvated", f"./{i}/POSCAR")
            shutil.copyfile("geo_opt.py", f"./{i}/geo_opt.py")
            shutil.copyfile("optimize.sh", f"./{i}/optimize.sh")
            os.chdir(f"{i}")
            
            subprocess.run(["sed", "-i", f"s/nnn/{str(i)}/g", "geo_opt.py"])

            subprocess.run(["sbatch", "optimize.sh"])
            os.chdir("../")
    
    def analyse(self):
        PZC_nelect = self.get_PZC_nelect()
        os.rename("PZC_calc", f"{PZC_nelect}")
        subprocess.run(["python", "new_plot_sc.py", "-n", f"{PZC_nelect}"])
        os.rename(f"{PZC_nelect}", "PZC_calc")

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

def analyse_GCBH(save_data=None, energy_operation=None, label=None):
    if save_data==None or save_data==True:
        E = []
        f = open("energies.txt", "w")
        traj = Trajectory("grandcanonical.traj", "w")
        os.chdir("./opt_folder")
        dirs = os.listdir()
        dirs = sorted(dirs)
        for dir in dirs[:-1]:
            os.chdir(dir)
            atoms = read("opt3.OUTCAR")
            e = atoms.get_potential_energy()
            if energy_operation==None or energy_operation==False:
                E.append(e)
            elif energy_operation==True:
                E.append(energy_operation(e))
            f.write(f"{energy_operation(e)}\n")
            traj.write(atoms)
            os.chdir("../")
        os.chdir("../")
    elif save_data==False:
        E = []
        with open("energies.txt", "r") as f:
            data = f.readlines()
            for i in data:
                    E.append(float(i))

    E = np.array(E)
    E = np.expand_dims(E, axis=0)

    fig = plt.figure(dpi = 200, figsize=(4.5,2))

    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1], figure=fig)  # Adjust the height_ratios as needed

    ax1 = plt.subplot(gs[0])
    im = ax1.imshow(E, aspect=12, cmap='YlOrRd')
    ax1.yaxis.set_ticks([])
    ax1.set_xlabel('Step')

    ax2 = plt.subplot(gs[1])
    colorbar = plt.colorbar(im, cax=ax2, orientation='horizontal')
    colorbar.set_label(label)

    plt.subplots_adjust(hspace=1.4, bottom=0.25)
    
    plt.savefig("analysis_horizontal.png")

    fig = plt.figure(dpi=200, figsize=(2,4))
    E = np.squeeze(E, axis=0)
    for i in E:    
        plt.hlines([i], [1], [xi+0.2 for xi in [1]], linewidth=0.5, color="navy")
    ax = fig.gca()
    for axis in ['left']:
        ax.spines[axis].set_linewidth(1.5)
    for axis in ['top', 'bottom', 'right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(bottom = False, top = False, left = True, right = False)
    plt.xticks([])
    ax.tick_params(axis = "x", direction = "in")
    ax.tick_params(axis = "y", direction = "out")
    ax.ticklabel_format(useOffset=False)
    plt.ylabel(label)
    plt.savefig("analysis_vertical.png", bbox_inches="tight")

def get_neighbor_list(atoms):
    nat_cut = natural_cutoffs(atoms, mult=1)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    f = open("Neighbor_List.txt", "w")
    # Printing neighbor list for all atoms
    for r in range(len(atoms)):
        indices, offsets = nl.get_neighbors(r)
        f.write(f"The neighbors for atom {r} are: " + "\n")
        pos = []
        f.write("{:<10} {:<10} {:<38} {:<10}\n".format("Index", "Symbol", "Position", "Distance"))
        for i, offset in zip(indices, offsets):
            pos = atoms.positions[i] + offset @ atoms.get_cell()    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
            dist = ((atoms[r].x - pos[0])**2 + (atoms[r].y - pos[1])**2 + (atoms[r].z - pos[2])**2)**(1/2)
            f.write("{:<10} {:<10} {:<38} {:<10}\n".format(str(i), str(atoms[i].symbol), str(pos), str(dist)))
    # Printing coordination number for all atoms
    f.write("\nCoordination numbers for all the atoms: \n")
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        f.write(str(i) + " " + str(len(indices)) + "\n")

def check_run_completion(location, output="OUTCAR"):
    cwd = os.getcwd()
    os.chdir(location)
    with open(output, "r") as f:
        content = f.readlines()
        c = 0
        for line in content:
            if line == " General timing and accounting informations for this job:\n":
                c = c + 1
    os.chdir(cwd)
    if c == 0:
        return False
    else:
        return True

def get_cell_info(atoms):
    cell = atoms.get_cell()
    
    volume = cell.volume
    lengths = cell.lengths()
    angles = cell.angles()

    print("Cell Vectors:")
    print(cell[:])
    print("Volume of the cell:", volume)
    print("Length of vector a:", lengths[0])
    print("Length of vector b:", lengths[1])
    print("Length of vector c:", lengths[2])
    print("Angle alpha:", angles[0])
    print("Angle beta:", angles[1])
    print("Angle gamma", angles[2])

def get_selective_dynamics(file_name, index):
    with open(file_name,'r') as f:
        lines = f.readlines()
    
    line = lines[index+9]

    if "T" in line:
        return True
    elif "F" in line:
        return False

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
        for atom in atoms:
                d = np.array([])
                indices, offsets = nl.get_neighbors(c)
                for i, offset in zip(indices, offsets):
                    pos = atoms.positions[i] + offset @ cell    # Code to account for periodic boundary condition. Offset list consists something like [0,1,0] and 
                                                                # offset@atoms.get_cell() gives [0,7.73277,0] where 7.73277 is the b vector length.
                    d = np.append(d, self.neighbor_distance(atoms, c, pos))
                c = c + 1
        close_atoms = 0
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
    
    def run_serial(self, atoms, opt_levels, disp, theta, scheme=None, n=None, restart=None):
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
                atoms = self.slide(atoms, theta, disp, scheme=scheme, n=n)
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
                    shutil.copyfile("vasp.out", f"{j+1}/level{level}_step{j+1}.out")
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
                        shutil.copyfile("OUTCAR", f"{last_step}/level{level}_step{last_step}.OUTCAR")
                        shutil.copyfile("CONTCAR", f"{last_step}/level{level}_step{last_step}.vasp")
                        shutil.copyfile("vasp.out", f"{last_step}/level{level}_step{last_step}.out")
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
                    shutil.copyfile("vasp.out", f"{j}/level{level}_step{j}.out")
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

def get_plot_settings(fig, x_label=None, y_label=None, fig_name=None, legend_location="upper left", show=None):
    ax = fig.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(bottom = True, top = True, left = True, right = True)
    ax.tick_params(axis = "x", direction = "in")
    ax.tick_params(axis = "y", direction = "in")
    ax.ticklabel_format(useOffset=False)    # To prevent the power scaling of the axes
    if x_label!=None:
        plt.xlabel(x_label, fontsize=12)
    if y_label!=None:
        plt.ylabel(y_label, fontsize=12)
    plt.legend(frameon = False, loc = legend_location, fontsize = "10")
    if fig_name!=None:
        plt.savefig(fig_name, bbox_inches='tight')
    if show == True:
        plt.show()

"""LAMMPS related codes"""

def fixed_layer_coord(atoms, n):    # For LAMMPS.
    cell = atoms.get_cell()
    layer_size = (atoms.get_cell()[0][0])/(2*n)
    xlo1 = layer_size*int(n/2)
    xhi1 = layer_size*(int(n/2)+1)
    xlo2 = layer_size*(n+int(n/2))
    xhi2 = layer_size*(n+int(n/2)+1)
    return [xlo1, xhi1, xlo2, xhi2]

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