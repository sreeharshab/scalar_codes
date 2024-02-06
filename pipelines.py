import os
import shutil
import subprocess
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import gridspec
from ase.io import read, write, Trajectory
from ase.calculators.vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.calculators.vasp.create_input import float_keys, exp_keys, string_keys, int_keys, bool_keys, list_int_keys, list_bool_keys, list_float_keys, special_keys, dict_keys
from ase.optimize import BFGS
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.neb import NEB
from ase.eos import calculate_eos
from ase.eos import EquationOfState as EOS
from ase.constraints import FixAtoms
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
import pandas as pd
import re
from distutils.dir_util import copy_tree

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

def cell_opt(atoms, kpts, npoints=5, eps=0.04, addnl_settings=None):
    """Optimizes the size of the simulation cell.

    :param atoms: Atoms to be optimized
    :type atoms: Atoms object
    :param kpts: KPOINTS used for the calculation
    :type kpts: list
    :param npoints: Number of optimization points for the calculation, defaults to 5
    :type npoints: int, optional
    :param eps: Variation in volume, defaults to 0.04
    :type eps: float, optional
    :param addnl_settings: Dictionary containing any additional VASP settings (either editing default settings of base_calc or adding more settings), defaults to None
    :type addnl_settings: dict, optional
    :return: Optimized atoms
    :rtype: Atoms object
    """
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
    """Optimizes the size of the required axis of the simulation cell.

    :param atoms: Atoms to be optimized
    :type atoms: Atoms object
    :param kpts: KPOINTS used for the calculation
    :type kpts: list
    :param axis: The axis to be optimized
    :type axis: int
    :param npoints: Number of optimization points for the calculation, defaults to 5
    :type npoints: int, optional
    :param eps: Variation in volume, defaults to 0.04
    :type eps: float, optional
    :param addnl_settings: Dictionary containing any additional VASP settings (either editing default settings of base_calc or adding more settings), defaults to None
    :type addnl_settings: dict, optional
    :return: Optimized atoms
    :rtype: Atoms object
    """
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


def geo_opt(atoms, mode="vasp", opt_levels=None, restart=None, fmax=0.02):
    """Performs geometry optimization on the system using inbuilt VASP optimizers (using the IBRION tag) or ASE optimizers.

    :param atoms: Atoms to be geometrically optimized
    :type atoms: Atoms object
    :param mode: Type of optimizer to be used, `"vasp"` for IBRION=2 and `"ase"` for BFGS, defaults to "vasp"
    :type mode: str, optional
    :param opt_levels: Dictionary of dictionaries, each dictionary containing settings for each level of calculation, defaults to 
        ``{
        1:{"kpts":[3,3,3]}, 
        2:{"kpts":[5,5,5]}, 
        3:{"kpts":[7,7,7]}
        }``
    :type opt_levels: dict, optional
    :param restart: Restarting a calculation if `restart=True`, defaults to None
    :type restart: bool, optional
    :param fmax: Maximum force on optimized atoms, defaults to 0.02
    :type fmax: float, optional
    """
    def save_files(level):
        shutil.copyfile("CONTCAR", f"opt{level}.vasp")
        shutil.copyfile("OUTCAR", f"opt{level}.OUTCAR")
        shutil.copyfile("vasp.out", f"opt{level}.out")
    
    def opt_by_vasp(opt_levels, level):
        calc = get_base_calc()
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
        atoms = read("OUTCAR", index=-1)
        save_files(level)
        return atoms
    
    def opt_by_ase(opt_levels, level):
        calc = get_base_calc()
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
        save_files(level)
        return atoms
    
    if not opt_levels:
        # for bulks.
        # other systems: pass in argument
        opt_levels = {
            1: {"kpts": [3, 3, 3]},
            2: {"kpts": [5, 5, 5]},
            3: {"kpts": [7, 7, 7]},
        }
    levels = opt_levels.keys()

    if restart != True:
        write("CONTCAR", atoms)
        for level in levels:
            if mode=="vasp":
                atoms = opt_by_vasp(opt_levels, level)
            elif mode=="ase":
                atoms = opt_by_ase(opt_levels, level)
    
    if restart == True:
        last_level = 0
        for level in levels:
            if os.path.exists(f"opt{level}.out"):
                last_level = level
        levels = list(levels)
        # if a new calc is started with restart=True
        if last_level==0:
            last_level = levels[0]-1
            if os.path.exists(f"CONTCAR"):
                pass
            else:
                write("CONTCAR", atoms)
        for level in range(last_level+1,levels[-1]+1):
            if mode=="vasp":
                atoms = opt_by_vasp(opt_levels, level)
            elif mode=="ase":
                atoms = opt_by_ase(opt_levels, level)
            
    return atoms

def bader(atoms, kpts, valence_electrons, addnl_settings=None):
    """Performs bader charge analysis on the system. Charges can be viewed in ACF.dat file or using ase gui and choosing the Initial Charges label in the view tab.

    :param atoms: Atoms for which charge should be determined
    :type atoms: Atoms object
    :param kpts: KPOINTS used for the calculation
    :type kpts: list
    :param valence_electrons: Dictionary containing the symbol of atoms as key and the corresponding valence electrons from POTCAR as value
    :type valence_electrons: dict
    :param addnl_settings: Dictionary containing any additional VASP settings (either editing default settings of base_calc or adding more settings), defaults to None
    :type addnl_settings: dict, optional
    """
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
    """Performs COHP analysis on the system. The output is saved as cohp-1.png.
    """
    def __init__(self, atoms, bonds, lobsterin_template=None):
        """Initializes the COHP class.

        :param atoms: Atoms on which COHP analysis is performed.
        :type atoms: Atoms object
        :param bonds: List of lists, where each list contains the indexes of two bonding atoms
        :type bonds: list
        :param lobsterin_template: Path of file which contains lobster template. If no path is given, the default template is used, defaults to None
        :type lobsterin_template: str, optional
        """
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
        """_summary_

        :param kpts: _description_
        :type kpts: _type_
        :param valence_electrons: _description_
        :type valence_electrons: _type_
        :param addnl_settings: _description_, defaults to None
        :type addnl_settings: _type_, optional
        """
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

class frequency:
    def __init__(self, atoms, vib_indices=None):
        self.atoms = atoms
        if vib_indices==None:
            vib_indices = self.get_vib_indices()
        elif vib_indices!=None:
            vib_indices = vib_indices
        self.vib_indices = vib_indices

    def get_vib_indices(self):
        atoms = self.atoms
        constr = atoms.constraints
        constr = [c for c in constr if isinstance(c, FixAtoms)]
        if constr!=[]:
            vib_indices = [a.index for a in atoms if a.index not in constr[0].index]
        elif constr==[]:
            vib_indices = [a.index for a in atoms]
        return vib_indices
    
    def run(self, kpts=None, mode="ase", scheme=None, addnl_settings=None):
        atoms = self.atoms
        calc = get_base_calc()
        if addnl_settings!=None:
            keys = addnl_settings.keys()
            for key in keys:
                set_vasp_key(calc, key, addnl_settings[key])

        if mode == "vasp":
            # avoid this on large structures, use ase instead
            # ncore/npar unusable, leads to kpoint errors
            # isym must be switched off, leading to large memory usage
            calc.set(
                kpts=kpts,
                ibrion=5,
                potim=0.015,
                nsw=500,
                ncore=None,
                npar=None,
                isym=0,
            )
            atoms.calc = calc
            atoms.get_potential_energy()

        elif mode == "ase":
            calc.set(kpts=kpts, lwave=True, isym=-1)  # according to michael
            atoms.calc = calc
            vib_indices = self.vib_indices

            if scheme == "serial":
                vib = Vibrations(atoms, indices=vib_indices)
                vib.run()  # this will save json files
        
            elif scheme == "parallel":
                for indice in vib_indices:
                    os.mkdir(f"{indice}")
                    shutil.copyfile("./freq.py", f"./{indice}/freq.py")
                    shutil.copyfile("./freq.sh", f"./{indice}/freq.sh")
                    shutil.copyfile("./POSCAR", f"./{indice}/POSCAR")
                    os.chdir(f"{indice}")
                    subprocess.run(["sed", "-i", f"s/iii/{str(indice)}/g", "freq.py"])
                    subprocess.run(["sbatch", "freq.sh"])
                    os.chdir("../")
    
    # todo: parse OUTCAR frequencies and modes for mode="vasp"
    def analysis(self, mode, potentialenergy, temperature, copy_json_files=None, **kwargs):
        """Note: This method only works for `mode="ase"`.

        :param atoms: _description_
        :type atoms: _type_
        :param potentialenergy: _description_
        :type potentialenergy: _type_
        :param temperature: _description_
        :type temperature: _type_
        :param copy_json_files: True only if `scheme="parallel"`, defaults to None
        :type copy_json_files: bool, optional
        """
        atoms = self.atoms
        vib_indices = self.vib_indices
        vib = Vibrations(atoms, indices=vib_indices)
        if copy_json_files==True:
            for indice in vib_indices:
                os.chdir(f"./{indice}")
                copy_tree("./vib", "../vib")
                os.chdir("../")
        vib.run()
        vib.summary(log="vibrations.txt")
        vib_energies = vib.get_energies()
        vib_energies = np.array([i for i in vib_energies if i.imag==0])
        if mode=="Harmonic":    
            thermo = HarmonicThermo(vib_energies = vib_energies, potentialenergy = potentialenergy)
            S = thermo.get_entropy(temperature)
            H = thermo.get_helmholtz_energy(temperature)  
            U = thermo.get_internal_energy(temperature)
            return S,H,U
        if mode=="IdealGas":
            thermo = IdealGasThermo(vib_energies = vib_energies, potentialenergy = potentialenergy, **kwargs)
            H = thermo.get_enthalpy(temperature)
            S = thermo.get_entropy(temperature)
            G = thermo.get_gibbs_energy(temperature)
            return H,S,G
    
    def check_vib_files(self):
        f = open("file_size.txt", "w")
        vib_indices = self.vib_indices
        for indice in vib_indices:
            os.chdir(f"./{indice}/vib")
            files = os.listdir()
            for file in files:
                f.write(str(indice) + "  " + str(os.path.getsize(file)) + "\n")
            os.chdir("../../")


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
    Use custom_nelect when you know PZC_nelect. Leave n_nelect=None and width_nelect=None when custom_nelect is used.
    """
    def run(self, atoms, opt_levels, n_nelect=None, width_nelect=None, custom_nelect=None, symmetrize_function=None, **kwargs):
        if symmetrize_function!=None:
            atoms = symmetrize_function(atoms, **kwargs)
            write("POSCAR_sym", atoms)

        # Running a neutral solvation (PZC) calculation.
        PZC_calc = False
        if os.path.exists(f"./POSCAR_solvated"):
            PZC_calc = True
        if PZC_calc==False:
            try:
                os.mkdir(f"PZC_calc")
            except FileExistsError:
                pass
            os.chdir("PZC_calc")
            geo_opt(atoms, mode="vasp", opt_levels=opt_levels, restart=True)
            shutil.copyfile("CONTCAR", "../POSCAR_solvated")
            os.chdir("../")
        
        if n_nelect!=None and width_nelect!=None:
            PZC_nelect = self.get_PZC_nelect()
            n_nelect = int(n_nelect/2)
            nelect = np.arange(PZC_nelect-n_nelect*width_nelect, PZC_nelect+n_nelect*width_nelect+width_nelect, width_nelect)
            nelect = np.delete(nelect,n_nelect) # Excluding PZC_nelect.
        elif custom_nelect!=None:
            nelect = custom_nelect
        
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

def dos(atoms, kpts, addnl_settings=None):
    calc = get_base_calc()
    if addnl_settings!=None:
        for key in addnl_settings.keys():
            set_vasp_key(calc, key, addnl_settings[key])
    set_vasp_key(calc, 'ismear', -5)
    set_vasp_key(calc, 'icharg', 11)    # Obtain CHGCAR from single point calculation.
    set_vasp_key(calc, 'lorbit', 11)
    set_vasp_key(calc, 'nedos', 1000)   # It should be between 1000 to 3000 based on the accuracy required.
    set_vasp_key(calc, 'kpts', kpts)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    return atoms
    # todo: Plotting DOS

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

class benchmark:
    def __init__(self, cores):
        self.cores = cores

    def submit_jobs(self):
        cores = self.cores
        for core in cores:
            os.mkdir(f"{core}")
            shutil.copyfile("job.sh", f"{core}/job.sh")
            shutil.copyfile("run.py", f"{core}/run.py")
            shutil.copyfile("POSCAR", f"{core}/POSCAR")
            os.chdir(f"./{core}")
            subprocess.run(["sed", "-i", f"s/ccc/{str(core)}/g", "job.sh"])
            subprocess.run(["sbatch", "job.sh"])
            os.chdir("..")
    
    def get_benchmark(self, outcar_location="./"):
        cores = self.cores
        times = []
        cwd = os.getcwd()
        for core in cores:
            os.chdir(f"./{core}")
            os.chdir(f"{outcar_location}")
            outcar_times = []
            with open("OUTCAR", "r") as file:
                for line in file:
                    match = re.search(r"LOOP\+:\s+cpu time\s+([0-9]+\.[0-9]+)(?:\s+: real time [0-9]+\.[0-9]+)?", line)
                    if match:
                        outcar_times.append(float(match.group(1)))
            times.append(mean(outcar_times))
            os.chdir(cwd)
        times = [(time/3600) for time in times]
        fig = plt.figure(dpi = 200, figsize=(6,5))
        plt.plot(cores, times, 'o-', color="black")
        get_plot_settings(fig, x_label="Number of cores", y_label="Time per ionic step (hr)", fig_name="time.png")
        cpu_times = [time*core for time,core in zip(times,cores)]
        fig = plt.figure(dpi = 200, figsize=(6,5))
        plt.plot(cores, cpu_times, 'o-', color="black")
        get_plot_settings(fig, x_label="Number of cores", y_label="CPU Time per ionic step (cpu-hr)", fig_name="cpu_time.png")

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