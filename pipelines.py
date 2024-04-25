import os
import shutil
import subprocess
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib import gridspec
import ase
from ase.io import read, write, Trajectory
from ase.calculators.vasp import Vasp
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
from scipy.optimize import curve_fit


"""VASP related codes using ASE"""

def get_base_calc():
    base_calc = Vasp(
        gga="PE",
        lreal="Auto",
        lplane=True,
        lwave=False,
        lcharg=False,
        ncore=8,
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
    """Performs geometry optimization on the system using inbuilt VASP optimizer (IBRION=2) or ASE's BFGS optimizer.

    :param atoms: Atoms to be geometrically optimized
    :type atoms: Atoms object
    :param mode: Type of optimizer to be used, "vasp" for IBRION=2 and "ase" for BFGS, defaults to "vasp"
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
    
    def opt_by_ase(atoms, opt_levels, level):
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
                atoms = opt_by_ase(atoms, opt_levels, level)
    
    if restart == True:
        last_level = 0
        for level in levels:
            if os.path.exists(f"opt{level}.out"):
                last_level = level
        levels = list(levels)
        if last_level==0:
            last_level = levels[0]-1
        for level in range(last_level+1,levels[-1]+1):
            if mode=="vasp":
                if os.path.exists("CONTCAR"):
                    pass
                else:
                    write("CONTCAR", atoms)
                atoms = opt_by_vasp(opt_levels, level)
            elif mode=="ase":
                if os.path.exists(f"opt{level}.traj"):
                    try:
                        atoms = read(f"opt{level}.traj@-1")
                    except ase.io.formats.UnknownFileTypeError:
                        pass
                atoms = opt_by_ase(atoms, opt_levels, level)
            
    return atoms

def bader(atoms, kpts, valence_electrons, addnl_settings=None, restart=None):
    """Performs bader charge analysis on the system. Charges can be viewed in ACF.dat file or using ase gui and choosing the Initial Charges label in the view tab.

    :param atoms: Atoms for which charge should be determined
    :type atoms: Atoms object
    :param kpts: KPOINTS used for the calculation
    :type kpts: list
    :param valence_electrons: Dictionary containing the symbol of atoms as key and the corresponding valence electrons from POTCAR as value, example: {"Si":4, "H":1}
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

    if restart is not True:
        run_vasp(atoms)
        run_bader()
        atoms_with_charge = read_bader(atoms)
        return atoms_with_charge
    elif restart is True:
        if os.path.exists("AECCAR0") and os.path.exists("AECCAR2") and not os.path.exists("ACF.dat"):
            run_bader()
            atoms_with_charge = read_bader(atoms)
            return atoms_with_charge
        elif os.path.exists("ACF.dat"):
            atoms_with_charge = read_bader(atoms)
            return atoms_with_charge
        else:
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
        :param bonds: List of lists, where each list contains the indexes of two bonding atoms, example: [[0,1],[1,2],[2,3]]
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
        """Runs a single point calculation on the system to generate the WAVECAR and CHGCAR files required for COHP analysis.

        :param kpts: KPOINTS used for the calculation
        :type kpts: list
        :param valence_electrons: Dictionary containing the symbol of atoms as key and the corresponding valence electrons from POTCAR as value, example: {"Si":4, "H":1}
        :type valence_electrons: dict
        :param addnl_settings: Dictionary containing any additional VASP settings (either editing default settings of base_calc or adding more settings), defaults to None
        :type addnl_settings: dict, optional
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
        """Adds the information about the bonds to the lobsterin file.
        """
        lobsterin = "lobsterin"

        with open(f"{lobsterin}", "w+") as fhandle:
            for line in self.lobsterin_template:
                fhandle.write(line)
            for b in self.bonds:
                fhandle.write(f"cohpBetween atom {b[0]+1} and atom {b[1]+1}\n")

    def run_lobster(self):
        """Runs the lobster executable.
        """
        lobster = os.getenv("LOBSTER")
        subprocess.run([lobster], capture_output=True)

    def plot(self, cohp_xlim, cohp_ylim, icohp_xlim, icohp_ylim):
        """Plots the COHP and ICOHP data.

        :param cohp_xlim: Lowest and highest COHP values in eV for COHP plot, example: [-2.6, 1]
        :type cohp_xlim: list
        :param cohp_ylim: Lowest and highest energy values in eV for COHP plot, example: [-11, 8.5]
        :type cohp_ylim: list
        :param icohp_xlim: Lowest and highest ICOHP values in eV for ICOHP plot, example: [-0.01, 1.5]
        :type icohp_xlim: list
        :param icohp_ylim: Lowest and highest energy values in eV for ICOHP plot, example: [-11, 8.5]
        :type icohp_ylim: list
        :return: None
        :rtype: None
        """
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

class frequency:
    """Performs vibrational analysis on the system using VASP or ASE. Use ASE for calculations involving large systems as it supports a parallel scheme.
    """
    def __init__(self, atoms, vib_indices=None):
        """Initializes the frequency class.

        :param atoms: Atoms for which vibrational analysis is performed
        :type atoms: Atoms object
        :param vib_indices: Indices of the atoms to be vibrated, if None, all atoms with Selective Dynamics F are ignored, defaults to None
        :type vib_indices: list, optional
        """
        self.atoms = atoms
        if vib_indices==None:
            vib_indices = self.get_vib_indices()
        elif vib_indices!=None:
            vib_indices = vib_indices
        self.vib_indices = vib_indices

    def get_vib_indices(self):
        """Returns the indices of the atoms to be vibrated. All atoms with Selective Dynamics F are ignored.

        :return: Indices of the atoms to be vibrated
        :rtype: list
        """
        atoms = self.atoms
        constr = atoms.constraints
        constr = [c for c in constr if isinstance(c, FixAtoms)]
        if constr!=[]:
            vib_indices = [a.index for a in atoms if a.index not in constr[0].index]
        elif constr==[]:
            vib_indices = [a.index for a in atoms]
        return vib_indices
    
    def run(self, kpts=None, mode="ase", scheme=None, addnl_settings=None):
        """Runs frequency calculation on the system.

        :param kpts: KPOINTS used for the calculation, a list of KPOINTS are to be provided if mode is "vasp" and mode is "ase" with "serial" scheme, defaults to None
        :type kpts: list, optional
        :param mode:  Mode used to run the frequency calculation, supports `mode="ase"` and `mode="vasp"`, defaults to "ase"
        :type mode: str, optional
        :param scheme: Scheme used to run the frequency calculation, supports `scheme="serial"` and `scheme="parallel"` only when `mode="ase"`, defaults to None
        :type scheme: str, optional
        :param addnl_settings: Dictionary containing any additional VASP settings (either editing default settings of base_calc or adding more settings), defaults to None
        :type addnl_settings: dict, optional
        """
        atoms = self.atoms
        calc = get_base_calc()
        if addnl_settings!=None:
            keys = addnl_settings.keys()
            for key in keys:
                set_vasp_key(calc, key, addnl_settings[key])

        if mode == "vasp":
            # Avoid this on large structures, use mode="ase" instead.
            # ncore/npar unusable, leads to kpoint errors.
            # isym must be switched off, leading to large memory usage.
            assert kpts!=None, "kpts must be provided when mode is vasp"
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
                assert kpts!=None, "kpts must be provided when mode is ase and scheme is serial"
                vib = Vibrations(atoms, indices=vib_indices)
                vib.run()  # this will save json files
        
            elif scheme == "parallel":
                assert os.path.exists("./freq.py"), "freq.py is required to perform parallel calculation, see examples for the structure of freq.py"
                assert os.path.exists("./freq.sh"), "freq.sh is required to perform parallel calculation, see examples for the structure of freq.sh"
                for indice in vib_indices:
                    os.mkdir(f"{indice}")
                    shutil.copyfile("./freq.py", f"./{indice}/freq.py")
                    shutil.copyfile("./freq.sh", f"./{indice}/freq.sh")
                    shutil.copyfile("./POSCAR", f"./{indice}/POSCAR")
                    os.chdir(f"{indice}")
                    subprocess.run(["sed", "-i", f"s/iii/{str(indice)}/g", "freq.py"])
                    subprocess.run(["sbatch", "freq.sh"])
                    os.chdir("../")
    
    def analysis(self, mode, thermo_style, potentialenergy, temperature, pressure=None, copy_json_files=None, **kwargs):
        """Performs analysis after the frequency calculation.

        :param mode: Mode used to run the frequency calculation, supports `mode="ase"` and `mode="vasp"`, defaults to "ase"
        :type mode: str
        :param thermo_style: The class used from the thermochemistry module of ASE, for surfaces, use `thermo_style="Harmonic"` and for gases, use `thermo_style="IdealGas"`
        :type thermo_style: str
        :param potentialenergy: Potential energy of the system from geo_opt calculation
        :type potentialenergy: float
        :param temperature: Temperature at which the analysis is performed
        :type temperature: float
        :param pressure: Pressure at which the analysis is performed, defaults to None
        :type pressure: float, optional
        :param copy_json_files: Copies .json files from invidual atom's index directory to a common vib folder, True only if `scheme="parallel"` in the run method, defaults to None
        :type copy_json_files: bool, optional
        :return: _description_
        :rtype: _type_
        """
        atoms = self.atoms
        if mode=="ase":
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
        elif mode=="vasp":
            with open("OUTCAR", "r") as f:
                vib_energies = np.array([])
                for line in f:
                    match = re.search(r'\b\d+\.\d+\s+meV\b', line)
                    if match and "f/i" not in line:
                        value = float(match.group().split()[0])*(10**-3)
                        vib_energies = np.append(vib_energies,value)
        if thermo_style=="Harmonic":    
            thermo = HarmonicThermo(vib_energies = vib_energies, potentialenergy = potentialenergy)
            S = thermo.get_entropy(temperature)
            F = thermo.get_helmholtz_energy(temperature)  
            U = thermo.get_internal_energy(temperature)
            return S,F,U
        elif thermo_style=="IdealGas":
            assert pressure!=None, "pressure must be provided when mode is IdealGas"
            assert kwargs!=None, "geometry, symmetry number and spin must be provided when mode is IdealGas"
            thermo = IdealGasThermo(vib_energies = vib_energies, potentialenergy = potentialenergy, atoms = atoms, **kwargs)
            H = thermo.get_enthalpy(temperature)
            S = thermo.get_entropy(temperature, pressure)
            G = thermo.get_gibbs_energy(temperature, pressure)
            return H,S,G
    
    def check_vib_files(self):
        """Checks if the .json files are present in the individual atom's index directory. This method only works for `mode="ase"` with `scheme="parallel"`.
        """
        f = open("file_size.txt", "w")
        vib_indices = self.vib_indices
        for indice in vib_indices:
            try:
                os.chdir(f"./{indice}/vib")
            except FileNotFoundError:
                assert False, f"vib folder not found in {indice}. Make sure you ran a parallel frequency calculation with ase before using this method."
            files = os.listdir()
            for file in files:
                f.write(str(indice) + "  " + str(os.path.getsize(file)) + "\n")
            os.chdir("../../")


class surface_charging:
    """Performs surface charging calculation using VASPsol.
    """
    def __init__(self):
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
    Example:
    If PZC_nelect = 40, n_nelect = 4 and width_nelect = 0.25, the values of nelect will be [39.50, 39.75, 40.25, 40.50].
    **kwargs are used to get the arguments of the symmetrize function.
    """
    def run(self, atoms, opt_levels, n_nelect=None, width_nelect=None, custom_nelect=None, symmetrize_function=None, **kwargs):
        """Runs a surface charging calculation.

        :param atoms: Atoms used for surface charging calculation
        :type atoms: Atoms object
        :param opt_levels: Dictionary of dictionaries, each dictionary containing settings for each level of calculation
        :type opt_levels: dict
        :param n_nelect: Number of nelects, use this argument when PZC_nelect is not known defaults to None
        :type n_nelect: int, optional
        :param width_nelect: Difference between each nelect, use this argument along with n_nelect when PZC_nelect is not known, defaults to None
        :type width_nelect: float, optional
        :param custom_nelect: List of custom nelects, defaults to None
        :type custom_nelect: list, optional
        :param symmetrize_function: Function used to symmetrize atoms, defaults to None
        :type symmetrize_function: function, optional
        """
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
        
        assert not (n_nelect==None and width_nelect==None and custom_nelect==None), "Either provide n_nelect and width_nelect or custom_nelect."
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
    
    def parse_output(self):
        """Parses the output of the surface charging calculation to get the energy, Fermi energy and Fermi shift.

        :return: Energy, Fermi energy and Fermi shift of the calculation
        :rtype: tuple
        """
        class ParseError(Exception):
            def __init__(self,message):
                self.message = message

        with open("OUTCAR", "r") as f:
            outcar = f.readlines()
        outcar = "".join(outcar)

        # Parsing the OUTCAR to get the energy of the system.
        try:
            e_pattern = re.compile(r'entropy=\s+-?\d+.\d+\s+energy\(sigma->0\)\s+=\s+(-?\d+\.\d+)')
            e = re.findall(e_pattern, outcar)[-1]
            e = float(e)
        except:
            raise ParseError("Energy not found in OUTCAR")
        
        # Parsing the OUTCAR to get Fermi energy.
        try:
            e_fermi_pattern = re.compile(r'E-fermi\s+:\s+(\-\d+\.\d+)')
            e_fermi = re.findall(e_fermi_pattern, outcar)[-1]
            e_fermi = float(e_fermi)
        except:
            raise ParseError("Fermi energy not found in OUTCAR")
        
        with open("vasp.out", "r") as f:
            vaspout = f.readlines()
        vaspout = "".join(vaspout)

        # Parsing vasp.out to get Fermi Shift.
        try:
            fermi_shift_pattern = re.compile(r"FERMI_SHIFT =\s+([\d.E+-]+)")
            fermi_shift_array = re.findall(fermi_shift_pattern, vaspout)
            fermi_shift_array = [float(i) for i in fermi_shift_array if float(i)!=0]
            fermi_shift = fermi_shift_array[-1]
        except:
            raise ParseError("Fermi shift not found in vasp.out")
        
        return e, e_fermi, fermi_shift
    
    def plot_parabola(self,PZC_nelect):
        """Generates the energy vs potential plot.

        :param PZC_nelect: Number of electrons at PZC
        :type PZC_nelect: float
        """
        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False
        subdirs = [f.path for f in os.scandir("./") if f.is_dir()]
        subdirs = map(lambda x:re.sub('./', '', x), subdirs)
        subdirs = [subdir for subdir in subdirs if isfloat(subdir)]

        n_elects, es, e_fermis, fermi_shifts = np.array([]), np.array([]), np.array([]), np.array([])
        for dir in subdirs:
            os.chdir(f"./{dir}")
            e, e_fermi, fermi_shift = self.parse_output()
            n_elects = np.append(n_elects, float(dir))
            es = np.append(es, e)
            e_fermis = np.append(e_fermis, e_fermi)
            fermi_shifts = np.append(fermi_shifts, fermi_shift)
            os.chdir("../")
        
        vacpot = 4.44
        work_functions = -e_fermis - fermi_shifts
        SHEpots = work_functions - vacpot
        es  = es + fermi_shifts*(n_elects - PZC_nelect)
        gs = es + work_functions*(n_elects - PZC_nelect)
        Lipots = np.array([SHEpot+3.04 for SHEpot in SHEpots])
        f = open("data.txt","w")
        for n_e,V,G in zip(n_elects,Lipots,gs):
            f.write("NELECT = " + str(n_e) + "\n")
            f.write("PotvsLi+/Li = "+ str(V) + "\n")
            f.write("G = " + str(G) + "\n")
            f.write("\n")
        f.close()

        quadratic = lambda x, a, b, c: a * x ** 2 + b * x + c
        parameter, _ = curve_fit(quadratic, Lipots, gs)
        with open("fit.txt", "w") as f:
            for i in parameter:
                f.write(f"{i}\n")
        f.close()
        fitted = lambda x: parameter[0] * x ** 2 + parameter[1] * x + parameter[2]

        fig = plt.figure(dpi = 200, figsize=(6,5))
        x = np.linspace(Lipots.min()-0.2, Lipots.max()+0.2, 100)
        y = list(map(fitted, x))
        plt.title('a={}\nb={}\nc={}'.format(*parameter))
        plt.scatter(Lipots, gs, label='original data', color='indigo')
        plt.plot(x, y, label='fitted', color='indigo')
        get_plot_settings(fig,"U vs Li+/Li (V)","G (eV)","g-pot.png","upper right")
        
        return fig
    
    def analysis(self):
        """Generates the energy vs potential plot.
        """
        pwd = os.getcwd()
        PZC_nelect = self.get_PZC_nelect()
        try:
            os.rename("PZC_calc", f"{PZC_nelect}")
            self.plot_parabola(PZC_nelect)
            os.rename(f"{PZC_nelect}", "PZC_calc")
        except Exception as error:
            os.chdir(pwd)
            os.rename(f"{PZC_nelect}", "PZC_calc")
            raise error

class gibbs_free_energy:
    """Gives the gibbs free energy of the system. If surface_charging is used, the parabola fit is used to obtain the energy vs potential. If geo_opt is used, OUTCAR is used to obtain energy. The vibrational energy is obtained using the frequency class.  Note: Only works if ASE is used to run the frequency calculation.
    """
    def __init__(self, calc_root_dir):
        """Initializes the gibbs_free_energy class.

        :param calc_root_dir: Root directory of the calculation
        :type calc_root_dir: string
        """
        self.calc_root_dir = calc_root_dir

    def get_parabola(self):
        """Fits a parabola to the energy vs potential plot using analysis.py.

        :return: Coefficients of the parabola
        :rtype: list
        """
        calc_root_dir = self.calc_root_dir
        pwd = os.getcwd()
        os.chdir(f"{calc_root_dir}")
        if os.path.exists("fit.txt"):
            pass
        else:
            assert os.path.exists("analysis.py"), "analysis.py not found in the root directory."
            subprocess.run(["python", "analysis.py"])
            assert os.path.exists("fit.txt"), "fit.txt not found in the root directory, check your surface charging calculation for completion."
        with open("fit.txt", "r") as f:
            lines = f.readlines()
        lines = np.array([float(line) for line in lines])
        os.chdir(pwd)
        return lines

    def get_energy(self, potential=None, outcar_location="./"):
        """Gives the energy of the system.

        :param potential: Potential at which energy is to be calculated, defaults to None
        :type potential: float, optional
        :param outcar_location: Location of OUTCAR in the root directory, defaults to "./"
        :type outcar_location: str, optional
        :return: Energy of the system
        :rtype: float
        """
        calc_root_dir = self.calc_root_dir
        pwd = os.getcwd()
        os.chdir(f"{calc_root_dir}")
        try:
            if potential is None:
                os.chdir(f"{outcar_location}")
                assert os.path.exists("OUTCAR"), "OUTCAR not found in the given location."
                atoms = read("OUTCAR")
                return atoms.get_potential_energy()
            else:
                fit_param = self.get_parabola()
                return fit_param[0]*potential**2 + fit_param[1]*potential + fit_param[2]
        finally:
            os.chdir(pwd)

    def get_vib_energy(self, temperature, pressure=None):
        """Obtain the vibrational energy of the system using the frequency calculation.

        :param temperature: Temperature at which the vibrational energy is to be calculated
        :type temperature: float
        :param pressure: Pressure at which the vibrational energy is to be calculated, pressure must be given if `mode="IdealGas"` in analysis method of the frequency class, defaults to None
        :type pressure: float, optional
        :return: Vibrational energy of the system
        :rtype: float
        """
        calc_root_dir = self.calc_root_dir
        pwd = os.getcwd()
        os.chdir(f"{calc_root_dir}/frequency")
        # Caching to prevent frequency analysis when Gibbs.txt for a specific temperature and pressure already exists.
        if os.path.exists("parameters.txt"):
            with open("parameters.txt", "r") as f:
                lines = f.readlines()
                old_temperature = float(lines[0])
                try:
                    old_pressure = float(lines[1])
                except ValueError or TypeError or IndexError:
                    old_pressure = None
        else:
            old_temperature = None
            old_pressure = None
        if os.path.exists("Gibbs.txt") and temperature==old_temperature and pressure==old_pressure:
            pass
        else:
            assert os.path.exists("analysis.py"), "analysis.py not found in the frequency directory."
            if pressure is None:
                subprocess.run(["python", "analysis.py", "--temperature", f"{temperature}"], check=True)
            elif pressure is not None:
                subprocess.run(["python", "analysis.py", "--temperature", f"{temperature}", "--pressure", f"{pressure}"], check=True)
            assert os.path.exists("Gibbs.txt"), "Gibbs.txt not found in the frequency directory, check your frequency calculation for completion."
        with open("Gibbs.txt", "r") as f:
            G = float(f.read())
        with open("parameters.txt", "w") as f:
            f.write(f"{temperature}\n")
            f.write(f"{pressure}")
        os.chdir(pwd)
        return G
    
    def get_gibbs_free_energy(self, temperature, pressure=None, potential=None, outcar_location="./"):
        """Gives the gibbs free energy of the system.

        :param temperature: Temperature at which the gibbs free energy is to be calculated
        :type temperature: float
        :param pressure: Pressure at which the vibrational energy is to be calculated, pressure must be given if `mode="IdealGas"` in analysis method of the frequency class, defaults to None
        :type pressure: float, optional
        :param potential: Potential at which energy is to be calculated, defaults to None
        :type potential: float, optional
        :param outcar_location: Location of OUTCAR in the root directory, defaults to "./"
        :type outcar_location: string, optional
        :return: Gibbs free energy of the system
        :rtype: float
        """
        return self.get_energy(potential=potential, outcar_location=outcar_location)+ self.get_vib_energy(temperature, pressure=pressure)

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

def analyse_GCBH(save_data=True, energy_operation=None, label=None):
    """Performs a visual analysis of the results from Grand Canonical Basin Hopping simulation performed using catalapp.

    :param save_data: Reads and saves data from the opt_folder, defaults to True
    :type save_data: bool, optional
    :param energy_operation: Function which operates on energy data from opt_folder, example: `def energy_operation(e):return (e+2)` where e is the energy read from opt_folder, defaults to None
    :type energy_operation: function, optional
    :param label: Label of the plots, defaults to None
    :type label: string, optional
    """
    if save_data is not False:
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
            if energy_operation is None:
                E.append(e)
                f.write(f"{e}\n")
            elif energy_operation is not None:
                E.append(energy_operation(e))
                f.write(f"{energy_operation(e)}\n")
            traj.write(atoms)
            os.chdir("../")
        os.chdir("../")
    elif save_data is False:
        E = []
        with open("energies.txt", "r") as f:
            data = f.readlines()
            for i in data:
                    E.append(float(i))

    assert E!=None, "No data found. Set save_data=True to save data."
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
    """Provides the neigbor list for the system. Output is provided in Neighbor_List.txt file. Neighbors of each atom, their positions and coordination numbers of each atom are provided based on ASE's natural cutoff distances.

    :param atoms: Atoms object for which the neighbor list is to be obtained
    :type atoms: Atoms object
    """
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

def check_run_completion(location):
    """Checks for completion of a VASP job at the provided location.

    :param location: Location of the VASP job
    :type location: string
    :return: True if the job is completed, False if the job is not completed
    :rtype: bool
    """
    cwd = os.getcwd()
    os.chdir(location)
    with open("OUTCAR", "r") as f:
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
    """Provides information about the volume, vector lengths and angles of the unit cell.

    :param atoms: Atoms object for which the cell information is to be obtained
    :type atoms: Atoms object
    """
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

class benchmark:
    """Performs computational benchmark of a VASP job.
    """
    def __init__(self, cores):
        """Initializes the benchmark class.

        :param cores: List of cores to be used for the benchmark
        :type cores: list
        """
        self.cores = cores

    def submit_jobs(self):
        """Submits the VASP jobs for the benchmark.
        """
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
        """Obtains the benchmark results.

        :param outcar_location: Location of the OUTCAR for the calculation, defaults to "./"
        :type outcar_location: str, optional
        """
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