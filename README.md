# Installation
Clone this repo using the following commands:
```
cd /path_to_home_directory/
git clone https://github.com/sreeharshab/scalar_codes.git
```
After successful cloning of the repo, open your `.bashrc` and insert this line:
```
export PYTHONPATH=/path_to_home_directory/scalar_codes:$PYTHONPATH
```
Exit your `.bashrc` and source it as follows:
```
source .bashrc
```
Note: Ensure that you have [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/), [Numpy](https://numpy.org/), [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/) installed in your machine!

# Resources
There are three parts to this repository. Part 1 and 2 are available in pipelines.py. Part 3 is available in gcbh.py.

Use `help()` to get detailed documentation about the resources. Furthermore, there are examples some of the resources in the examples folder.

Note: Presently, documentation is available only for select resources. Full documentation will be available soon.

## Part 1: Automation Codes for VASP Related Calculations
All of the code is available in pipelines.py. The following are the available features:
1. cell_opt: Optimizes the size of the simulation cell.
2. axis_opt: Optimizes the size of the required axis of the simulation cell.
3. geo_opt: Performs geometry optimization on the system using inbuilt VASP optimizers (using the IBRION tag) or ASE optimizers.
4. bader: Performs bader charge analysis on the system. Charges can be viewed in ACF.dat file or using ase gui and choosing the Initial Charges label in the view tab.
5. COHP: Performs COHP analysis on the system. The output is saved as cohp-1.png.
6. NEB: Performs Nudged Elastic Band calculation to obtain transition state between initial and final images. Intermediate images can be generated using either linear interpolation or [Opt'n Path](http://forge.cbp.ens-lyon.fr/redmine/projects/optnpath/wiki) program. NEB can be run using [ASE](https://wiki.fysik.dtu.dk/ase/) or [VTST](https://theory.cm.utexas.edu/vtsttools/) scripts.
7. frequency: Performs vibrational analysis on the system using VASP or ASE. Use ASE for calculations involving large systems as it supports a parallel scheme.
8. surface_charging: Performs surface charging calculation using [VASPsol](https://github.com/henniggroup/VASPsol).
9. dos: Performs a DOS calculation. The code for visualizing DOS will be implemented soon.
10. analyse_GCBH: Performs a visual analysis of the results from Grand Canonical Basin Hopping simulation performed using gcbh.py.
11. get_neighbor_list: Provides the neigbor list for the system. Output is provided in Neighbor_List.txt file. Neighbors of each atom, their positions and coordination numbers of each atom are provided based on ASE's natural cutoff distances.
12. check_run_completion: Checks for completion of a VASP job at the provided location.
13. get_cell_info: Provides information about the volume, vector lengths and angles of the unit cell.
14. get_selective_dynamics: Checks whether selective dynamics is true or false for the atom corresponding to the provided index.
15. benchmark: Performs computational benchmark of a VASP job.

## Part 2: Codes to Study Silicon Systems
16. create_sigma3_gb: Creates a Σ3 grain boundary with n layers using top_grain.vasp and bottom_grain.vasp files (available in the examples folder).
17. slide_sigma3_gb: Slides Σ3 grain boundary. Serial and parallel runs are implemented. In each run, step and linear schemes are implemented. Note that step scheme is effective for studying the stick-slip sliding behavior and linear scheme is effective for studying elastic deformation.
18. intercalate_Li: Inserts Li in all the interstice positions of Σ3 grain boundary.
19. symmetrize_Si100_surface: Symmetrizes Si (100) strcuture along z axis. This is necessary to perform surface charging calculations using [VASPsol](https://github.com/henniggroup/VASPsol).
20. cure_Si_surface_with_H: In order to study Si surfaces, it is essestial to create a bulk like environment far from the surface. To create this environment, we need to add hydrogens to our surface model (with vacuum) to cure the dangling bonds. This code inserts works best for (100) Si surface where it insertes two hydrogens if the coordination of silicon is less than four.

## Part 3: Grand Canonical Basin Hopping
Dr. Geng Sun, a former postdoc in Prof. Philippe Sautet's lab, developed gcbh.py. This performs a global optimization on the system in consideration based on the input modifiers.

<!-- # Contributing
Contributions to improve this repo are always welcome. Any contribution should be  -->