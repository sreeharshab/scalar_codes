There are two parts to this repository.

## Part 1: Grand Canonical Basin Hopping
Dr. Geng Sun, a former postdoc in Prof. Philippe Sautet's lab, developed gcbh.py. This performs a global optimization on the system in consideration based on the input modifiers.

## Part 2: Codes to Study Grain Boundaries of Silicon
All of the code is available in pipelines.py. The following are the available features:
1. cell_opt: Optimizes the size of the simulation cell.
2. axis_opt: Optimizes the size of the required axis of the simulation cell.
3. geo_opt: Performs geometry optimization on the system using inbuilt VASP optimizers (using the IBRION tag) or ASE optimizers.
4. freq: Performs vibrational analysis on the system using VASP or ASE (parallelization is good using ASE).
5. bader: Performs bader charge analysis on the system. You can view the charges in ACF.dat file or using ase gui and choosing the Initial Charges label in the view tab.
6. COHP: Performs COHP analysis on the system. Presently only suitable for systems containing only Si, O, Al and Li. Feel free to understand and edit the code according to your needs. The output is saved as cohp-1.png.
7. NEB: Performs Nudged Elastic Band calculation to obtain transition state between initial and final images. Intermediate images can be generated using either linear interpolation or Opt'n Path program. NEB can be run using ASE or VTST scripts.
8. pbc_correction: Corrects atom's coordinates if they are out of the simulation cell (wrapping operation).
9. create_sigma3_gb: Creates a Σ3 grain boundary with n layers using top_grain.vasp and bottom_grain.vasp files.
10. slide_sigma3_gb: Slides Σ3 grain boundary. Serial and parallel runs are implemented. In each run, step and linear schemes are implemented. Note that step scheme is effective for studying the stick-slip sliding behavior and linear scheme is effective for studying elastic deformation. Restart option is unavailable for parallel runs.
11. intercalate_Li: Inserts Li in all the interstice positions of Σ3 grain boundary.
12. symmetrize_sigma3_gb: Symmetrizes Σ3 grain boundary strucutre along y axis. This is necessary to perform surface charging calculations.
13. cure_Si_surface_with_H: In order to study Si surfaces, it is essestial to create a bulk like environment far from the surface. To create this environment, we need to add H to one side of our surface model (with vacuum) to cure the dangling bonds. This code inserts H for (100) Si surface.
14. dos: Performs a DOS calculation. The code for visualizing DOS will be implemented soon.
15. get_neighbor_list: Provides the neigbor list for the system. Output is provided in Neighbor_List.txt file. Neighbors of each atom, their positions and coordination numbers of each atom are provided based on ASE's natural cutoff distances.
16. check_run_completion: Checks for completion of a VASP job at the provided location.