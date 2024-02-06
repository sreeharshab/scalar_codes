from silicon import slide_sigma3_gb
from numpy import pi
from ase.io import read

if __name__=='__main__':
    atoms = read("POSCAR")
    n = 11
    n_steps = 20
    disp = (3.8663853050000001/5)
    theta = (0*pi)/180
    system = slide_sigma3_gb(n_steps)

    # Running simulation
    system.run_parallel(atoms, disp, theta, scheme="step", restart=False)
    # # If scheme="linear"
    # system.run_parallel(atoms, opt_levels, disp, theta, scheme="linear", n=n, restart=False)
    # # If restart=True and scheme="linear"
    # system.run_parallel(atoms, opt_levels, disp, theta, scheme="linear", n=n, restart=True, largest_level=3)
    # # largest_level is the largest level in geo_opt.py.
