from ase.io import read
from pipelines import surface_charging
from silicon import symmetrize_Si100_surface

if __name__=="__main__":
    atoms = read("POSCAR")

    opt_levels = {
        1: {"kpts": [5,5,1]},
        2: {"kpts": [9,9,1]},
    }
    for j in range(1,3,1):
        opt_levels[j]["setups"] = {'Li': '_sv'}
        opt_levels[j]["encut"] = 400
        opt_levels[j]["nsw"] = 2000
        opt_levels[j]["nelm"] = 200
        opt_levels[j]["algo"] = "Normal"
        opt_levels[j]["prec"] = "Accurate"
        opt_levels[j]["amin"] = 0.01
        opt_levels[j]["ediffg"] = -0.02
        opt_levels[j]["lsol"] = True
        opt_levels[j]["eb_k"] = 5.0
        opt_levels[j]["lambda_d_k"] = 2.3
        opt_levels[j]["nc_k"] = 0.001
        opt_levels[j]["ncore"] = 8
    
    n_nelect=4
    width_nelect=0.25
    
    sc = surface_charging()
    sc.run(atoms, opt_levels, n_nelect=n_nelect, width_nelect=width_nelect)