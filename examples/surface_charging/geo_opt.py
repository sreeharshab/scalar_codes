import ase
from ase.io import read
from pipelines import geo_opt

if __name__=="__main__":
    atoms = read('POSCAR')

    opt_levels = {
        1: {"kpts": [9,9,1]},
    }
    max_level = len(opt_levels)
    for j in range(1,2,1):
        opt_levels[j]["setups"] = {'Li': '_sv'}
        opt_levels[j]["encut"] = 400
        opt_levels[j]["ibrion"] = -1
        opt_levels[j]["nsw"] = 0
        opt_levels[j]["nelm"] = 200
        opt_levels[j]["algo"] = "Normal"
        opt_levels[j]["prec"] = "Accurate"
        opt_levels[j]["amin"] = 0.01
        opt_levels[j]["ediffg"] = -0.02
        opt_levels[j]["lsol"] = True
        opt_levels[j]["nelect"] = nnn   # Do not remove/change this line!
        opt_levels[j]["eb_k"] = 5.0
        opt_levels[j]["lambda_d_k"] = 2.3
        opt_levels[j]["nc_k"] = 0.001
        opt_levels[j]["ncore"] = 8

    geo_opt(atoms,mode='vasp',opt_levels=opt_levels, restart=True)