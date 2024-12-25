from ase.io import read, write
from pipelines import frequency

if __name__=="__main__":
    atoms = read("POSCAR")
    vib_indices = [iii]
    addnl_settings = {
        "setups": {'Li': '_sv'},
        "nelm": 300,
        "encut": 400,
        "algo": "Normal",
        "prec": "Accurate",
        "amin": 0.01,
        "lsol": True,
        "eb_k": 5.0,
        "lambda_d_k": 2.3,
        "nc_k": 0.001,
        "ncore": 4,
    }
    freq = frequency(atoms, vib_indices=vib_indices)
    freq.run(kpts=[5,5,1], mode="ase", scheme="serial", addnl_settings=addnl_settings)