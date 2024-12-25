from pipelines import DOS
from ase.io import read

if __name__=="__main__":
    atoms = read("POSCAR")
    dos = DOS()
    addnl_settings = {"ncore": 8, "encut": 600, "ismear": 0, "sigma": 0.03, "ispin": 2, "lvdw": True, "ivdw": 12, "ldau": True, "ldautype": 2, "ldaul": [2,-1], "ldauu": [3.5,0], "lmaxmix": 4, "prec": "High"}
    dos.run(atoms, kpts=[4,12,6], addnl_settings=addnl_settings)
    dos.parse_doscar()
    d_dos_up, d_dos_down = dos.get_orbital_projected_dos("d")
    print(dos.get_band_gap())
    print(dos.get_band_center(dos_up=d_dos_up, dos_down=d_dos_down))
    dos.plot(d_dos_up, label="d DOS up", fig_name="d_DOS_up.png")
    dos.plot(d_dos_down, label="d DOS down", fig_name="d_DOS_down.png")