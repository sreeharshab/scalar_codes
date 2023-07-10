#!/usr/bin/env python

""" more like a usage demo / template than unit testing
todo: try this in pytest
"""

import os
from ase.io import read, write


@pytest.fixture
def struct_dir():
    struct_dir = os.getenv("SIGB_TESTS_DIR")
    assert struct_dir is not None, "set SIGB_TEST_DIR before running tests"


def test_cell_opt(struct_dir):
    from pipelines import cell_opt

    atoms = read(f"{struct_dir}/cell_opt.vasp")
    atoms.info["kpts"] = [7, 7, 7]
    cell_opt(atoms)


def test_axis_opt():
    pass


def test_geo_opt(struct_dir):
    from pipelines import geo_opt

    atoms = read(f"{struct_dir}/geo_opt.vasp")
    atoms.info["kpts"] = [1, 7, 5]
    opted = geo_opt(atoms)


def test_freq(struct_dir):
    from pipelines import freq

    atoms = read(f"{struct_dir}/freq.vasp")
    atoms.info["kpts"] = [1, 7, 5]
    freq(atoms, mode="ase")


def test_COHP(struct_dir):
    from pipelines import COHP

    atoms = read(f"{struct_dir}/cohp.vasp")
    cohp = COHP(atoms, bonds=bonds)
    cohp.run_vasp()
    cohp.write_lobsterin()
    cohp.run_lobster()
    cohp.plot()


def test_NEB(struct_dir):
    from pipelines import NEB

    initial = read(f"{struct_dir}/neb_initial.vasp")
    initial.info["kpts"] = [1, 7, 5]
    final = read(f"{struct_dir}/neb_final.vasp")
    final.info["kpts"] = [1, 7, 5]
    neb = NEB(initial, final)
    neb.interpolate(method="optnpath", nimage=8)
    neb.write_input(backend="vtst")


def test_Dimer():
    pass
