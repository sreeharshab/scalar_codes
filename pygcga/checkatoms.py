#!/bin/env python
from ase.data import covalent_radii
from ase.data import chemical_symbols
import networkx as nx
import numpy as np
from ase.constraints import FixAtoms
import logging


class CheckAtoms(object):
    def __init__(self, max_bond=1.5, min_bond=0.7,
                 verbosity=False,
                 check_close_contact=True,
                 check_connectivity=True,
                 bond_range=None,
                 mic=True):
        """
        :param max_bond:
        :param min_bond:
        :param verbosity:
        :param check_close_contact:
        :param check_connectivity:
        :param bond_range: dict, bond distances range like {('H','Pt'):[1.5,8.0],('Pt','Pt'):[2.0,10.0]}
        """
        if bond_range is None:
            self.radius = dict(zip(chemical_symbols, covalent_radii))
            self.bmin = min_bond
            self.bmax = max_bond
            self.brange = None
        else:
            self.brange = {}
            for key, value in bond_range.items():
                self.brange[key] = sorted(value)

        self.G = None
        self.mic = mic
        self.results = {}
        self.verbosity = verbosity
        self.check_close_contact = check_close_contact
        self.check_connectivity = check_connectivity

    def _build_graph(self, atoms):
        _d = atoms.get_all_distances(mic=self.mic)
        symbols = atoms.get_chemical_symbols()
        n = atoms.get_number_of_atoms()
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))
        self.results['close_contact'] = False
        for i in range(n):
            for j in range(i + 1, n):
                if self.brange is None:
                    d_equlibrium = self.radius[symbols[i]] + self.radius[symbols[j]]
                    if _d[i][j] < self.bmax * d_equlibrium:
                        self.G.add_edge(i, j)
                    if _d[i][j] < self.bmin * d_equlibrium:
                        msg = "%s(%d) and %s(%d) d=%.3f: too close" % (symbols[i], i, symbols[j], j, _d[i][j])
                        logging.debug(msg)
                        if self.verbosity:
                            print(msg)
                        self.results['close_contact'] = True
                else:
                    ekey1 = (symbols[i], symbols[j])
                    ekey2 = (symbols[j], symbols[i])
                    if ekey1 not in self.brange.keys() and ekey2 not in self.brange.keys():
                        raise RuntimeError("I did not find the bond distance range for element %s %s" % ekey1)
                    else:
                        if ekey1 in self.brange.keys():
                            rmin, rmax = self.brange.get(ekey1)
                        else:
                            rmin, rmax = self.brange.get(ekey2)
                        if _d[i][j] < rmax:
                            self.G.add_edge(i, j)
                        if _d[i][j] < rmin:
                            msg = "%s(%d) and %s(%d) d=%.3f: too close" % (symbols[i], i, symbols[j], j, _d[i][j])
                            logging.debug(msg)
                            if self.verbosity:
                                print(msg)
                            self.results['close_contact'] = True

    def __update__(self, atoms):
        self.G = None
        self.results = {}
        self._build_graph(atoms)
        self.results['connected'] = nx.is_connected(self.G)
        logging.debug('structure connectivity is {}'.format(self.results['connected']))

        good_structure = True
        if self.check_close_contact and self.results['close_contact']:
            good_structure = False
        if self.check_connectivity and (not self.results['connected']):
            good_structure = False
        self.results['good'] = good_structure

        _subgraphs = list(nx.connected_component_subgraphs(self.G))
        _subgraphs.sort(key=lambda x: len(x))
        images = []
        indexes_group = []
        _a = atoms.copy()
        for _g in _subgraphs:
            _nodes = _g.nodes()
            try:
                _index = sorted(_nodes.keys())
            except AttributeError:
                _index = sorted(list(_nodes))
            else:
                indexes_group.append(_index)
                images.append(_a[_index])
        self.results['images'] = images
        self.results['indexes_group'] = indexes_group
        # Check the consistency of the sub groups.
        assert len(_a) == sum([len(x) for x in images])
        if self.results['connected']:
            assert len(self.results['indexes_group']) == 1
        else:
            assert len(self.results['indexes_group']) > 1

    def is_connected(self, atoms):
        self.__update__(atoms)
        return self.results['connected']

    def is_close_contact(self, atoms):
        self.__update__(atoms)
        return self.results['close_contact']

    def is_good(self, atoms, return_components=None, quickanswer=True):
        """
         :param atoms: ase.atoms.Atoms object
         :param return_components: return two key:value information "images" and "indexes_group"
         :param quickanswer: just answer True or False
         :return:
         """
        self.__update__(atoms)
        if return_components:
            quickanswer = False
        if quickanswer:
            return self.results['good']
        else:
            return self.results

    # def connecting_atoms(self,atoms,check_constraints=True):
    #     self.__update__(atoms)
    #     fixed_indices=[]
    #     if check_constraints:
    #         for c in atoms.constraints:
    #             if isinstance(c,FixAtoms):
    #                 fixed_indices.extend(c.get_indices())
    #     # check if all the fixed atoms are in the biggest group:
    #     largest_subgroup=self.results['indexes_group'][-1]
    #     all_included=True
    #     for index in fixed_indices:
    #         if index not in largest_subgroup:
    #             all_included=False
    #     if not all_included:
    #         raise RuntimeError("some fixed atoms are not in the largest sub group, this is unexpected\n")
    #
    #     transformed_atoms=atoms.copy()
    #     for gi in self.results['indexes_group']:
    #         index_i,index_max =self.__find_most_close_pair(atoms=atoms,indices=[gi,largest_subgroup],mic=mic)
    #         # grouping atoms in gi
    #         dx=self.__group_atoms_by_pbc(atoms=atoms,ref_index=index_i,indices=gi)
    #         transformed_atoms.set_positions(transformed_atoms.get_positons()+dx)
    #         # vector from index_i to index_max
    #         vector=transformed_atoms.get_distance(index_max,index_i,vector=True,mic=)
    #
    #
    # def __find_most_close_pair(self,atoms=None,indices=None):
    #     assert len(indices) == 2
    #     index_a=None
    #     index_b=None
    #     distance=1.0e8
    #     for ia in indices[0]:
    #         for ib in indices[1]:
    #             if ia == ib:
    #                 raise RuntimeError("same index in two groups?, this is not expected")
    #             d=atoms.get_distance(ia,ib,mic=self.mic)
    #             if d < distance:
    #                 distance=d
    #                 index_a=ia
    #                 index_b=ib
    #     assert index_a is not None
    #     assert index_b is not None
    #     return index_a, index_b
    #
    # def __group_atoms_by_pbc(self,atoms=None,ref_index=None,indices=None):
    #     """
    #     Sometimes, the sub atoms group (in indices) are close with each other across a periodic transformation,
    #     This function will return an delta X (natoms,3), which can put them together, regardless of boundary
    #     """
    #     delta_x=np.zeros((atoms.get_number_of_atoms(),3))
    #     if not atoms.get_pbc().any():
    #         sys.stderr.write("WARNNING: grouping atoms is not useful for non-periodic system\n")
    #         return delta_x
    #     assert ref_index in indices
    #     for index in indices:
    #         if index == ref_index:
    #             continue
    #         else:
    #             v1=atoms.get_distance(ref_index,index,vector=True,mic=False)
    #             v2=atoms.get_distance(ref_index,index,vector=True,mic=True)
    #             # we want to move atom index from v1 to v2
    #             delta_x[index]=v2-v1
    #     return delta_x

    # def connecting_atoms(self,atoms):
    #     raise RuntimeError("need to be rewritten")
    #     connected=self.check_connected(atoms)
    #     if connected:
    #         return atoms
    #     else:
    #         symbols=atoms.get_chemical_symbols()
    #         lgroup=results['group'][0]
    #         distances=atoms.get_all_distances()
    #         positions=atoms.get_positions()
    #         for gid in range(1,len(results['group'])):
    #             group=results['group'][gid]
    #             pairmin=1.0e15
    #             pair=[-1,-1]
    #             for ida in lgroup:
    #                 for idb in group:
    #                     assert ida != idb
    #                     if distances[ida][idb] < pairmin:
    #                         pairmin=distances[ida][idb]
    #                         pair=[ida,idb]
    #             #the displacement vector
    #             dv=positions[pair[0]]-positions[pair[1]]
    #             dequlibrium=self.radius[symbols[ida]]+self.radius[symbols[idb]]
    #             dv=dv*(pairmin-dequlibrium)/pairmin
    #             for idb in group:
    #                 positions[idb]=positions[idb]+dv
    #         atoms=atoms.copy()
    #         atoms.set_positions(positions)
    #         return atoms
