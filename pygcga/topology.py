import numpy as np
from ase.data import covalent_radii
# from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list
import networkx as nx
from ase.data import atomic_numbers


class Topology(object):
    """
    This class is used to maintain and help to easily access the coordination numbers of one system
    """

    def __init__(self, atoms, ratio=None, radius=None):
        """"Two atoms are bonded when their distance smaller than 1.3 * (r_i+r_j)"""
        self.atoms = atoms
        symbols = self.atoms.get_chemical_symbols()

        if ratio is None and radius is None:
            ratio = 1.3

        if ratio is not None and radius is not None:
            raise RuntimeError("you can use either ratio or radius, not both of them")

        if ratio is not None and radius is None:
            cutoffs = [covalent_radii[i] for i in self.atoms.get_atomic_numbers()]
            cutoffs = np.array(cutoffs) * ratio
        else:
            assert isinstance(radius, dict)
            t = {}
            for k, v in radius.items():
                if isinstance(k, str):
                    t[atomic_numbers[k]] = v
            try:
                cutoffs = [t[i] for i in self.atoms.get_atomic_numbers()]
            except KeyError:
                raise RuntimeError("atomic radius for some elements does not exist")
        self.cutoffs = cutoffs
        # #self.nl=NeighborList(cutoffs,skin=0.01,sorted=True,self_interaction=False)
        # self.nl=NeighborList(cutoffs,skin=0.01,self_interaction=False,bothways=True)
        # self.nl.update(self.atoms)

        self._internal_memory = {}
        for index in range(self.atoms.get_number_of_atoms()):
            self._internal_memory[index] = {'neighbors_symbols': [], 'neighbors_indexes': [], 'neighbors_offset': [],
                                            'neighbors_distances': [], 'neighbors_vectors': []}

        index_i, index_j, distances_d, vectors_D, shift_S = neighbor_list('ijdDS', self.atoms, cutoff=self.cutoffs,
                                                                          self_interaction=False)
        for i, j, d, D, S in zip(index_i, index_j, distances_d, vectors_D, shift_S):
            neighbor_details = self._internal_memory[i]
            neighbor_details['neighbors_symbols'].append(symbols[j])
            neighbor_details['neighbors_indexes'].append(j)
            neighbor_details['neighbors_offset'].append(S)
            neighbor_details['neighbors_distances'].append(d)
            neighbor_details['neighbors_vectors'].append(D)

    # def __create_neighbor_information(self, index):
    #     neighbor_details = {'neighbors_symbols': [], 'neighbors_indexes': [], 'neighbors_offset': [],
    #                         'neighbors_distances':[], 'neighbors_vectors':[]}
    #     symbols = self.atoms.get_chemical_symbols()
    #     # indices, offsets = self.nl.get_neighbors(index)
    #     # index_i, index_j, distances_d, vectors_D, shift_S = neighbor_list('ijdDS', self.atoms,
    #     #                                                                   cutoff=self.cutoffs, self_interaction=False)
    #     #debug
    #     print index_i, len(index_i), len(self.atoms)
    #     print len(index_j)
    #     for j, d, D, S in zip(index_j, distances_d, vectors_D, shift_S):
    #         neighbor_details['neighbors_symbols'].append(symbols[j])
    #         neighbor_details['neighbors_indexes'].append(j)
    #         neighbor_details['neighbors_offset'].append(S)
    #         neighbor_details['neighbors_distances'].append(d)
    #         neighbor_details['neighbors_vectors'].append(D)
    #     self._internal_memory[index] = neighbor_details

    def get_coordination_number(self, index, neighbor_element=None):
        """
        :param index: index for the central atom
        :param neighbor_element: str or list, the symbols of neighbouring atoms.
        :return: int
        """
        # if index not in self._internal_memory.keys():
        #     self.__create_neighbor_information(index)
        if neighbor_element is None:
            return len(self._internal_memory[index]['neighbors_symbols'])
        else:
            if isinstance(neighbor_element, str):
                neighbor_element = [neighbor_element]
            stat_i = self._internal_memory[index]['neighbors_symbols']
            return sum([stat_i.count(s) for s in neighbor_element])

    def get_neighboring_element(self, index, neighbor_element=None):
        """
        :param index: which atom is considered ?
        :param neighbor_element: which kind of elements are confided as neighbours
        :return: [(index_j, distance_j, vector_ij)]
        """
        result = []
        if neighbor_element is None:
            t = list(set(self.atoms.get_chemical_symbols()))
        else:
            t = [neighbor_element]

        for s, i, d, v in zip(self._internal_memory[index]['neighbors_symbols'],
                              self._internal_memory[index]['neighbors_indexes'],
                              self._internal_memory[index]['neighbors_distances'],
                              self._internal_memory[index]['neighbors_vectors']):
            if s in t:
                result.append((i, d, v))
        return result

    def get_coordination_details(self, index):
        """
        :param index: index for the central atom
        :return: dict
        dict has several different keys:
        neighbors_symbols: a list of neighboring atoms chemical sybmols
        neighbors_indexes: a list of neighboring atom indexes
        neighbors_distances: distances
        neighbors_vectors: distance vectors
        neighbors_offset: offset of neighbouring atoms vector will be a.positions[j]-a.positions[i]+offset.dot(a.cell)
        """
        # if index not in self._internal_memory.keys():
        #     self.__create_neighbor_information(index)
        if index not in self._internal_memory:
            for _k, _v in self._internal_memory.items():
                print("{} = {}".format(_k, _v))
            raise KeyError("{} not in self._internal_memory".format(index))
        return self._internal_memory[index]


class Topology2(object):
    """
    This class is used to maintain and help to easily access the coordination numbers of one system;

    Two atoms have chemical bond, when and only when:
    1) their distance  d <  2.0 * (r_i +r_j)
    and 2) No third atom located within the sphere radius (d/2.0) and centered at midpoint of atom i atom j
    """

    def __init__(self, atoms, pbc=None, cutoff=1.3):
        """"Two atoms are bonded when their distance smaller than 1.3 * (r_i+r_j)"""
        self.atoms = atoms
        self.pbc = pbc
        self.cutoff = cutoff

        self.cG = self._search_bond(self.atoms)

    def _search_bond(self, atoms=None):
        cG = nx.Graph()
        if atoms is None:
            a = self.atoms.copy()
        else:
            a = atoms.copy()

        if self.pbc is None:
            pbc = a.get_pbc().any()
        else:
            pbc = self.pbc

        if pbc:
            raise RuntimeError("Currently, only isolated clusters are supported")

        _DM = a.get_all_distances(mic=pbc)

        positions = a.get_positions()
        natoms = a.get_number_of_atoms()
        _an = a.get_atomic_numbers()
        symbols = a.get_chemical_symbols()

        cG.add_nodes_from([(p, dict(symbol=q)) for p, q in zip(range(natoms), symbols)])
        for ii in range(natoms):
            for jj in range(ii + 1, natoms):
                has_bond = True
                if _DM[ii][jj] > self.cutoff * (covalent_radii[_an[ii]] + covalent_radii[_an[jj]]):
                    has_bond = False
                else:
                    # midpoint=0.5*(positions[ii]+positions[jj])
                    for kk in range(natoms):
                        if kk == ii or kk == jj:
                            continue
                        vector1 = positions[jj] - positions[ii]
                        vector2 = positions[kk] - positions[ii]
                        # I would like make sure the project of v2 to v1 locates in the range of v1
                        proj2on1 = np.dot(vector1, vector2) / np.linalg.norm(vector1)
                        if proj2on1 < 0.0:
                            continue
                        elif proj2on1 > np.linalg.norm(vector1):
                            # the atom kk is out of the range of this bond
                            continue
                        else:
                            theta = np.arccos(
                                np.dot(vector1, vector2) / np.linalg.norm(vector2) / np.linalg.norm(vector1))
                            if theta < 0.0 or theta > np.pi / 2.0:
                                raise RuntimeError("Unexpected theta=%3.f" % theta)
                            proj_d = np.linalg.norm(vector2) * np.sin(theta)
                            if proj_d < covalent_radii[_an[kk]]:
                                # atom kk located in the line of ii and jj
                                has_bond = False
                                break
                if has_bond:
                    cG.add_edge(ii, jj, length=_DM[ii][jj])
        return cG

    def get_connectivity_graph(self):
        return self.cG

    def get_coordination_number(self, index, neighbor_element=None):
        """
        :param index: index for the central atom
        :param neighbor_element: str or list, the symbols of neighbouring atoms.
        :return: int
        """
        if isinstance(neighbor_element, str):
            neighbor_element = [neighbor_element]
        CN = 0
        for nb in self.cG.neighbors(index):
            if neighbor_element is None:
                CN += 1
            elif self.cG.nodes[nb]['symbol'] in neighbor_element:
                CN += 1
        return CN

    def get_coordination_details(self, index):
        """
        :param index: index for the central atom
        :return: dict
        """
        neighbors = []
        for nb in self.cG.neighbors(index):
            neighbors.append(nb)
        return neighbors

    def is_isomorphic(self, atoms):
        from networkx.algorithms import isomorphism
        cG = self._search_bond(atoms=atoms)
        nm = isomorphism.categorical_node_match('symbol', 'X')
        return isomorphism.is_isomorphic(cG, self.cG, node_match=nm)

    def get_framework(self, element=None):
        if element is None:
            raise RuntimeError("You must specify an element to define an framework, like Pt")
        elif isinstance(element, str):
            element = [element]
        node_list = []
        for nd in self.cG.nodes():
            if self.cG.nodes[nd]['symbol'] in element:
                node_list.append(nd)
        return self.cG.subgraph(node_list)
