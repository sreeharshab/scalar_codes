raise RuntimeError("This file is already deprecated. If you get here, that means something went wrong with the code")
try:
    from ase.constraints import NeverDelete
except ImportError:
    import sys
    import numpy as np
    from ase.constraints import FixConstraint
    from ase.constraints import slice2enlist
    from ase.constraints import ints2string

    sys.stderr.write("pygcga.neverdelete is used, This constraint does not support ase.io.read\n")

    class NeverDelete(FixConstraint):
        """
        This is a dumpy NeverDelete constraint, Which is only are used for identify the atoms which should not be deleted
        during a grand canonical sampling method.
        """
        def __init__(self, indices=None, mask=None):
            """Constrain chosen atoms.

            Parameters
            ----------
            indices : list of int
               Indices for those atoms that should never be deleted.
            mask : list of bool
               One boolean per atom indicating if the atom should be
               constrained or not.
            """

            if indices is None and mask is None:
                raise ValueError('Use "indices" or "mask".')
            if indices is not None and mask is not None:
                raise ValueError('Use only one of "indices" and "mask".')

            if mask is not None:
                indices = np.arange(len(mask))[np.asarray(mask, bool)]
            else:
                # Check for duplicates:
                srt = np.sort(indices)
                if (np.diff(srt) == 0).any():
                    raise ValueError(
                        'NeverDelete: The indices array contained duplicates. '
                        'Perhaps you wanted to specify a mask instead, but '
                        'forgot the mask= keyword.')
            self.index = np.asarray(indices, int)

            if self.index.ndim != 1:
                raise ValueError('Wrong argument to NeverDelete class!')

            self.removed_dof = 0

        def adjust_positions(self, atoms, new):
            # new[self.index] = atoms.positions[self.index]
            pass

        def adjust_forces(self, atoms, forces):
            # forces[self.index] = 0.0
            pass

        def index_shuffle(self, atoms, ind):
            # See docstring of superclass
            index = []
            for new, old in slice2enlist(ind, len(atoms)):
                if old in self.index:
                    index.append(new)
            if len(index) == 0:
                raise IndexError('All indices in NeverDelete not part of slice')
            self.index = np.asarray(index, int)

        def get_indices(self):
            return self.index

        def __repr__(self):
            return 'NeverDelete(indices=%s)' % ints2string(self.index)

        def todict(self):
            return {'name': 'NeverDelete',
                    'kwargs': {'indices': self.index}}


