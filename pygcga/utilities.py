#!/usr/bin/env python
import hashlib
import os
import tarfile

try:
    import cPickle as pickle    # Python2
except ImportError:
    import pickle               # Python3


class PropertyNotImplementedError(NotImplementedError):
    pass


class NoReasonableStructureFound(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FileDatabase:
    """
    Notes: 2018-04-17
    Geng Sun used the following code from amp code (Peterson Brown),
    since it installed different images as separate files,
    it is more secure than a single ase.db file.

    Following is this original comments from Peterson@Brown university:
    ------------------------------------------------------------------

    Using a database file, such as shelve or sqlitedict, that can handle
    multiple processes writing to the file is hard.

    Therefore, we take the stupid approach of having each database entry be
    a separate file. This class behaves essentially like shelve, but saves each
    dictionary entry as a plain pickle file within the directory, with the
    filename corresponding to the dictionary key (which must be a string).

    Like shelve, this also keeps an internal (memory dictionary) representation
    of the variables that have been accessed.

    Also includes an archive feature, where files are instead added to a file
    called 'archive.tar.gz' to save disk space. If an entry exists in both the
    loose and archive formats, the loose is taken to be the new (correct)
    value.
    """

    def __init__(self, filename):
        """Open the filename at specified location. flag is ignored; this
        format is always capable of both reading and writing."""
        if not filename.endswith(os.extsep + 'gadb'):
            filename += os.extsep + 'gadb'
        self.path = filename
        self.loosepath = os.path.join(self.path, 'loose')
        self.tarpath = os.path.join(self.path, 'archive.tar.gz')
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            os.mkdir(self.loosepath)
        self._memdict = {}  # Items already accessed; stored in memory.

    @classmethod
    def open(Cls, filename, flag=None):
        """Open present for compatibility with shelve. flag is ignored; this
        format is always capable of both reading and writing.
        """
        return Cls(filename=filename)

    def close(self):
        """Only present for compatibility with shelve.
        """
        return

    def keys(self):
        """Return list of keys, both of in-memory and out-of-memory
        items.
        """
        keys = os.listdir(self.loosepath)
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                keys = list(set(keys + tf.getnames()))
        return keys

    def values(self):
        """Return list of values, both of in-memory and out-of-memory
        items. This moves all out-of-memory items into memory.
        """
        keys = self.keys()
        return [self[key] for key in keys]

    def __len__(self):
        return len(self.keys())

    def __setitem__(self, key, value):
        self._memdict[key] = value
        path = os.path.join(self.loosepath, str(key))
        if os.path.exists(path):
            with open(path, 'r') as f:
                if f.read() == pickle.dumps(value):
                    return  # Nothing to update.
        with open(path, 'wb') as f:
            pickle.dump(value, f)

    def __getitem__(self, key):
        if key in self._memdict:
            return self._memdict[key]
        keypath = os.path.join(self.loosepath, key)
        if os.path.exists(keypath):
            with open(keypath, 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                return pickle.load(tf.extractfile(key))
        else:
            raise KeyError(str(key))

    def update(self, newitems):
        for key, value in newitems.items():
            self.__setitem__(key, value)

    def archive(self):
        """Cleans up to save disk space and reduce huge number of files.

        That is, puts all files into an archive.  Compresses all files in
        <path>/loose and places them in <path>/archive.tar.gz.  If archive
        exists, appends/modifies.
        """
        loosefiles = os.listdir(self.loosepath)
        print('Contains %i loose entries.' % len(loosefiles))
        if len(loosefiles) == 0:
            print(' -> No action taken.')
            return
        if os.path.exists(self.tarpath):
            with tarfile.open(self.tarpath) as tf:
                names = [_ for _ in tf.getnames() if _ not in
                         os.listdir(self.loosepath)]
                for name in names:
                    tf.extract(member=name, path=self.loosepath)
        loosefiles = os.listdir(self.loosepath)
        print('Compressing %i entries.' % len(loosefiles))
        with tarfile.open(self.tarpath, 'w:gz') as tf:
            for file in loosefiles:
                tf.add(name=os.path.join(self.loosepath, file),
                       arcname=file)
        print('Cleaning up: removing %i files.' % len(loosefiles))
        for file in loosefiles:
            os.remove(os.path.join(self.loosepath, file))

    def save_images(self, images):
        """ This works similar with self.update"""
        hashed_images = hash_images(images)
        self.update(hashed_images)

    def get_a_population(self, n, looks_like=None, fitness_calculator=None, **kwargs):
        """
        :param n: number of structures included
        :param looks_like: function, if it is specified, only unique structure is returned. Function looks_like take
               two parameters
        :param fitness_calculator: if specified, the fitness is calculated with this function, taking the atoms object
        as the first paramter
        :param kwargs: other parameters fed into the fitness_calculator function.
        :return: a list of atoms [(atoms,fitness),(atoms,fitness)...]
        """
        images = self.values()
        if fitness_calculator is None:
            images.sort(key=lambda x: x.get_potential_energy())
        else:
            images.sort(reverse=True,key=lambda x: fitness_calculator(x, **kwargs))
        _return_images=[]
        while len(images) > 0 and len(_return_images) < n:
            atoms=images.pop(0)
            existed=False
            for a in _return_images:
                if looks_like(a, atoms):
                    existed=True
                    break
            if (not existed):
                _return_images.append(atoms)
        return _return_images

    def get_an_ensemble(self,looks_like=None,fitness_calculator=None,energy_gap=0.25,**kwargs):
        images=self.values()
        images.sort(reverse=True, key=lambda x: fitness_calculator(x, **kwargs))
        _maxfit=fitness_calculator(images[0],**kwargs)
        _return_images=[]
        excess_gap=False
        while len(images) > 0 and (not excess_gap):
            atoms=images.pop(0)
            existed=False
            if _maxfit-fitness_calculator(atoms,**kwargs) > energy_gap:
                excess_gap=True
                continue
            for a in _return_images:
                if looks_like(a,atoms):
                    existed=True
                    break
            if not existed:
                _return_images.append(atoms)
        return _return_images


class Population(object):
    """Serves as a container (dictionary-like) for (key, value) pairs that
    also serves to calculate them.

    Works by default with python's shelve module, but something that is built
    to share the same commands as shelve will work fine; just specify this in
    dbinstance.

    Designed to hold things like neighborlists, which have a hash, value
    format.

    This will work like a dictionary in that items can be accessed with
    data[key], but other advanced dictionary functions should be accessed with
    through the .d attribute:

    >>> data = Data(...)
    >>> data.open()
    >>> keys = data.d.keys()
    >>> values = data.d.values()
    """

    def __init__(self, filename, db=FileDatabase):
        self.db = db
        self.filename = filename
        self.d = None

    # def calculate_items(self, images, parallel, log=None):

    def __getitem__(self, key):
        self.open()
        return self.d[key]

    def close(self):
        """Safely close the database.
        """
        if self.d:
            self.d.close()
        self.d = None

    def open(self, mode='r'):
        """Open the database connection with mode specified.
        """
        if self.d is None:
            self.d = self.db.open(self.filename, mode)

    def __del__(self):
        self.close()

    def get_all_images(self):
        if self.d is None:
            self.d=self.db.open(self.filename,'r')
        return self.d.values()

    def get_number_of_structures(self):
        if self.d is None:
            self.d=self.db.open(self.filename,'r')
        return len(self.d)


def get_hash(atoms):
    """Creates a unique signature for a particular ASE atoms object.

    This is used to check whether an image has been seen before. This is just
    an md5 hash of a string representation of the atoms object.

    Parameters
    ----------
    atoms : ASE dict
        ASE atoms object.

    Returns
    -------
        Hash string key of 'atoms'.
    """
    string = str(atoms.pbc)
    for number in atoms.cell.flatten():
        string += '%.15f' % number
    string += str(atoms.get_atomic_numbers())
    for number in atoms.get_positions().flatten():
        string += '%.15f' % number

    md5 = hashlib.md5(string.encode('utf-8'))
    hash = md5.hexdigest()
    return hash


def hash_images(images):
    """
    images must be a list
    """
    dict_images={}
    for image in images:
        hash = get_hash(image)
        if hash in dict_images.keys():
            continue
        dict_images[hash] = image
    return dict_images


def is_same_ensemble(ensemble1=None,ensemble2=None,comparator=None):
    if len(ensemble1) != len(ensemble2):
        return False
    else:
        for a1,a2 in zip(ensemble1,ensemble2):
            if not comparator(a1,a2):
                return False
        return True





