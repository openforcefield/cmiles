"""util functions for test suite"""
from pkg_resources import resource_filename
import os
import re


def get_fn(filename, written=False):
    """Get the full path to one of the reference files shipped for testing

        These files are in torsionfit/testing/reference

    :param
        name: str
            Name of file to load

    :returns
        fn : str
            full path to file
    """
    if written:
        fn = resource_filename('cmiles', os.path.join('tests', 'reference', 'writes', filename))
    else:
        fn = resource_filename('cmiles', os.path.join('tests', 'reference', filename))

    #if not os.path.exists(fn):
    #    raise ValueError('%s does not exist. If you just added it you will have to re install' % fn)

    return fn


def get_smiles_lists(f1, f2):

    smiles = []
    for f in (f1, f2):
        smiles.append(get_smiles_list(f))
    return list(zip(smiles[0], smiles[-1]))


def get_smiles_list(f):
    smiles = open(f, 'r').read().split('\n')[:-1]
    for i in range(len(smiles)):
        smiles[i] = smiles[i].split(' ')[0]
    return smiles