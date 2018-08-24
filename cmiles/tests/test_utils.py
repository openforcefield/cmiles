"""Test util functions"""

import cmiles
import pytest
from pkg_resources import resource_filename
import os
from openeye import oechem

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


def test_load_molecule():
    """Test load molecules"""
    inputs = ['CCCC', get_fn('butane.pdb'), get_fn('butane.smi'), get_fn('butane.xyz')]
    outputs = []
    for i in inputs:
        outputs.append(cmiles.utils.load_molecule(i))
    for o in outputs:
        assert oechem.OEMolToSmiles((o)) == 'CCCC'
