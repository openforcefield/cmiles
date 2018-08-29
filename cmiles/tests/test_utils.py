"""Test util functions"""

import cmiles
import pytest
from pkg_resources import resource_filename
import os
from openeye import oechem

from .utils import get_fn

def test_load_molecule():
    """Test load molecules"""
    inputs = ['CCCC', get_fn('butane.pdb'), get_fn('butane.smi'), get_fn('butane.xyz')]
    outputs = []
    for i in inputs:
        outputs.append(cmiles.utils.load_molecule(i))
    for o in outputs:
        assert oechem.OEMolToSmiles((o)) == 'CCCC'
