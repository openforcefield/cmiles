"""Test util functions"""

import cmiles
import pytest
from pkg_resources import resource_filename
import os

rdkit_missing = False
try:
    from rdkit import Chem
except ImportError:
    rdkit_missing = True

try:
    from openeye import oechem
    openeye_missing = False
except ImportError:
    openeye_missing = True

from .utils import get_fn

using_rdkit = pytest.mark.skipif(rdkit_missing, reason="Cannot run without RDKit")
using_openeye = pytest.mark.skipif(openeye_missing, reason="Cannot run without OpenEye")

@using_openeye
def test_load_molecule():
    """Test load molecules"""
    inputs = ['CCCC', get_fn('butane.pdb'), get_fn('butane.smi'), get_fn('butane.xyz')]
    outputs = []
    for i in inputs:
        outputs.append(cmiles.utils.load_molecule(i))
    for o in outputs:
        assert oechem.OEMolToSmiles((o)) == 'CCCC'

@using_openeye
def test_is_mapped_oe():
    """Test is mapped"""
    mapped_smiles = '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'
    mapped_mol = cmiles.utils.load_molecule(mapped_smiles, backend='openeye')
    assert cmiles.utils.is_mapped(mapped_mol, backend='openeye') == True
    cmiles.utils.remove_map(mapped_mol, backend='openeye')
    assert cmiles.utils.is_mapped(mapped_mol, backend='openeye') == False


@using_rdkit
def test_is_mapped_rd():
    """Test is mapped"""
    mapped_smiles = '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'
    mapped_mol = cmiles.utils.load_molecule(mapped_smiles, backend='rdkit')
    assert cmiles.utils.is_mapped(mapped_mol, backend='rdkit') == True
    cmiles.utils.remove_map(mapped_mol, backend='rdkit')
    assert cmiles.utils.is_mapped(mapped_mol, backend='rdkit') == False

