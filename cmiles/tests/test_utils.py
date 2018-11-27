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
    assert cmiles.utils.is_mapped(mapped_mol) == True
    cmiles.utils.remove_map(mapped_mol)
    assert cmiles.utils.is_mapped(mapped_mol) == False


@using_rdkit
def test_is_mapped_rd():
    """Test is mapped"""
    mapped_smiles = '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'
    mapped_mol = cmiles.utils.load_molecule(mapped_smiles, backend='rdkit')
    assert cmiles.utils.is_mapped(mapped_mol) == True
    cmiles.utils.remove_map(mapped_mol)
    assert cmiles.utils.is_mapped(mapped_mol) == False


# @using_openeye
# def test_mol_from_json_oe():
#     """Test oemol from json"""
#     import numpy as np
#     hooh = {
#         'symbols': ['H', 'O', 'O', 'H'],
#         'geometry': [
#              1.84719633,  1.47046223,  0.80987166,
#              1.3126021,  -0.13023157, -0.0513322,
#             -1.31320906,  0.13130216, -0.05020593,
#             -1.83756335, -1.48745318,  0.80161212
#         ],
#         'name': 'HOOH',
#         'connectivity': [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
#     }
#     oe_mol = cmiles.utils.load_molecule(hooh)
#     assert oe_mol.GetMaxAtomIdx() == 4
#     assert oe_mol.GetMaxBondIdx() == 3
#     coordinates = oe_mol.GetCoords()
#     geometry = np.array(hooh['geometry'], dtype=float).reshape(int(len(hooh['geometry'])/3), 3)*cmiles.utils.BOHR_2_ANGSTROM
#     for i in range(len(coordinates)):
#         for j in range(3):
#             assert coordinates[i][j] == pytest.approx(geometry[i][j], 0.0000001)


@using_rdkit
def test_mol_from_json_rd():
    """Test rdmol from json"""
    import numpy as np
    hooh = {
        'symbols': ['H', 'O', 'O', 'H'],
        'geometry': [
             1.84719633,  1.47046223,  0.80987166,
             1.3126021,  -0.13023157, -0.0513322,
            -1.31320906,  0.13130216, -0.05020593,
            -1.83756335, -1.48745318,  0.80161212
        ],
        'name': 'HOOH',
        'connectivity': [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
    }
    rd_mol = cmiles.utils.load_molecule(hooh, backend='rdkit')
    assert rd_mol.GetNumAtoms() == 4
    assert rd_mol.GetNumBonds() == 3
    geometry = np.array(hooh['geometry'], dtype=float).reshape(int(len(hooh['geometry'])/3), 3)*cmiles.utils.BOHR_2_ANGSTROM
    coordinates = rd_mol.GetConformer().GetPositions()
    for i in range(len(coordinates)):
        for j in range(3):
            assert coordinates[i][j] == pytest.approx(geometry[i][j], 0.0000001)


