"""Test util functions"""

import cmiles
import pytest
from pkg_resources import resource_filename
import os
import warnings

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
    mol = cmiles.utils.load_molecule('[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]')
    assert oechem.OEMolToSmiles(mol) == 'CCCC'


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


@using_openeye
def test_mol_from_json_oe():
    """Test oemol from json"""
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
    oe_mol = cmiles.utils.load_molecule(hooh)
    assert oe_mol.GetMaxAtomIdx() == 4
    assert oe_mol.GetMaxBondIdx() == 3
    coordinates = oe_mol.GetCoords()
    geometry = np.array(hooh['geometry'], dtype=float).reshape(int(len(hooh['geometry'])/3), 3)*cmiles.utils.BOHR_2_ANGSTROM
    for i in range(len(coordinates)):
        for j in range(3):
            assert coordinates[i][j] == pytest.approx(geometry[i][j], 0.0000001)


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

@using_openeye
@pytest.mark.parametrize("input1, input2", [('C[C@](N)(O)F', 'CC(N)(O)F'),
                                   ('C(=C/Cl)\\F', 'C(=CCl)F'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4')])
def test_has_stereochemistry_oe(input1, input2):
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, input1)
    oechem.OEAddExplicitHydrogens(mol)
    assert cmiles.utils.has_stereo_defined(mol, backend='openeye')

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, input2)
    oechem.OEAddExplicitHydrogens(mol)
    with pytest.raises(ValueError):
        cmiles.utils.has_stereo_defined(mol, backend='openeye')

@using_rdkit
@pytest.mark.parametrize("input1, input2", [('C[C@](N)(O)F', 'CC(N)(O)F'),
                                   ('C(=C/Cl)\\F', 'C(=CCl)F'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4')])
def test_has_stereochemistry_oe(input1, input2):
    mol = Chem.MolFromSmiles(input1)
    mol = Chem.AddHs(mol)
    assert cmiles.utils.has_stereo_defined(mol, backend='rdkit')

    mol = Chem.MolFromSmiles(input2)
    mol = Chem.AddHs(mol)
    with pytest.raises(ValueError):
        cmiles.utils.has_stereo_defined(mol, backend='rdkit')


@using_openeye
def test_canonical_order():
    """Test canonical atom order"""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, 'HOOH')

    # add map
    for atom in mol.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)

    assert oechem.OEMolToSmiles(mol) == '[H:1][O:2][O:3][H:4]'

    mol_2 = cmiles.utils.canonical_order_atoms_oe(mol, in_place=False)
    for atom in mol.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    for atom in mol_2.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    assert oechem.OEMolToSmiles(mol) == '[H:1][O:2][O:3][H:4]'
    assert oechem.OEMolToSmiles(mol_2) == '[H:3][O:1][O:2][H:4]'

    cmiles.utils.canonical_order_atoms_oe(mol, in_place=True)
    for atom in mol.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    assert oechem.OEMolToSmiles(mol) == '[H:3][O:1][O:2][H:4]'


@using_rdkit
def test_canonical_order_rd():
    """Test canonical atom order"""
    mol = Chem.MolFromSmiles('CO')

    # add map
    mol = Chem.AddHs(mol)
    for i in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(i).SetAtomMapNum(i+1)

    assert Chem.MolToSmiles(mol) == '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'

    mol_2 = cmiles.utils.canonical_order_atoms_rd(mol, h_last=False)
    assert Chem.MolToSmiles(mol_2) == '[H:1][O:5][C:6]([H:2])([H:3])[H:4]'

    mol_3 = cmiles.utils.canonical_order_atoms_rd(mol)
    assert Chem.MolToSmiles(mol_3) == '[O:1]([C:2]([H:4])([H:5])[H:6])[H:3]'


@using_openeye
def test_explicit_h_oe():
    """Test input SMILES for explicit H"""

    implicit_h = 'COC(C)c1c(Cl)ccc(F)c1Cl'
    some_explicit_h = 'C[C@@H](c1c(ccc(c1Cl)F)Cl)OC'
    explicit_h = '[H]c1c(c(c(c(c1F)Cl)[C@]([H])(C([H])([H])[H])OC([H])([H])[H])Cl)[H]'
    mapped = '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'

    assert implicit_h.find('H') == -1
    assert some_explicit_h.find('H')
    assert explicit_h.find('H')
    assert mapped.find('H')

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, implicit_h)
    assert cmiles.utils.has_explicit_hydrogen(mol) == False

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, some_explicit_h)
    cmiles.utils.has_explicit_hydrogen(mol) == False

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, explicit_h)
    assert cmiles.utils.has_explicit_hydrogen(mol)

    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, mapped)
    assert cmiles.utils.has_explicit_hydrogen(mol)

    # no need for H
    o = 'O=O'
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, o)
    assert cmiles.utils.has_explicit_hydrogen(mol)


def test_explicit_h_rd():
    """Test input SMILES for explicit H"""

    implicit_h = 'COC(C)c1c(Cl)ccc(F)c1Cl'
    some_explicit_h = 'C[C@@H](c1c(ccc(c1Cl)F)Cl)OC'
    explicit_h = '[H]c1c(c(c(c(c1F)Cl)[C@]([H])(C([H])([H])[H])OC([H])([H])[H])Cl)[H]'
    mapped = '[C:1]([O:2][H:6])([H:3])([H:4])[H:5]'

    mol = Chem.MolFromSmiles(implicit_h)
    assert cmiles.utils.has_explicit_hydrogen(mol, backend='rdkit') == False

    mol = Chem.MolFromSmiles(some_explicit_h)
    cmiles.utils.has_explicit_hydrogen(mol, backend='rdkit') == False

    mol = Chem.MolFromSmiles(explicit_h)
    assert cmiles.utils.has_explicit_hydrogen(mol, backend='rdkit')

    mol = Chem.MolFromSmiles(mapped)
    assert cmiles.utils.has_explicit_hydrogen(mol, backend='rdkit')

    # no need for H
    o = 'O=O'
    mol = Chem.MolFromSmiles(o)
    assert cmiles.utils.has_explicit_hydrogen(mol, backend='rdkit')


@using_openeye
@pytest.mark.parametrize("smiles, output", [('CN(C)C(=N)NC(=N)N', True),
                                           ('COC1=CC(CN(C)C2=CC=C3N=C(N)N=C(N)C3=C2)=C(OC)C=C1', False),
                                           ('[H][C@](C)(O)[C@@]([H])(N=C(O)[C@]1([H])C[C@@]([H])(CCC)CN1C)[C@@]1([H])O[C@]([H])(SC)[C@]([H])(O)[C@@]([H])(O)[C@@]1([H])O',
                                            False)])
def test_chiral_bond_exception(smiles, output):
    """ Test bonds to ignore """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    oechem.OEAddExplicitHydrogens(mol)
    ignore = False
    for bond in mol.GetBonds():
        ignore = cmiles.utils._ignore_stereo_flag(mol, bond)
        if ignore:
            break
    assert ignore == output
