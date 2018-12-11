"""
Unit and regression test for the cmiles package.
"""

# Import package, test suite, and other packages as needed

import pytest
import sys
from .utils import get_fn, get_smiles_lists, get_smiles_list
import cmiles

rdkit_missing = False
try:
    from rdkit import Chem
except ImportError:
    rdkit_missing = True

openeye_missing = False
try:
    from openeye import oechem
except ImportError:
    openeye_missing = True

using_rdkit = pytest.mark.skipif(rdkit_missing, reason="Cannot run without RDKit")
using_openeye = pytest.mark.skipif(openeye_missing, reason="Cannot run without OpenEye")


def test_cmiles_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cmiles" in sys.modules

@pytest.fixture
def smiles_input():
    return([
        "C[C@@H](c1c(ccc(c1Cl)F)Cl)OC",
        "CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4",
        "N[C@@](C)(F)C(=O)O",
        "c1ccc2c(c1)C=CCC2=O",
        "c1ccc2c(c1)cccc2O",
        "CC[N+]#C",
        "Cc1ccccc1",
    ])

@pytest.fixture
def rd_iso_expected():
    return(["CO[C@@H](C)c1c(Cl)ccc(F)c1Cl",
            "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
            "C[C@](N)(F)C(=O)O",
            "O=C1CC=Cc2ccccc21",
            "Oc1cccc2ccccc12",
            "C#[N+]CC",
            "Cc1ccccc1"

    ])

@pytest.fixture
def rd_map_expected():
    return (['[F:1][c:7]1[c:5]([H:14])[c:6]([H:15])[c:8]([Cl:2])[c:10]([C@@:13]([O:4][C:11]([H:16])([H:17])[H:18])([C:12]([H:19])([H:20])[H:21])[H:22])[c:9]1[Cl:3]',
             '[O:1]=[C:8](/[C:9](=[C:10](/[C:30]([N:27]([C:28]([H:45])([H:46])[H:47])[C:29]([H:48])([H:49])[H:50])([H:51])[H:52])[H:36])[H:35])[N:25]([c:18]1[c:17]([O:7][C@:34]2([H:59])[C:32]([H:55])([H:56])[O:6][C:31]([H:53])([H:54])[C:33]2([H:57])[H:58])[c:20]([H:41])[c:23]2[n:5][c:11]([H:37])[n:4][c:21]([N:26]([c:19]3[c:13]([H:39])[c:12]([H:38])[c:15]([F:2])[c:16]([Cl:3])[c:14]3[H:40])[H:44])[c:24]2[c:22]1[H:42])[H:43]',
             '[O:1]=[C:4]([O:3][H:8])[C@:7]([F:2])([N:5]([H:9])[H:10])[C:6]([H:11])([H:12])[H:13]',
             '[O:1]=[C:2]1[c:9]2[c:7]([H:16])[c:5]([H:14])[c:6]([H:15])[c:8]([H:17])[c:10]2[C:3]([H:12])=[C:4]([H:13])[C:11]1([H:18])[H:19]',
             '[O:1]([c:9]1[c:5]([H:16])[c:3]([H:14])[c:7]([H:18])[c:10]2[c:6]([H:17])[c:2]([H:13])[c:4]([H:15])[c:8]([H:19])[c:11]12)[H:12]',
             '[C:1](#[N+:2][C:4]([C:3]([H:6])([H:7])[H:8])([H:9])[H:10])[H:5]',
             '[c:1]1([H:8])[c:2]([H:9])[c:4]([H:11])[c:6]([C:7]([H:13])([H:14])[H:15])[c:5]([H:12])[c:3]1[H:10]'
            ])

@pytest.fixture
def rd_can_expected():
    return (['COC(C)c1c(Cl)ccc(F)c1Cl',
             'CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',
             'CC(N)(F)C(=O)O',
             'O=C1CC=Cc2ccccc21',
             'Oc1cccc2ccccc12',
             'C#[N+]CC',
             'Cc1ccccc1'
    ])

@pytest.fixture
def rd_h_expected():
    return (['[H][c]1[c]([H])[c]([Cl])[c]([C@@]([H])([O][C]([H])([H])[H])[C]([H])([H])[H])[c]([Cl])[c]1[F]',
             '[H]/[C]([C](=[O])[N]([H])[c]1[c]([O][C@]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C](/[H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]',
             '[H][O][C](=[O])[C@]([F])([N]([H])[H])[C]([H])([H])[H]',
             '[H][C]1=[C]([H])[C]([H])([H])[C](=[O])[c]2[c]([H])[c]([H])[c]([H])[c]([H])[c]21',
             '[H][O][c]1[c]([H])[c]([H])[c]([H])[c]2[c]([H])[c]([H])[c]([H])[c]([H])[c]12',
             '[H][C]#[N+][C]([H])([H])[C]([H])([H])[H]',
             '[H][c]1[c]([H])[c]([H])[c]([C]([H])([H])[H])[c]([H])[c]1[H]'
    ])
@pytest.fixture
def oe_iso_expected():
    return(['C[C@@H](c1c(ccc(c1Cl)F)Cl)OC',
            'CN(C)C/C=C/C(=O)Nc1cc2c(cc1O[C@H]3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F',
            'C[C@](C(=O)O)(N)F',
            'c1ccc2c(c1)C=CCC2=O',
            'c1ccc2c(c1)cccc2O',
            'CC[N+]#C',
            'Cc1ccccc1'
    ])

@pytest.fixture
def oe_map_expected():
    return (['[H:14][c:1]1[c:2]([c:5]([c:3]([c:6]([c:4]1[F:11])[Cl:13])[C@:9]([H:22])([C:7]([H:16])([H:17])[H:18])[O:10][C:8]([H:19])([H:20])[H:21])[Cl:12])[H:15]',
             '[H:35][c:1]1[c:2]([c:12]([c:13]([c:5]([c:9]1[N:27]([H:58])[c:14]2[c:7]3[c:3]([c:10]([c:11]([c:4]([c:8]3[n:25][c:6]([n:26]2)[H:40])[H:38])[O:32][C@:21]4([C:18]([C:19]([O:31][C:20]4([H:47])[H:48])([H:45])[H:46])([H:43])[H:44])[H:49])[N:28]([H:59])[C:17](=[O:30])/[C:15](=[C:16](\\[H:42])/[C:24]([H:56])([H:57])[N:29]([C:22]([H:50])([H:51])[H:52])[C:23]([H:53])([H:54])[H:55])/[H:41])[H:37])[H:39])[Cl:34])[F:33])[H:36]',
             '[H:8][C:2]([H:9])([H:10])[C@:3]([C:1](=[O:5])[O:6][H:13])([N:4]([H:11])[H:12])[F:7]',
             '[H:12][c:1]1[c:2]([c:4]([c:6]2[c:5]([c:3]1[H:14])[C:7](=[C:8]([C:10]([C:9]2=[O:11])([H:18])[H:19])[H:17])[H:16])[H:15])[H:13]',
             '[H:12][c:1]1[c:2]([c:5]([c:9]2[c:8]([c:4]1[H:15])[c:6]([c:3]([c:7]([c:10]2[O:11][H:19])[H:18])[H:14])[H:17])[H:16])[H:13]',
             '[H:5][C:1]#[N+:4][C:3]([H:9])([H:10])[C:2]([H:6])([H:7])[H:8]',
             '[H:8][c:1]1[c:2]([c:4]([c:6]([c:5]([c:3]1[H:10])[H:12])[C:7]([H:13])([H:14])[H:15])[H:11])[H:9]'])

@pytest.fixture
def oe_can_expected():
    return (['CC(c1c(ccc(c1Cl)F)Cl)OC',
             'CN(C)CC=CC(=O)Nc1cc2c(cc1OC3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F',
             'CC(C(=O)O)(N)F',
             'c1ccc2c(c1)C=CCC2=O',
             'c1ccc2c(c1)cccc2O',
             'CC[N+]#C',
             'Cc1ccccc1'
    ])

@pytest.fixture
def oe_h_expected():
    return (['[H]c1c(c(c(c(c1F)Cl)[C@]([H])(C([H])([H])[H])OC([H])([H])[H])Cl)[H]',
             '[H]c1c(c(c(c(c1N([H])c2c3c(c(c(c(c3nc(n2)[H])[H])O[C@]4(C(C(OC4([H])[H])([H])[H])([H])[H])[H])N([H])C(=O)/C(=C(\\[H])/C([H])([H])N(C([H])([H])[H])C([H])([H])[H])/[H])[H])[H])Cl)F)[H]',
             '[H]C([H])([H])[C@](C(=O)O[H])(N([H])[H])F',
             '[H]c1c(c(c2c(c1[H])C(=C(C(C2=O)([H])[H])[H])[H])[H])[H]',
             '[H]c1c(c(c2c(c1[H])c(c(c(c2O[H])[H])[H])[H])[H])[H]',
             '[H]C#[N+]C([H])([H])C([H])([H])[H]',
             '[H]c1c(c(c(c(c1[H])[H])C([H])([H])[H])[H])[H]'])


@using_rdkit
def test_rdkit_isomeric(smiles_input, rd_iso_expected):
    """testing rdkit isomeric canonical smiles"""
    for i, o in zip(smiles_input, rd_iso_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=False, explicit_hydrogen=False) == o


@using_rdkit
def test_rdkit_map(smiles_input, rd_map_expected):
    """Testing rdkit canonical map ordering"""
    for i, o in zip(smiles_input, rd_map_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=True, explicit_hydrogen=True) == o


@using_rdkit
def test_rdkit_canonical(smiles_input, rd_can_expected):
    """Testing rdkit canonical smiles"""
    for i, o in zip(smiles_input, rd_can_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=False, mapped=False, explicit_hydrogen=False) == o


@using_rdkit
def test_rdkit_explicit_h(smiles_input, rd_h_expected):
    """Testing rdkit explicit hydrogen"""
    for i, o in zip(smiles_input, rd_h_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=False, explicit_hydrogen=True) == o


@using_openeye
def test_openeye_explicit_h(smiles_input, oe_h_expected):
    """Testing openeye explicit hydrogen"""
    for i, o in zip(smiles_input, oe_h_expected):
        oe_mol = oechem.OEMol()
        oechem.OESmilesToMol(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=False, explicit_hydrogen=True) == o


@using_openeye
def test_openeye_isomeric(smiles_input, oe_iso_expected):
    """testing rdkit isomeric canonical smiles"""
    for i, o in zip(smiles_input, oe_iso_expected):
        oe_mol = oechem.OEMol()
        oechem.OESmilesToMol(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=False, explicit_hydrogen=False) == o


@using_openeye
def test_openeye_map(smiles_input, oe_map_expected):
    """Testing rdkit canonical map ordering"""
    for i, o in zip(smiles_input, oe_map_expected):
        oe_mol = oechem.OEMol()
        oechem.OESmilesToMol(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=True, explicit_hydrogen=True) == o


@using_openeye
def test_openeye_canonical(smiles_input, oe_can_expected):
    """Testing rdkit canonical smiles"""
    for i, o in zip(smiles_input, oe_can_expected):
        oe_mol = oechem.OEMol()
        oechem.OESmilesToMol(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=False, mapped=False, explicit_hydrogen=False) == o


@using_rdkit
@using_openeye
def test_oe_cmiles():
    """Regression test oe cmiles"""
    expected_output = {'canonical_smiles': 'CN(C)CC=CC(=O)Nc1cc2c(cc1OC3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F',
                       'canonical_isomeric_smiles': 'CN(C)C/C=C/C(=O)Nc1cc2c(cc1O[C@@H]3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F',
                       'canonical_isomeric_explicit_hydrogen_smiles':
                           '[H]c1c(c(c(c(c1N([H])c2c3c(c(c(c(c3nc(n2)[H])[H])O[C@@]4(C(C(OC4([H])[H])([H])[H])([H])[H])[H])N([H])C(=O)/C(=C(\\[H])/C([H])([H])N(C([H])([H])[H])C([H])([H])[H])/[H])[H])[H])Cl)F)[H]',
                       'canonical_explicit_hydrogen_smiles':
                           '[H]c1c(c(c(c(c1N([H])c2c3c(c(c(c(c3nc(n2)[H])[H])OC4(C(C(OC4([H])[H])([H])[H])([H])[H])[H])N([H])C(=O)C(=C([H])C([H])([H])N(C([H])([H])[H])C([H])([H])[H])[H])[H])[H])Cl)F)[H]',
                       'canonical_isomeric_explicit_hydrogen_mapped_smiles':
                           '[H:35][c:1]1[c:2]([c:12]([c:13]([c:5]([c:9]1[N:27]([H:58])[c:14]2[c:7]3[c:3]([c:10]([c:11]([c:4]([c:8]3[n:25][c:6]([n:26]2)[H:40])[H:38])[O:32][C@@:21]4([C:18]([C:19]([O:31][C:20]4([H:47])[H:48])([H:45])[H:46])([H:43])[H:44])[H:49])[N:28]([H:59])[C:17](=[O:30])/[C:15](=[C:16](\\[H:42])/[C:24]([H:56])([H:57])[N:29]([C:22]([H:50])([H:51])[H:52])[C:23]([H:53])([H:54])[H:55])/[H:41])[H:37])[H:39])[Cl:34])[F:33])[H:36]',
                       'provenance': 'cmiles_0.0.0+1.geb7d850.dirty_openeye_2018.Feb.b6'}
    smiles = 'CN(C)C/C=C/C(=O)Nc1cc2c(cc1O[C@@H]3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F'
    output = cmiles.to_molecule_id(smiles, canonicalization='openeye', strict=False)
    assert expected_output['canonical_smiles'] == output['canonical_smiles']
    assert expected_output['canonical_isomeric_smiles'] == output['canonical_isomeric_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_smiles'] == output['canonical_isomeric_explicit_hydrogen_smiles']
    assert expected_output['canonical_explicit_hydrogen_smiles'] == output['canonical_explicit_hydrogen_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == output['canonical_isomeric_explicit_hydrogen_mapped_smiles']


@using_rdkit
def test_rd_cmiles():
    """Regression test rdkit cmiles"""
    expected_output = {'canonical_smiles': 'CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',
 'canonical_isomeric_smiles': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1',
 'canonical_isomeric_explicit_hydrogen_smiles': '[H]/[C]([C](=[O])[N]([H])[c]1[c]([O][C@]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C](/[H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]',
 'canonical_explicit_hydrogen_smiles': '[H][C]([C](=[O])[N]([H])[c]1[c]([O][C]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C]([H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]',
 'canonical_isomeric_explicit_hydrogen_mapped_smiles': '[O:1]=[C:8](/[C:9](=[C:10](/[C:30]([N:27]([C:28]([H:45])([H:46])[H:47])[C:29]([H:48])([H:49])[H:50])([H:51])[H:52])[H:36])[H:35])[N:25]([c:18]1[c:17]([O:7][C@:34]2([H:59])[C:32]([H:55])([H:56])[O:6][C:31]([H:53])([H:54])[C:33]2([H:57])[H:58])[c:20]([H:41])[c:23]2[n:5][c:11]([H:37])[n:4][c:21]([N:26]([c:19]3[c:13]([H:39])[c:12]([H:38])[c:15]([F:2])[c:16]([Cl:3])[c:14]3[H:40])[H:44])[c:24]2[c:22]1[H:42])[H:43]',
 'provenance': 'cmiles_0.0.0+7.gc71f3a6.dirty_rdkit_2018.03.3'}
    smiles = 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1'
    output = cmiles.to_molecule_id(smiles, canonicalization='rdkit', strict=False)
    assert expected_output['canonical_smiles'] == output['canonical_smiles']
    assert expected_output['canonical_isomeric_smiles'] == output['canonical_isomeric_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_smiles'] == output['canonical_isomeric_explicit_hydrogen_smiles']
    assert expected_output['canonical_explicit_hydrogen_smiles'] == output['canonical_explicit_hydrogen_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == output['canonical_isomeric_explicit_hydrogen_mapped_smiles']


@using_openeye
@using_rdkit
def test_diff_smiles():
    """Test different SMILES of same molecule"""
    input_smiles = ['C[C@H](c1c(ccc(c1Cl)F)Cl)OC', 'CO[C@H](C)c1c(Cl)ccc(F)c1Cl']
    cmiles_1 = cmiles.to_molecule_id(input_smiles[0], strict=False)
    cmiles_2 = cmiles.to_molecule_id(input_smiles[1], strict=False)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] == cmiles_2['canonical_isomeric_smiles']
    assert cmiles_1['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == cmiles_2['canonical_isomeric_explicit_hydrogen_mapped_smiles']

    cmiles_1 = cmiles.to_molecule_id(input_smiles[0], canonicalization='rdkit', strict=False)
    cmiles_2 = cmiles.to_molecule_id(input_smiles[1], canonicalization='rdkit', strict=False)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] == cmiles_2['canonical_isomeric_smiles']
    assert cmiles_1['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == cmiles_2['canonical_isomeric_explicit_hydrogen_mapped_smiles']

@using_openeye
@using_rdkit
def test_initial_iso():
    """test given chirality"""

    input_smiles = ["CC[C@@H](C)N",  "CC[C@H](C)N"]
    cmiles_1 = cmiles.to_molecule_id(input_smiles[0], strict=False)
    cmiles_2 = cmiles.to_molecule_id(input_smiles[-1], strict=False)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] != cmiles_2['canonical_isomeric_smiles']

    cmiles_1 = cmiles.to_molecule_id(input_smiles[0], canonicalization='rdkit', strict=False)
    cmiles_2 = cmiles.to_molecule_id(input_smiles[-1], canonicalization='rdkit', strict=False)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] != cmiles_2['canonical_isomeric_smiles']


@using_rdkit
@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_stereo.smi'), get_fn('drug_bank_mapped_smi_rd.smi')))
def test_drug_bank_rd(input, output):
    """

    Parameters
    ----------
    input
    output

    Returns
    -------

    """

    mol = Chem.MolFromSmiles(input)
    assert cmiles.generator.to_canonical_smiles_rd(mol, mapped=True, isomeric=True, explicit_hydrogen=True) == output


@using_openeye
@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_stereo.smi'), get_fn('drug_bank_mapped_smi_oe.smi')))
def test_drug_bank_oe(input, output):
    """

    Parameters
    ----------
    input
    output

    Returns
    -------

    """
    mol = oechem.OEMol()
    oechem.OEParseSmiles(mol, input)
    assert cmiles.generator.to_canonical_smiles_oe(mol, mapped=True, isomeric=True, explicit_hydrogen=True) == output


@using_rdkit
@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_sm.smi'), get_fn('drug_bank_inchi_rd.txt')))
def test_inchi(input, output):
    """Check that inchis are the same"""

    rd_mol = Chem.MolFromSmiles(input)
    rd_inchi = cmiles.generator.to_inchi_and_key(rd_mol)[0]

    assert rd_inchi == output


@using_rdkit
@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_sm.smi'), get_fn('drug_bank_inchikey_rd.txt')))
def test_inchi_key(input, output):
    """Check that inchi key is the same"""

    rd_mol = Chem.MolFromSmiles(input)
    rd_inchi_key = cmiles.generator.to_inchi_and_key(rd_mol)[1]

    assert rd_inchi_key == output

@using_openeye
@using_rdkit
def test_input_mapped():
    smiles = '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'
    mol_id = cmiles.to_molecule_id(smiles)

    mol_1 = cmiles.utils.load_molecule(mol_id['canonical_isomeric_smiles'])
    mol_2 = cmiles.utils.load_molecule(mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'])
    assert cmiles.utils.is_mapped(mol_1) == False
    assert cmiles.utils.is_mapped(mol_2) == True

    mol_id = cmiles.to_molecule_id(smiles, canonicalization='rdkit')

    mol_1 = cmiles.utils.load_molecule(mol_id['canonical_isomeric_smiles'], backend='rdkit')
    mol_2 = cmiles.utils.load_molecule(mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'], backend='rdkit')
    assert cmiles.utils.is_mapped(mol_1) == False
    assert cmiles.utils.is_mapped(mol_2) == True


def _map_from_json(hooh, backend, map_smiles):

    molecule = cmiles.utils.load_molecule(hooh, backend=backend)
    if backend == 'openeye':
        mapped_smiles_1 = cmiles.generator.to_canonical_smiles_oe(molecule, isomeric=True, explicit_hydrogen=True, mapped=True)
        molecule.SetData("json_geometry", False)
        mapped_smiles_2 = cmiles.generator.to_canonical_smiles_oe(molecule, isomeric=True, explicit_hydrogen=True, mapped=True)
    if backend == 'rdkit':
        mapped_smiles_1 = cmiles.generator.to_canonical_smiles_rd(molecule, isomeric=True, explicit_hydrogen=True, mapped=True)
        molecule.SetProp("_json_geometry", '0')
        mapped_smiles_2 = cmiles.generator.to_canonical_smiles_rd(molecule, isomeric=True, explicit_hydrogen=True, mapped=True)

    assert mapped_smiles_1 == '[H:1][O:2][O:3][H:4]'
    assert mapped_smiles_2 == map_smiles

@using_openeye
@using_rdkit
@pytest.mark.parametrize("backend, map_smiles", [('openeye', '[H:3][O:1][O:2][H:4]'),
                                                 ('rdkit', '[O:1]([O:2][H:4])[H:3]')])
def test_map_order(backend, map_smiles):
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

    _map_from_json(hooh=hooh, backend=backend, map_smiles=map_smiles)


@using_openeye
@using_rdkit
def test_keep_chiral_stereo():
    """Test that reading from json molecule retains the order of json geometry and stereochemistry"""

    json_mol = {'symbols': ['C', 'C', 'N', 'O', 'F', 'H', 'H', 'H', 'H', 'H', 'H'],
                'geometry': [1.490934395068127, -0.022852472359013117, -1.935709059338355,
                             -0.07992863034848685, -0.42027454585371643, 0.4300901370510521,
                             -1.6008431326210255,  1.7962788702240675,  0.9893952378782299,
                             -1.578310435156546,  -2.623152319435938, 0.12587101271275358,
                              1.5081897367264838, -0.8595839767115931, 2.4023274238804375,
                              2.643487029125874,  -1.686714912858618,  -2.3700985298604698,
                              0.29985967115960716, 0.42241227312506313, -3.568237727722486,
                              2.7917672897488948,  1.5663042901906687, -1.6694857028577224,
                             -0.4416762043595982,  3.317083889862761,  1.2129328698056736,
                             -2.732926456425621,   2.1997415241410825, -0.5153340816908529,
                             -2.648885919666481,  -2.3294408246718734, -1.337378806095166],
                'molecular_charge': 0,
                'molecular_multiplicity': 1,
                'connectivity': [[0, 1, 1],
                  [1, 2, 1],
                  [1, 3, 1],
                  [1, 4, 1],
                  [0, 5, 1],
                  [0, 6, 1],
                  [0, 7, 1],
                  [2, 8, 1],
                  [2, 9, 1],
                  [3, 10, 1]]
                }

    mol_id = cmiles.to_molecule_id(json_mol)

    assert mol_id['canonical_smiles'] == 'CC(N)(O)F'
    assert mol_id['canonical_isomeric_smiles'] == 'C[C@@](N)(O)F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]C([H])([H])[C@@](N([H])[H])(O[H])F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:6][C:1]([H:7])([H:8])[C@@:2]([N:3]([H:9])[H:10])([O:4][H:11])[F:5]'

    # generate rd canonicalized smiles - the order should still be as before even though that is not the rdkit canonical
    # order. We want to retain the order for json molecules to their geometry
    mol_id = cmiles.to_molecule_id(json_mol, canonicalization='rdkit')
    assert mol_id['canonical_smiles'] == 'CC(N)(O)F'
    assert mol_id['canonical_isomeric_smiles'] == 'C[C@@](N)(O)F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_smiles'] == '[H][O][C@]([F])([N]([H])[H])[C]([H])([H])[H]'
    assert mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[C:1]([C@@:2]([N:3]([H:9])[H:10])([O:4][H:11])[F:5])([H:6])([H:7])[H:8]'

    # Now the other stereoisomer
    json_mol = {'symbols': ['C', 'C', 'N', 'O', 'F', 'H', 'H', 'H', 'H', 'H', 'H'],
                'geometry': [1.490934395068127, -0.022852472359013117, -1.935709059338355,
                             -0.07992863034848685, -0.42027454585371643, 0.4300901370510521,
                             -1.6008431326210255, 1.7962788702240675, 0.9893952378782299,
                              1.544484919393002,  -1.0715460728389934, 2.461713642916755,
                             -1.6405346423022924, -2.4261921600567007, 0.04846706513552157,
                              2.643487029125874,  -1.686714912858618,  -2.3700985298604698,
                              0.29985967115960716, 0.42241227312506313, -3.568237727722486,
                              2.7917672897488948,  1.5663042901906687, -1.6694857028577224,
                              -0.4416762043595982, 3.317083889862761, 1.2129328698056736,
                               -2.732926456425621, 2.1997415241410825, -0.5153340816908529,
                                2.4021616230193055, -2.619467530461027, 1.9699541458951846],
                'molecular_charge': 0,
                'molecular_multiplicity': 1,
                'connectivity': [[0, 1, 1],
                [1, 2, 1],
                [1, 3, 1],
                [1, 4, 1],
                [0, 5, 1],
                [0, 6, 1],
                [0, 7, 1],
                [2, 8, 1],
                [2, 9, 1],
                [3, 10, 1]],
                }
    mol_id = cmiles.to_molecule_id(json_mol)

    assert mol_id['canonical_smiles'] == 'CC(N)(O)F'
    assert mol_id['canonical_isomeric_smiles'] == 'C[C@](N)(O)F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]C([H])([H])[C@](N([H])[H])(O[H])F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:6][C:1]([H:7])([H:8])[C@:2]([N:3]([H:9])[H:10])([O:4][H:11])[F:5]'

    # generate rd canonicalized smiles - the order should still be as before even though that is not the rdkit canonical
    # order. We want to retain the order for json molecules to their geometry
    mol_id = cmiles.to_molecule_id(json_mol, canonicalization='rdkit')
    assert mol_id['canonical_smiles'] == 'CC(N)(O)F'
    assert mol_id['canonical_isomeric_smiles'] == 'C[C@](N)(O)F'
    assert mol_id['canonical_isomeric_explicit_hydrogen_smiles'] == '[H][O][C@@]([F])([N]([H])[H])[C]([H])([H])[H]'
    assert mol_id['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[C:1]([C@:2]([N:3]([H:9])[H:10])([O:4][H:11])[F:5])([H:6])([H:7])[H:8]'

@using_openeye
@using_rdkit
def test_bond_stereo():
    """Test bond steroe from json molecule"""
    json_mol_from_oe_map = {'symbols': ['C', 'C', 'F', 'Cl', 'H', 'H'],
                'geometry': [0.7558174176630313,
                              -0.9436196701031863,
                              -0.5135812331847833,
                              -0.7123369866046005,
                              0.7689600644555532,
                              0.5875385545305212,
                              1.2485802802219408,
                              -3.180729126504143,
                              0.5903747404566769,
                              -1.3805989906253051,
                              3.6349234648338813,
                              -0.7522673418877901,
                              1.6921967038297914,
                              -0.786834118158881,
                              -2.319716469002742,
                              -1.6036583681666305,
                              0.5072991602038667,
                              2.4076517490881173],
                'molecular_charge': 0,
                'molecular_multiplicity': 1,
                'connectivity': [[0, 1, 2], [0, 2, 1], [1, 3, 1], [0, 4, 1], [1, 5, 1]]}

    json_mol_from_rd_map = {'symbols': ['H', 'H', 'F', 'Cl', 'C', 'C'],
                            'geometry': [1.6921967038297914,
                            -0.786834118158881,
                            -2.319716469002742,
                            -1.6036583681666305,
                            0.5072991602038667,
                            2.4076517490881173,
                            1.2485802802219408,
                            -3.180729126504143,
                            0.5903747404566769,
                            -1.3805989906253051,
                            3.6349234648338813,
                            -0.7522673418877901,
                            0.7558174176630313,
                            -0.9436196701031863,
                            -0.5135812331847833,
                            -0.7123369866046005,
                            0.7689600644555532,
                            0.5875385545305212],
                            'molecular_charge': 0,
                            'molecular_multiplicity': 1,
                            'connectivity': [[4, 5, 2], [4, 2, 1], [5, 3, 1], [4, 0, 1], [5, 1, 1]],}

    id_oe_to_oe = cmiles.to_molecule_id(json_mol_from_oe_map, canonicalization='openeye')
    id_oe_to_rd = cmiles.to_molecule_id(json_mol_from_oe_map, canonicalization='rdkit')
    id_rd_to_oe = cmiles.to_molecule_id(json_mol_from_rd_map, canonicalization='openeye')
    id_rd_to_rd = cmiles.to_molecule_id(json_mol_from_rd_map, canonicalization='rdkit')

    assert id_oe_to_oe['canonical_smiles'] == id_rd_to_oe['canonical_smiles'] == 'C(=CCl)F'
    assert id_oe_to_oe['canonical_isomeric_smiles'] == id_rd_to_oe['canonical_isomeric_smiles'] == 'C(=C/Cl)\\F'
    assert id_oe_to_oe['canonical_explicit_hydrogen_smiles'] == id_rd_to_oe['canonical_explicit_hydrogen_smiles'] == '[H]C(=C([H])Cl)F'
    assert id_oe_to_oe['canonical_isomeric_explicit_hydrogen_smiles'] == id_rd_to_oe['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]/C(=C(/[H])\\Cl)/F'
    assert id_oe_to_oe['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:5]/[C:1](=[C:2](/[H:6])\\[Cl:4])/[F:3]'
    assert id_rd_to_oe['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:1]/[C:5](=[C:6](/[H:2])\\[Cl:4])/[F:3]'

    assert id_oe_to_rd['canonical_smiles'] == id_rd_to_rd['canonical_smiles'] == 'FC=CCl'
    assert id_oe_to_rd['canonical_isomeric_smiles'] == id_rd_to_rd['canonical_isomeric_smiles'] == 'F/C=C/Cl'
    assert id_oe_to_rd['canonical_explicit_hydrogen_smiles'] == id_rd_to_rd['canonical_explicit_hydrogen_smiles'] == '[H][C]([F])=[C]([H])[Cl]'
    assert id_oe_to_rd['canonical_isomeric_explicit_hydrogen_smiles'] == id_rd_to_rd['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]/[C]([F])=[C](/[H])[Cl]'
    assert id_oe_to_rd['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[C:1](=[C:2](/[Cl:4])[H:6])(\\[F:3])[H:5]'
    assert id_rd_to_rd['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:1]/[C:5]([F:3])=[C:6](/[H:2])[Cl:4]'

    # Now the other stereoisomer
    json_mol_from_oe_map = {'symbols': ['C', 'C', 'F', 'Cl', 'H', 'H'],
                 'geometry': [0.7558174176630313,
                  -0.9436196701031863,
                  -0.5135812331847833,
                  -0.7123369866046005,
                  0.7689600644555532,
                  0.5875385545305212,
                  1.2485802802219408,
                  -3.180729126504143,
                  0.5903747404566769,
                  -2.098028327659392,
                  0.3039736895037833,
                  3.4718143896933764,
                  1.6921967038297914,
                  -0.786834118158881,
                  -2.319716469002742,
                  -1.1578923904597032,
                  2.5868975732566115,
                  -0.2324103574431031],
                 'molecular_charge': 0,
                 'molecular_multiplicity': 1,
                 'connectivity': [[0, 1, 2], [0, 2, 1], [1, 3, 1], [0, 4, 1], [1, 5, 1]]
                }
    json_mol_from_rd_map = {'symbols': ['H', 'H', 'F', 'Cl', 'C', 'C'],
                            'geometry': [1.6921967038297914,
                            -0.786834118158881,
                            -2.319716469002742,
                            -1.1578923904597032,
                            2.5868975732566115,
                            -0.2324103574431031,
                            1.2485802802219408,
                            -3.180729126504143,
                            0.5903747404566769,
                            -2.098028327659392,
                            0.3039736895037833,
                            3.4718143896933764,
                            0.7558174176630313,
                            -0.9436196701031863,
                            -0.5135812331847833,
                            -0.7123369866046005,
                            0.7689600644555532,
                            0.5875385545305212],
                            'molecular_charge': 0,
                            'molecular_multiplicity': 1,
                            'connectivity': [[4, 5, 2], [4, 2, 1], [5, 3, 1], [4, 0, 1], [5, 1, 1]]}

    id_oe_to_oe = cmiles.to_molecule_id(json_mol_from_oe_map, canonicalization='openeye')
    id_oe_to_rd = cmiles.to_molecule_id(json_mol_from_oe_map, canonicalization='rdkit')
    id_rd_to_oe = cmiles.to_molecule_id(json_mol_from_rd_map, canonicalization='openeye')
    id_rd_to_rd = cmiles.to_molecule_id(json_mol_from_rd_map, canonicalization='rdkit')

    assert id_oe_to_oe['canonical_smiles'] == id_rd_to_oe['canonical_smiles'] == 'C(=CCl)F'
    assert id_oe_to_oe['canonical_isomeric_smiles'] == id_rd_to_oe['canonical_isomeric_smiles'] == 'C(=C\\Cl)\\F'
    assert id_oe_to_oe['canonical_explicit_hydrogen_smiles'] == id_rd_to_oe['canonical_explicit_hydrogen_smiles'] == '[H]C(=C([H])Cl)F'
    assert id_oe_to_oe['canonical_isomeric_explicit_hydrogen_smiles'] == id_rd_to_oe['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]/C(=C(\\[H])/Cl)/F'
    assert id_oe_to_oe['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:5]/[C:1](=[C:2](\\[H:6])/[Cl:4])/[F:3]'
    assert id_rd_to_oe['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:1]/[C:5](=[C:6](\\[H:2])/[Cl:4])/[F:3]'

    assert id_oe_to_rd['canonical_smiles'] == id_rd_to_rd['canonical_smiles'] == 'FC=CCl'
    assert id_oe_to_rd['canonical_isomeric_smiles'] == id_rd_to_rd['canonical_isomeric_smiles'] == 'F/C=C\\Cl'
    assert id_oe_to_rd['canonical_explicit_hydrogen_smiles'] == id_rd_to_rd['canonical_explicit_hydrogen_smiles'] == '[H][C]([F])=[C]([H])[Cl]'
    assert id_oe_to_rd['canonical_isomeric_explicit_hydrogen_smiles'] == id_rd_to_rd['canonical_isomeric_explicit_hydrogen_smiles'] == '[H]/[C]([F])=[C](\\[H])[Cl]'
    assert id_oe_to_rd['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[C:1](=[C:2](\\[Cl:4])[H:6])(\\[F:3])[H:5]'
    assert id_rd_to_rd['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == '[H:1]/[C:5]([F:3])=[C:6](\\[H:2])[Cl:4]'


@using_openeye
def _chemical_formula_oe(smiles):
    from openeye import oechem
    molecule = cmiles.utils.load_molecule(smiles, backend='openeye')
    oechem.OEAddExplicitHydrogens(molecule)
    return (cmiles.generator.molecular_formula(molecule))

@using_rdkit
def _chemical_formula_rd(smiles):
    from rdkit import Chem
    molecule = cmiles.utils.load_molecule(smiles, backend='rdkit')
    molecule = Chem.AddHs(molecule)
    return (cmiles.generator.molecular_formula(molecule))

@using_rdkit
@using_openeye
@pytest.mark.parametrize("backend", [_chemical_formula_oe, _chemical_formula_rd])
@pytest.mark.parametrize("smiles, bench", [("CCCC", "C4H10"), ("C", "CH4"), ("CC(Br)(Br)", "C2H4Br2")]) # This uncovered a bug in load_molecule. The period makes it look like a filename
                                                               #  ("[Li+].[Li+].[O2-]", "LiO2")]) ToDo fix this bug
def test_molecule_formula(backend, smiles, bench):
    assert backend(smiles) == bench


def _test_unique_protomer(state1, state2):
    """Test unique protomer"""
    mol_1 = cmiles.utils.load_molecule(state1)
    mol_2 = cmiles.utils.load_molecule(state2)
    assert cmiles.get_unique_protomer(mol_1) == cmiles.get_unique_protomer(mol_2)


def _test_unique_tautomer(state1, state2):
    """Test standardize tautomer rdkit"""
    assert cmiles.standardize_tautomer(state1) == cmiles.standardize_tautomer(state2)

@using_rdkit
@using_openeye
@pytest.mark.parametrize('state1, state2',
                          [
                           ('CN4CCN(c2nc1cc(Cl)ccc1[nH]c3ccccc23)CC4', 'CN4CCN(c2[nH]c1cc(Cl)ccc1nc3ccccc23)CC4'), # 1,5 tautomerism
                           ('CC=CC(=O)C', 'CC(=CC=C)O'), #keto-enol
                           ('NC=O', 'N=CO'), # amide - imidic
                           ('C1CC(=O)NC1', 'C1CC(O)=NC1'), # lactam-lactim

                           ('C1=CN=CNC1=O', 'C1=CNC=NC1=O'), # pyrimodone
                           ('C1=CN=CNC1=O', 'C1=CN=CN=C1O'),# pyrimodol - pytimodone
                           ('C1=NC=NN1','N1C=NN=C1' ), #triazole
                           ('C1(=NC(=NC(=N1)O)O)O', 'C1(NC(NC(N1)=O)=O)=O'), # cyanic - cyanuric
                           ('CC(=O)S[H]', 'CC(=S)O[H]'),
                           ('C[N](=O)=O', 'C[N+]([O-])=O'),# nitromethane
                           ('CN=[N]#N', 'CN=[N+]=[N-]'),  # azide
                           ('C[S](C)(=O)=O', 'C[S++](C)([O-])[O-]'), # sulfone
                           ('C[P](C)(=O)[O-]', 'C[P+](C)([O-])[O-]'), # phosphinate
                           ('OC=C', 'O=CC'), # keto-enol
                           ('[H]NC=C', '[H]N=CC'), # imine - enamine
                           ('CN=O', 'C=NO'), # nitoroso-oxime
                           ('C1=CC(NN1)=O','C1=NNC(C1)=O' ),# pyrazolone
                           ('C1=CC(NN1)=O', 'C1(=CC=N[N]1[H])O'), # pyrazolone
                           ('C1(=NC=CNC=C1)C', 'C1(=CC=NC=CN1)C'),# diazepine
                           ('C1=CC=C(O1)O','C1=CCC(O1)=O' ), #furanol-furanone
                           ('C1=CC(=CO1)O', 'C1=CC(CO1)=O')#  furanol - furanone
                           ])
@pytest.mark.parametrize('backend', [_test_unique_protomer, _test_unique_tautomer])
def test_universal_id(backend, state1, state2):
    backend(state1, state2)


# only captured with openeye - unique protomer also included ionization states. rdkit only standardizes for tautomers.
@using_openeye
@pytest.mark.parametrize("state1, state2",
                         [('CC(=O)O','CC(=O)[O-]'), # protonation states
                          ('NCC(=O)O', '[NH3+]CC(=O)[O-]'),# amino acid
                          ('CC(=O)S[H]', 'CC(=O)[S-]'),
                          ('CC(=O)S[H]', 'CC(=S)[O-]'),
                         ])
def test_unique_protomers(state1, state2):
    _test_unique_protomer(state1, state2)

# only captured with redkit. captures indoles. Neither openeye nor rdkit capture isoindoles
@using_rdkit
@pytest.mark.parametrize("state1, state2",
                         [('CP(=O)(C)S', 'CP(=S)(C)O'), # mesomer
                          ('CC(=[NH2+])N','C[C+](N)N'), # mesomer
                          ('C=C=O', 'C#CO'), # keten-ynol
                          ('c1ccc2c(c1)cc([nH]2)O', 'c1ccc2c(c1)CC(=O)N2'), # indole_ol - indole-one
                          ('c1ccc2c(c1)cc([nH]2)N', 'c1ccc2c(c1)CC(=N2)N')# indole - indole
                         ])
def test_unique_tautomers(state1, state2):
    _test_unique_tautomer(state1, state2)