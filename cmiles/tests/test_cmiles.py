"""
Unit and regression test for the cmiles package.
"""

# Import package, test suite, and other packages as needed
import cmiles
import pytest
import sys
from .utils import get_fn, get_smiles_lists

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


def test_cmiles_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "cmiles" in sys.modules

@pytest.fixture
def smiles_input():
    return([
        "CC(c1c(ccc(c1Cl)F)Cl)OC",
        "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[CH]4CCOC4",
        "NC(C)(F)C(=O)O",
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
    return (['[H:1][c:14]1[c:15]([H:2])[c:17]([Cl:11])[c:19]([C@@:22]([H:9])([O:13][C:20]([H:3])([H:4])[H:5])[C:21]([H:6])([H:7])[H:8])[c:18]([Cl:12])[c:16]1[F:10]',
             '[H:1]/[C:34]([C:33](=[O:26])[N:50]([H:9])[c:43]1[c:42]([O:32][C@:59]2([H:25])[C:57]([H:21])([H:22])[O:31][C:56]([H:19])([H:20])[C:58]2([H:23])[H:24])[c:45]([H:7])[c:48]2[n:30][c:36]([H:3])[n:29][c:46]([N:51]([H:10])[c:44]3[c:38]([H:5])[c:37]([H:4])[c:40]([F:27])[c:41]([Cl:28])[c:39]3[H:6])[c:49]2[c:47]1[H:8])=[C:35](/[H:2])[C:55]([H:17])([H:18])[N:52]([C:53]([H:11])([H:12])[H:13])[C:54]([H:14])([H:15])[H:16]',
             '[H:1][O:9][C:10](=[O:7])[C@:13]([F:8])([N:11]([H:2])[H:3])[C:12]([H:4])([H:5])[H:6]',
             '[H:1][C:11]1=[C:12]([H:2])[C:19]([H:7])([H:8])[C:10](=[O:9])[c:17]2[c:15]([H:5])[c:13]([H:3])[c:14]([H:4])[c:16]([H:6])[c:18]21',
             '[H:1][O:9][c:17]1[c:13]([H:5])[c:11]([H:3])[c:15]([H:7])[c:18]2[c:14]([H:6])[c:10]([H:2])[c:12]([H:4])[c:16]([H:8])[c:19]12',
             '[H:1][C:7]#[N+:8][C:10]([H:5])([H:6])[C:9]([H:2])([H:3])[H:4]',
             '[H:1][c:9]1[c:10]([H:2])[c:12]([H:4])[c:14]([C:15]([H:6])([H:7])[H:8])[c:13]([H:5])[c:11]1[H:3]'
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
            'CN(C)C/C=C/C(=O)Nc1cc2c(cc1O[C@@H]3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F',
            'C[C@](C(=O)O)(N)F',
            'c1ccc2c(c1)C=CCC2=O',
            'c1ccc2c(c1)cccc2O',
            'CC[N+]#C',
            'Cc1ccccc1'
    ])

@pytest.fixture
def oe_map_expected():
    return (['[H:14][c:1]1[c:2]([c:5]([c:3]([c:6]([c:4]1[F:11])[Cl:13])[C@:9]([H:22])([C:7]([H:16])([H:17])[H:18])[O:10][C:8]([H:19])([H:20])[H:21])[Cl:12])[H:15]',
             '[H:35][c:1]1[c:2]([c:12]([c:13]([c:5]([c:9]1[N:27]([H:58])[c:14]2[c:7]3[c:3]([c:10]([c:11]([c:4]([c:8]3[n:25][c:6]([n:26]2)[H:40])[H:38])[O:32][C@@:21]4([C:18]([C:19]([O:31][C:20]4([H:47])[H:48])([H:45])[H:46])([H:43])[H:44])[H:49])[N:28]([H:59])[C:17](=[O:30])/[C:15](=[C:16](\\[H:42])/[C:24]([H:56])([H:57])[N:29]([C:22]([H:50])([H:51])[H:52])[C:23]([H:53])([H:54])[H:55])/[H:41])[H:37])[H:39])[Cl:34])[F:33])[H:36]',
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
             '[H]c1c(c(c(c(c1N([H])c2c3c(c(c(c(c3nc(n2)[H])[H])O[C@@]4(C(C(OC4([H])[H])([H])[H])([H])[H])[H])N([H])C(=O)/C(=C(\\[H])/C([H])([H])N(C([H])([H])[H])C([H])([H])[H])/[H])[H])[H])Cl)F)[H]',
             '[H]C([H])([H])[C@](C(=O)O[H])(N([H])[H])F',
             '[H]c1c(c(c2c(c1[H])C(=C(C(C2=O)([H])[H])[H])[H])[H])[H]',
             '[H]c1c(c(c2c(c1[H])c(c(c(c2O[H])[H])[H])[H])[H])[H]',
             '[H]C#[N+]C([H])([H])C([H])([H])[H]',
             '[H]c1c(c(c(c(c1[H])[H])C([H])([H])[H])[H])[H]'])


@pytest.mark.skipif(rdkit_missing, reason="Cannot test without RDKit")
def test_rdkit_isomeric(smiles_input, rd_iso_expected):
    """testing rdkit isomeric canonical smiles"""
    for i, o in zip(smiles_input, rd_iso_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=False, explicit_hydrogen=False) == o


def test_rdkit_map(smiles_input, rd_map_expected):
    """Testing rdkit canonical map ordering"""
    for i, o in zip(smiles_input, rd_map_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=True, explicit_hydrogen=True) == o


def test_rdkit_canonical(smiles_input, rd_can_expected):
    """Testing rdkit canonical smiles"""
    for i, o in zip(smiles_input, rd_can_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=False, mapped=False, explicit_hydrogen=False) == o


def test_rdkit_explicit_h(smiles_input, rd_h_expected):
    """Testing rdkit explicit hydrogen"""
    for i, o in zip(smiles_input, rd_h_expected):
        rd_mol = Chem.MolFromSmiles(i)
        assert cmiles.to_canonical_smiles_rd(rd_mol, isomeric=True, mapped=False, explicit_hydrogen=True) == o


def test_openeye_explicit_h(smiles_input, oe_h_expected):
    """Testing openeye explicit hydrogen"""
    for i, o in zip(smiles_input, oe_h_expected):
        oe_mol = oechem.OEMol()
        oechem.OEParseSmiles(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=False, explicit_hydrogen=True) == o


def test_oepneye_isomeric(smiles_input, oe_iso_expected):
    """testing rdkit isomeric canonical smiles"""
    for i, o in zip(smiles_input, oe_iso_expected):
        oe_mol = oechem.OEMol()
        oechem.OEParseSmiles(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=False, explicit_hydrogen=False) == o


def test_openeye_map(smiles_input, oe_map_expected):
    """Testing rdkit canonical map ordering"""
    for i, o in zip(smiles_input, oe_map_expected):
        oe_mol = oechem.OEMol()
        oechem.OEParseSmiles(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=True, mapped=True, explicit_hydrogen=True) == o


def test_openeye_canonical(smiles_input, oe_can_expected):
    """Testing rdkit canonical smiles"""
    for i, o in zip(smiles_input, oe_can_expected):
        oe_mol = oechem.OEMol()
        oechem.OEParseSmiles(oe_mol, i)
        assert cmiles.to_canonical_smiles_oe(oe_mol, isomeric=False, mapped=False, explicit_hydrogen=False) == o


def test_oe_mol():
    """test warning when converting rdmol to oemol"""
    molecule = cmiles.rd.Chem.MolFromSmiles('CC(c1c(ccc(c1Cl)F)Cl)OC')
    with pytest.warns(UserWarning):
        cmiles.to_canonical_smiles(molecule, canonicalization='openeye')


def test_rd_mol():
    """test warning when converting oemol to rdmol"""
    molecule = oechem.OEMol()
    oechem.OEParseSmiles(molecule, 'CC(c1c(ccc(c1Cl)F)Cl)OC')
    with pytest.warns(UserWarning):
        cmiles.to_canonical_smiles(molecule, canonicalization='rdkit')
    pass


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
    molecule = oechem.OEMol()
    oechem.OEParseSmiles(molecule, 'CN(C)CC=CC(=O)Nc1cc2c(cc1OC3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F')
    output = cmiles.to_canonical_smiles(molecule, canonicalization='openeye')
    assert expected_output['canonical_smiles'] == output['canonical_smiles']
    assert expected_output['canonical_isomeric_smiles'] == output['canonical_isomeric_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_smiles'] == output['canonical_isomeric_explicit_hydrogen_smiles']
    assert expected_output['canonical_explicit_hydrogen_smiles'] == output['canonical_explicit_hydrogen_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == output['canonical_isomeric_explicit_hydrogen_mapped_smiles']


def test_rd_cmiles():
    """Regression test rdkit cmiles"""
    expected_output = {'canonical_smiles': 'CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1',
 'canonical_isomeric_smiles': 'CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1',
 'canonical_isomeric_explicit_hydrogen_smiles': '[H]/[C]([C](=[O])[N]([H])[c]1[c]([O][C@]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C](/[H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]',
 'canonical_explicit_hydrogen_smiles': '[H][C]([C](=[O])[N]([H])[c]1[c]([O][C]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C]([H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]',
 'canonical_isomeric_explicit_hydrogen_mapped_smiles': '[H:1]/[C:34]([C:33](=[O:26])[N:50]([H:9])[c:43]1[c:42]([O:32][C@:59]2([H:25])[C:57]([H:21])([H:22])[O:31][C:56]([H:19])([H:20])[C:58]2([H:23])[H:24])[c:45]([H:7])[c:48]2[n:30][c:36]([H:3])[n:29][c:46]([N:51]([H:10])[c:44]3[c:38]([H:5])[c:37]([H:4])[c:40]([F:27])[c:41]([Cl:28])[c:39]3[H:6])[c:49]2[c:47]1[H:8])=[C:35](/[H:2])[C:55]([H:17])([H:18])[N:52]([C:53]([H:11])([H:12])[H:13])[C:54]([H:14])([H:15])[H:16]',
 'provenance': 'cmiles_0.0.0+7.gc71f3a6.dirty_rdkit_2018.03.3'}
    molecule = cmiles.rd.Chem.MolFromSmiles('CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1')
    output = cmiles.to_canonical_smiles(molecule, canonicalization='rdkit')
    assert expected_output['canonical_smiles'] == output['canonical_smiles']
    assert expected_output['canonical_isomeric_smiles'] == output['canonical_isomeric_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_smiles'] == output['canonical_isomeric_explicit_hydrogen_smiles']
    assert expected_output['canonical_explicit_hydrogen_smiles'] == output['canonical_explicit_hydrogen_smiles']
    assert expected_output['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == output['canonical_isomeric_explicit_hydrogen_mapped_smiles']


def test_diff_smiles():
    """Test different SMILES of same molecule"""
    input = ['CC(c1c(ccc(c1Cl)F)Cl)OC', 'COC(C)c1c(Cl)ccc(F)c1Cl']
    oe_mol_1 = oechem.OEMol()
    oechem.OEParseSmiles(oe_mol_1, input[0])
    oe_mol_2 = oechem.OEMol()
    oechem.OEParseSmiles(oe_mol_2, input[-1])
    cmiles_1 = cmiles.to_canonical_smiles(oe_mol_1)
    cmiles_2 = cmiles.to_canonical_smiles(oe_mol_2)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] == cmiles_2['canonical_isomeric_smiles']
    assert cmiles_1['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == cmiles_2['canonical_isomeric_explicit_hydrogen_mapped_smiles']

    rd_mol_1 = cmiles.rd.Chem.MolFromSmiles(input[0])
    rd_mol_2 = cmiles.rd.Chem.MolFromSmiles(input[-1])
    cmiles_1 = cmiles.to_canonical_smiles(rd_mol_1, canonicalization='rdkit')
    cmiles_2 = cmiles.to_canonical_smiles(rd_mol_2, canonicalization='rdkit')
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] == cmiles_2['canonical_isomeric_smiles']
    assert cmiles_1['canonical_isomeric_explicit_hydrogen_mapped_smiles'] == cmiles_2['canonical_isomeric_explicit_hydrogen_mapped_smiles']


def test_initial_iso():
    """test given chirality"""

    input = ["CC[C@@H](C)N",  "CC[C@H](C)N"]
    oe_mol_1 = oechem.OEMol()
    oechem.OEParseSmiles(oe_mol_1, input[0])
    oe_mol_2 = oechem.OEMol()
    oechem.OEParseSmiles(oe_mol_2, input[-1])
    cmiles_1 = cmiles.to_canonical_smiles(oe_mol_1)
    cmiles_2 = cmiles.to_canonical_smiles(oe_mol_2)
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] != cmiles_2['canonical_isomeric_smiles']

    rd_mol_1 = cmiles.rd.Chem.MolFromSmiles(input[0])
    rd_mol_2 = cmiles.rd.Chem.MolFromSmiles(input[-1])
    cmiles_1 = cmiles.to_canonical_smiles(rd_mol_1, canonicalization='rdkit')
    cmiles_2 = cmiles.to_canonical_smiles(rd_mol_2, canonicalization='rdkit')
    assert cmiles_1['canonical_smiles'] == cmiles_2['canonical_smiles']
    assert cmiles_1['canonical_isomeric_smiles'] != cmiles_2['canonical_isomeric_smiles']


@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_sm.smi'), get_fn('drug_bank_mapped_smi_rd.smi')))
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


@pytest.mark.parametrize("input, output", get_smiles_lists(get_fn('drug_bank_sm.smi'), get_fn('drug_bank_mapped_smi_oe.smi')))
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
    assert cmiles.generator.to_canonical_smiles_oe(mol, mapped=True, isomeric=True, explicit_hydrogen=True,
                                                   generate_conformer=False) == output