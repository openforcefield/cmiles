"""
Unit and regression test for the cmiles package.
"""

# Import package, test suite, and other packages as needed
import cmiles
import pytest
import sys

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
        "CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[CH]4CCOC4"
    ])

@pytest.fixture
def rd_iso_expected():
    return(["CO[C@@H](C)c1c(Cl)ccc(F)c1Cl",
    "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1"
    ])

@pytest.fixture
def rd_map_expected():
    return (['[H:1][C:12]([H:2])([O:9][C@@:18]([H:4])([C:19]([H:5])([H:6])[H:7])[c:20]1[c:16]([Cl:17])[c:15]([F:10])[c:14]([H:3])[c:22]([H:8])[c:21]1[Cl:11])[H:13]',
             '[H:1][C@@:19]1([O:17][c:35]2[c:36]([H:16])[c:37]3[c:38]([c:39]([H:15])[c:49]2[N:45]([H:12])[C:46](=[O:47])/[C:50]([H:21])=[C:52](\\[H:10])[C:51]([H:11])([H:23])[N:43]([C:44]([H:5])([H:18])[H:25])[C:48]([H:22])([H:24])[H:26])[c:40]([N:57]([H:14])[c:58]2[c:8]([H:2])[c:53]([H:20])[c:55]([F:27])[c:54]([Cl:28])[c:59]2[H:9])[n:41][c:42]([H:13])[n:56]3)[C:6]([H:30])([H:31])[O:7][C:29]([H:32])([H:34])[C:33]1([H:3])[H:4]'
    ])

@pytest.fixture
def rd_can_expected():
    return (['COC(C)c1c(Cl)ccc(F)c1Cl',
             'CN(C)CC=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC1CCOC1'
    ])

@pytest.fixture
def rd_h_expected():
    return (['[H][c]1[c]([H])[c]([Cl])[c]([C@@]([H])([O][C]([H])([H])[H])[C]([H])([H])[H])[c]([Cl])[c]1[F]',
             '[H]/[C]([C](=[O])[N]([H])[c]1[c]([O][C@]2([H])[C]([H])([H])[O][C]([H])([H])[C]2([H])[H])[c]([H])[c]2[n][c]([H])[n][c]([N]([H])[c]3[c]([H])[c]([H])[c]([F])[c]([Cl])[c]3[H])[c]2[c]1[H])=[C](/[H])[C]([H])([H])[N]([C]([H])([H])[H])[C]([H])([H])[H]'
    ])
@pytest.fixture
def oe_iso_expected():
    return([
    ])

@pytest.fixture
def oe_map_expected():
    return ([])

@pytest.fixture
def oe_can_expected():
    return ([
    ])

@pytest.fixture
def oe_h_expected():
    return ([])

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

@pytest.mark.parametrize("input, oe_iso_expected", [
    ("CC(c1c(ccc(c1Cl)F)Cl)OC",
     "C[C@@H](c1c(ccc(c1Cl)F)Cl)OC"),
    ("CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[CH]4CCOC4",
     "CN(C)C/C=C/C(=O)Nc1cc2c(cc1O[C@@H]3CCOC3)ncnc2Nc4ccc(c(c4)Cl)F")
])
@pytest.mark.skipif(openeye_missing, reason="Cannot test without openeye")
def test_oe_isomeric(input, oe_iso_expected):
    """testing openeye isomeric smiles"""
    oemol = oechem.OEMol()
    oechem.OEParseSmiles(oemol, input)
    assert cmiles.to_canonical_smiles_oe(oemol, isomeric=True, explicit_hydrogen=False, mapped=False) == oe_iso_expected

def test_oe_version():
    pass

def test_rd_version():
    pass

def test_oe_mol():
    pass

def test_rd_mol():
    pass

def test_oe_cmiles():
    pass

def test_rd_cmiles():
    pass

def test_warnings():
    pass
