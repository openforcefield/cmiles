"""Test util functions"""

from cmiles import utils
import pytest
import numpy as np

toolkits = list()
toolkits_name = list()
if utils.has_rdkit:
    from cmiles import _cmiles_rd
    toolkits.append(_cmiles_rd)
    toolkits_name.append('rdkit')
if utils.has_openeye:
    from cmiles import _cmiles_oe
    toolkits.append(_cmiles_oe)
    toolkits_name.append('openeye')

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

using_rdkit = pytest.mark.skipif(not utils.has_rdkit, reason="Cannot run without RDKit")
using_openeye = pytest.mark.skipif(not utils.has_openeye, reason="Cannot run without OpenEye")


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_load_molecule(toolkit):
    """Test load molecules"""
    mol = utils.load_molecule('[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]', toolkit=toolkit)
    if toolkit == 'openeye':
        from openeye import oechem
        assert oechem.OEMolToSmiles(mol) == 'CCCC'
    if toolkit == 'rdkit':
        from rdkit import Chem
        assert Chem.MolToSmiles(mol) == 'CCCC'


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_is_mapped(toolkit):
    """Test is mapped"""
    mapped_smiles = '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]'
    mapped_mol = utils.load_molecule(mapped_smiles, toolkit=toolkit)
    assert utils.has_atom_map(mapped_mol) == True
    utils.remove_atom_map(mapped_mol)
    assert utils.has_atom_map(mapped_mol) == False


@pytest.mark.parametrize('toolkit_str', toolkits_name)
def test_mol_from_json(toolkit_str):
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
    mol = utils.load_molecule(hooh, toolkit=toolkit_str)
    if toolkit_str == 'openeye':
        assert mol.GetMaxAtomIdx() == 4
        assert mol.GetMaxBondIdx() == 3
        coordinates = mol.GetCoords()
    if toolkit_str == 'rdkit':
        assert mol.GetNumAtoms() == 4
        assert mol.GetNumBonds() == 3
        coordinates = mol.GetConformer().GetPositions()
    geometry = np.array(hooh['geometry'], dtype=float).reshape(int(len(hooh['geometry'])/3), 3)*utils.BOHR_2_ANGSTROM
    for i in range(len(coordinates)):
        for j in range(3):
            assert coordinates[i][j] == pytest.approx(geometry[i][j], 0.0000001)


@pytest.mark.parametrize('toolkit_name', toolkits_name)
@pytest.mark.parametrize("input1, input2", [('C[C@](N)(O)F', 'CC(N)(O)F'),
                                   ('C(=C/Cl)\\F', 'C(=CCl)F'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)CC=CC(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4'),
                                   ('CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)O[C@H]4CCOC4',
                                   'CN(C)C/C=C/C(=O)NC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OC4CCOC4')])
def test_has_stereochemistry(input1, input2, toolkit_name):
    mol = utils.load_molecule(input1, toolkit_name)
    if toolkit_name == 'openeye':
        from openeye import oechem
        oechem.OEAddExplicitHydrogens(mol)
    if toolkit_name == 'rdkit':
        from rdkit import Chem
        mol = Chem.AddHs(mol)
    assert utils.has_stereo_defined(mol) == True

    mol = utils.load_molecule(input2, toolkit_name)
    if toolkit_name == 'openeye':
        from openeye import oechem
        oechem.OEAddExplicitHydrogens(mol)
    if toolkit_name == 'rdkit':
        from rdkit import Chem
        mol = Chem.AddHs(mol)
    with pytest.raises(ValueError):
        utils.has_stereo_defined(mol)

@using_openeye
def test_canonical_order_oe():
    """Test canonical atom order"""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, 'HOOH')

    # add map
    for atom in mol.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)

    assert oechem.OEMolToSmiles(mol) == '[H:1][O:2][O:3][H:4]'

    mol_2 = _cmiles_oe.canonical_order_atoms(mol, in_place=False)
    for atom in mol.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    for atom in mol_2.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    assert oechem.OEMolToSmiles(mol) == '[H:1][O:2][O:3][H:4]'
    assert oechem.OEMolToSmiles(mol_2) == '[H:3][O:1][O:2][H:4]'

    _cmiles_oe.canonical_order_atoms(mol, in_place=True)
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

    mol_2 = _cmiles_rd.canonical_order_atoms(mol, h_last=False)
    assert Chem.MolToSmiles(mol_2) == '[H:1][O:5][C:6]([H:2])([H:3])[H:4]'

    mol_3 = _cmiles_rd.canonical_order_atoms(mol)
    assert Chem.MolToSmiles(mol_3) == '[O:1]([C:2]([H:4])([H:5])[H:6])[H:3]'


@pytest.mark.parametrize('toolkit_name', toolkits_name)
@pytest.mark.parametrize('input, output', [('COC(C)c1c(Cl)ccc(F)c1Cl', False),
                                          ('C[C@@H](c1c(ccc(c1Cl)F)Cl)OC', False),
                                          ('O=O', True),
                                          ('[CH3:1][CH2:3][CH2:4][CH3:2]', False)])
def test_explicit_h(input, output, toolkit_name):
    """Test input SMILES for explicit H"""
    mol = utils.load_molecule(input, toolkit=toolkit_name)
    assert utils.has_explicit_hydrogen(mol) == output

@using_rdkit
@pytest.mark.parametrize('input, output', [('[H]c1c(c(c(c(c1F)Cl)[C@]([H])(C([H])([H])[H])OC([H])([H])[H])Cl)[H]', False),
                                          ('[C:1]([O:2][H:6])([H:3])([H:4])[H:5]', False)])
def test_explicit_h_rd(input, output):
    """Test input SMILES for explicit H"""

    mol = utils.load_molecule(input, toolkit='rdkit')
    assert utils.has_explicit_hydrogen(mol) == output

@using_openeye
@pytest.mark.parametrize('input, output', [('[H]c1c(c(c(c(c1F)Cl)[C@]([H])(C([H])([H])[H])OC([H])([H])[H])Cl)[H]', True),
                                          ('[C:1]([O:2][H:6])([H:3])([H:4])[H:5]', True)])
def test_explicit_h_oe(input, output):
    """Test input SMILES for explicit H"""

    mol = utils.load_molecule(input, toolkit='openeye')
    assert utils.has_explicit_hydrogen(mol) == output


@pytest.mark.parametrize('toolkit', toolkits_name)
@pytest.mark.parametrize("smiles", ['CN(C)C(=N)NC(=N)N', 'N=CO'])
def test_chiral_bond_exception(smiles, toolkit):
    """ Test bonds to ignore """
    mol = utils.load_molecule(smiles, toolkit)
    if toolkit == 'openeye':
        from openeye import oechem
        oechem.OEAddExplicitHydrogens(mol)
    if toolkit == 'rdkit':
        from rdkit import Chem
        mol = Chem.AddHs(mol)
    assert utils.has_stereo_defined(mol) == True


@pytest.mark.parametrize('toolkit, toolkit_name', list(zip(toolkits, toolkits_name)))
@pytest.mark.parametrize("smiles, output", [('CN(C)C(=N)NC(=N)N', True),
                                           ('COC1=CC(CN(C)C2=CC=C3N=C(N)N=C(N)C3=C2)=C(OC)C=C1', False),
                                           ('[H][C@](C)(O)[C@@]([H])(N=C(O)[C@]1([H])C[C@@]([H])(CCC)CN1C)[C@@]1([H])O[C@]([H])(SC)[C@]([H])(O)[C@@]([H])(O)[C@@]1([H])O',
                                            False),
                                            ('N=CO', True)])
def test_chiral_bond_exception_2(smiles, output, toolkit, toolkit_name):
    """ Test bonds to ignore """
    mol = utils.load_molecule(smiles, toolkit_name)
    if toolkit_name == 'openeye':
        oechem.OEAddExplicitHydrogens(mol)
    if toolkit_name == 'rdkit':
        mol = Chem.AddHs(mol)
    ignore = False
    for bond in mol.GetBonds():
        ignore = toolkit._ignore_stereo_flag(bond)
        if ignore:
            break
    assert ignore == output


@pytest.mark.parametrize('toolkit, toolkit_name', list(zip(toolkits, toolkits_name)))
def test_get_atom_map(toolkit, toolkit_name):
    smiles = 'C[C@@H](c1c(ccc(c1Cl)F)Cl)OC'
    mol = utils.load_molecule(smiles, toolkit_name)
    if toolkit_name == 'openeye':
        from openeye import oechem
        oechem.OEAddExplicitHydrogens(mol)
        for a in mol.GetAtoms():
            a.SetMapIdx(a.GetIdx()+1)
        mapped_smiles = oechem.OEMolToSmiles(mol)

    if toolkit_name == 'rdkit':
        from rdkit import Chem
        mol = Chem.AddHs(mol)
        for a in mol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx()+1)
        mapped_smiles = Chem.MolToSmiles(mol)

    atom_map = utils.get_atom_map(mol, mapped_smiles)

    for m in atom_map:
        assert m == (atom_map[m] + 1)


@pytest.mark.parametrize('mapped_smiles, expected_table', [('[H:5][C:1]([H:6])([H:7])[C:3]([H:11])([H:12])[C:4]([H:13])([H:14])[C:2]([H:8])([H:9])[H:10]',
                                                          np.array([[0, 2, 1], [1, 3, 1],[2, 3, 1],[0, 4, 1],[0, 5, 1],
                                                                    [0, 6, 1],[1, 7, 1],[1, 8, 1],[1, 9, 1],[2, 10, 1],
                                                                    [2, 11, 1],[3, 12, 1],[3, 13, 1]])),
                                                           ('[H:7][c:1]1[c:2]([c:4]([o:5][c:3]1[H:9])[O:6][H:10])[H:8]',
                                                            np.array([[2, 4, 1], [2, 0, 2], [0, 1, 1], [1, 3, 2],
                                                                      [3, 4, 1], [3, 5, 1], [2, 8, 1], [0, 6, 1],
                                                                      [1, 7, 1], [5, 9, 1]]))])
@pytest.mark.parametrize('toolkit', toolkits_name)
def test_connectivity(mapped_smiles, expected_table, toolkit):
    """Test connectivity table"""
    molecule = utils.load_molecule(mapped_smiles, toolkit)
    atom_map = utils.get_atom_map(molecule, mapped_smiles)
    connectivity_table = utils.get_connectivity_table(molecule, atom_map)

    for bond in connectivity_table:
        xi = np.isin(expected_table, bond[:2])
        match = np.where(np.array([i[:2].sum() for i in xi]) == 2)[0]
        # assert that a match was found and only one was found
        assert len(match) == 1
        # assert that bond order is the same
        assert expected_table[match][0][-1] == bond[-1]


@pytest.mark.parametrize('toolkit, toolkit_name', list(zip(toolkits, toolkits_name)))
@pytest.mark.parametrize('permute', [True, False])
def test_map_order_geometry(permute, toolkit, toolkit_name):
    """Test map ordered geometry"""
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
    mol = utils.load_molecule(hooh, toolkit=toolkit_name, permute_xyz=permute)
    mapped_smiles = utils.mol_to_smiles(mol, isomeric=True, explicit_hydrogen=True, mapped=True)
    atom_map = utils.get_atom_map(mol, mapped_smiles)
    symbols, geometry = toolkit.get_map_ordered_geometry(mol, atom_map)

    json_geom = np.asarray(hooh['geometry']).reshape(int(len(geometry)/3), 3)
    geometry_array = np.asarray(geometry).reshape(int(len(geometry)/3), 3)

    for m in atom_map:
        for i in range(3):
            assert json_geom[atom_map[m]][i] == pytest.approx(geometry_array[m-1][i], 0.0000001)
    if not permute:
        assert hooh['geometry'] == pytest.approx(geometry, 0.0000001)


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_permute_json(toolkit):
    """Test permute json xyz, symbols and connectivity to match mapped smiles"""
    molecule_ids = {'canonical_smiles': 'OO',
                    'canonical_isomeric_smiles': 'OO',
                    'canonical_isomeric_explicit_hydrogen_smiles': '[H]OO[H]',
                    'canonical_explicit_hydrogen_smiles': '[H]OO[H]',
                    'canonical_isomeric_explicit_hydrogen_mapped_smiles': '[H:3][O:1][O:2][H:4]',
                    'unique_protomer_representation': 'OO',
                    'unique_tautomer_representation': 'OO',
                    'provenance': 'cmiles_v0.1.1+59.g0b7a12d.dirty_openeye_2018.Oct.b7',
                    'standard_inchi': 'InChI=1S/H2O2/c1-2/h1-2H',
                    'inchi_key': 'MHAJPDPJQMAIIY-UHFFFAOYSA-N',
                    'molecular_formula': 'H2O2'}
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
        'molecular_multiplicity': 1
    }
    mol = utils.mol_from_json(hooh, toolkit=toolkit)
    atom_map = utils.get_atom_map(mol, '[H:3][O:1][O:2][H:4]')
    permuted_hooh = utils.permute_qcschema(hooh, molecule_ids, toolkit=toolkit)

    json_geom = np.asarray(hooh['geometry']).reshape(int(len(hooh['geometry'])/3), 3)
    permuted_geom = np.asarray(permuted_hooh['geometry']).reshape(int(len(hooh['geometry'])/3), 3)

    assert hooh['geometry'] != permuted_hooh['geometry']
    for m in atom_map:
        for i in range(3):
            assert json_geom[atom_map[m]][i] == pytest.approx(permuted_geom[m-1][i], 0.0000001)

