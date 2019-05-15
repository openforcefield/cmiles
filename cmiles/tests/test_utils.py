"""Test util functions"""

from cmiles import utils
import pytest
import numpy as np

toolkits = list()
toolkits_name = list()
if utils.has_rdkit:
    from cmiles import _cmiles_rd
    from rdkit import Chem
    toolkits.append(_cmiles_rd)
    toolkits_name.append('rdkit')
if utils.has_openeye:
    from cmiles import _cmiles_oe
    from openeye import oechem
    toolkits.append(_cmiles_oe)
    toolkits_name.append('openeye')

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
@pytest.mark.parametrize('input, output', [('[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]', True),
                                           ('[H:3][C:1]([H])([H:5])[C]([H])([H:7])[H:8]', True),
                                           ('CCCC', False)])
def test_is_mapped(toolkit, input, output):
    """Test is mapped"""
    mapped_mol = utils.load_molecule(input, toolkit=toolkit)
    assert utils.has_atom_map(mapped_mol) == output
    utils.remove_atom_map(mapped_mol)
    assert utils.has_atom_map(mapped_mol) == False


@pytest.mark.parametrize('toolkit', toolkits_name)
@pytest.mark.parametrize('input, output', [('[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]', False),
                                           ('[H:3][C:1]([H:4])([H:5])[C]([H:6])([H:7])[H:8]', True),
                                           ('CCCC', True)])
def test_is_missing_map(toolkit, input, output):
    #ToDo - Known problem that RDKit does not add explicit H to molecules even with explicit H SMILES so if map of H is missing it will not pick it up
    mol = utils.load_molecule(input, toolkit=toolkit)
    assert utils.is_missing_atom_map(mol) == output


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
    with pytest.warns(UserWarning):
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
    with pytest.warns(UserWarning):
        utils.has_stereo_defined(mol)


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

@using_openeye
@pytest.mark.parametrize('smiles', ['OCCO', 'C(CO)O', '[H]C([H])(C([H])([H])O[H])O[H]',
                                    '[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]',
                                   '[H][O][C]([H])([H])[C]([H])([H])[O][H]',
                                    '[O:1]([C:3]([C:4]([O:2][H:6])([H:9])[H:10])([H:7])[H:8])[H:5]'])
def test_atom_map(smiles):
    """Test that atom map orders geometry the same way every time no matter the SMILES used to create the molecule"""
    import cmiles
    mapped_smiles = '[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]'
    mol_id_oe = cmiles.to_molecule_id(mapped_smiles, canonicalization='openeye')
    oemol = utils.load_molecule(mapped_smiles, toolkit='openeye')
    mapped_symbols = ['C', 'C', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H']
    mapped_geometry = [-1.6887193912042044, 0.8515190939276903, 0.8344587822904272, -4.05544806361675, -0.3658269566455062,
                       -0.22848169646448416, -1.6111611950422127, 0.4463128276938808, 3.490617694146934, -3.97756355964586,
                       -3.0080934853087373, 0.25948499322223956, -1.6821252026076652, 2.891135395246369, 0.4936556190978574,
                       0.0, 0.0, 0.0, -4.180315034973438, -0.09210893239246959, -2.2748227320305525, -5.740516456782416,
                       0.4115539217904015, 0.6823267491485907, -0.07872657410528058, 1.2476492272884379, 4.101615944163073,
                       -5.514569080545831, -3.7195945404657222, -0.4441653010509862]

    mol = cmiles.utils.load_molecule(smiles, toolkit='openeye')
    if not utils.has_explicit_hydrogen(mol):
        mol = utils.add_explicit_hydrogen(mol)
    atom_map = utils.get_atom_map(mol, mapped_smiles=mapped_smiles)
    # use the atom map to add coordinates to molecule. First reorder mapped geometry to order in molecule
    mapped_coords = np.array(mapped_geometry, dtype=float).reshape(int(len(mapped_geometry)/3), 3)
    coords = np.zeros((mapped_coords.shape))
    for m in atom_map:
        coords[atom_map[m]] = mapped_coords[m-1]
    # flatten
    coords = coords.flatten()
    # convert to Angstroms
    coords = coords*utils.BOHR_2_ANGSTROM
    # set coordinates in oemol
    mol.SetCoords(coords)
    mol.SetDimension(3)

    # Get new atom map
    atom_map = utils.get_atom_map(mol, mapped_smiles)
    symbols, geometry = _cmiles_oe.get_map_ordered_geometry(mol, atom_map)
    assert geometry == mapped_geometry
    assert symbols == mapped_symbols


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


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_get_atom_map_mapped_smiles(toolkit):
    smiles_1 = '[H]C([H])(C([H])([H])O[H])O[H]'
    smiles_2 = '[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]'
    mol_1 = utils.load_molecule(smiles_1, toolkit=toolkit)
    if not utils.has_explicit_hydrogen(mol_1):
        mol_1 = utils.add_explicit_hydrogen(mol_1)
    mol_2 = utils.load_molecule(smiles_2, toolkit=toolkit)
    if not utils.has_explicit_hydrogen(mol_2):
        mol_2 = utils.add_explicit_hydrogen(mol_2)


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_remove_restore_atom_map(toolkit):
    mapped_smiles = '[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]'
    mapped_mol = utils.load_molecule(mapped_smiles, toolkit=toolkit)

    utils.remove_atom_map(mapped_mol)
    assert utils.has_atom_map(mapped_mol) == False
    assert utils.is_missing_atom_map(mapped_mol) == True

    utils.restore_atom_map(mapped_mol)
    assert utils.has_atom_map(mapped_mol) == True
    assert utils.is_missing_atom_map(mapped_mol) == False

    smiles = 'OCCO'
    mol = utils.load_molecule(smiles, toolkit=toolkit)
    with pytest.warns(UserWarning):
        utils.restore_atom_map(mol)


@pytest.mark.parametrize('toolkit', toolkits_name)
@pytest.mark.parametrize('smiles, canonicalization', [('[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]', 'openeye'),
                                                      ('[O:1]([C:3]([C:4]([O:2][H:6])([H:9])[H:10])([H:7])[H:8])[H:5]', 'rdkit')])
def test_is_map_canonical(toolkit, smiles, canonicalization):
    molecule = utils.load_molecule(smiles, toolkit)

    canonical = utils.is_map_canonical(molecule)
    if toolkit == canonicalization:
        assert canonical
    else:
        assert not canonical


@pytest.mark.parametrize('toolkit', toolkits_name)
@pytest.mark.parametrize('smiles', ['COC(C)c1c(Cl)ccc(F)c1Cl',
                                 '[H:3][C:1]([H:4])([H:5])[C:2]([H:6])([H:7])[H:8]',
                         '[H:5][C:1]([H:6])([C:2]([H:7])([H:8])[O:4][H:10])[O:3][H:9]',
                        '[O:1]([C:3]([C:4]([O:2][H:6])([H:9])[H:10])([H:7])[H:8])[H:5]' ])
def test_atom_order_in_mol_copy(toolkit, smiles):
    """Test that atom orders do not change when copying molecule"""
    import copy
    mol = utils.load_molecule(smiles, toolkit=toolkit)
    if not utils.has_explicit_hydrogen(mol):
        mol = utils.add_explicit_hydrogen(mol)
    molcopy = copy.deepcopy(mol)
    for a1, a2 in zip(mol.GetAtoms(), molcopy.GetAtoms()):
        if toolkit == 'openeye':
            assert a1.GetIdx() == a2.GetIdx()
            assert a1.GetName() == a2.GetName()
            assert a1.GetMapIdx() == a2.GetMapIdx()
        if toolkit == 'rdkit':
            assert a1.GetIdx() == a2.GetIdx()
            assert a1.GetAtomMapNum() == a2.GetAtomMapNum()
            assert a1.GetSmarts() == a2.GetSmarts()


@pytest.mark.parametrize('toolkit', toolkits_name)
def test_canonical_label(toolkit):
    """Test canonical label"""
    if toolkit == 'openeye':
        output = '[CH2:3]([CH2:4][OH:2])[OH:1]'
    if toolkit == 'rdkit':
        output = '[OH:1][CH2:3][CH2:4][OH:2]'
    assert utils.to_canonical_label('[O:1]([C:3]([C:4]([O:2][H:6])([H:9])[H:10])([H:7])[H:8])[H:5]',
                                    (0, 1, 2, 3), toolkit) == output
    with pytest.raises(RuntimeError):
        utils.to_canonical_label('OCCO', (0, 1, 2, 3), toolkit)
