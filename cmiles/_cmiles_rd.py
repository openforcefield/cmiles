"""

"""
import rdkit as toolkit
from rdkit import Chem
from .utils import _symbols, ANGSROM_2_BOHR
import warnings
import copy


def mol_from_json(symbols, connectivity, geometry, permute_xyz=False):
    """
    Generate RDkit.Chem.Mol from QCSchema molecule specs.
    Parameters
    ----------
    inp_molecule: dict
        Must include symbols and connectivity. Geometry is optional. If geometry is given, stereochemistry will be taken
        from coordinates

    Returns
    -------
    molecule: rdkit.Chem.Mol
    """

    from rdkit.Geometry.rdGeometry import Point3D

    _BO_DISPATCH_TABLE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

    geometry = geometry.reshape(int(len(geometry)/3), 3)
    conformer = Chem.Conformer(len(symbols))
    has_geometry = True

    molecule = Chem.Mol()
    em = Chem.RWMol(molecule)
    for i, s in enumerate(symbols):
        atom = em.AddAtom(Chem.Atom(_symbols[s]))
        atom_position = Point3D(geometry[i][0], geometry[i][1], geometry[i][2])
        conformer.SetAtomPosition(atom, atom_position)

    # Add connectivity
    for bond in connectivity:
        bond_type = _BO_DISPATCH_TABLE[bond[-1]]
        em.AddBond(bond[0], bond[1], bond_type)

    molecule = em.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except:
        raise RuntimeError("Could not sanitize molecule")

    # Add coordinates
    if has_geometry:
        initial_conformer_id = molecule.AddConformer(conformer, assignId=True)
        # Assign stereochemistry from coordinates
        from rdkit.Chem import rdmolops
        rdmolops.AssignStereochemistryFrom3D(molecule, confId=initial_conformer_id, replaceExistingTags=True)
        if not permute_xyz:
            # Add a tag to keep current order
            molecule.SetProp("_json_geometry", '1')

    return molecule


def canonical_order_atoms(molecule, h_last=True):
    """
    Canonical order atoms in RDKit molecule. Eaach atom in the molecule is given a map index that corresponds to the RDkit
    rank for that atom (+1). RDKit atom ranking ranks hydrogens first and then the heavy atoms. When
    h_last is set to True, the map indices are reordered to put hydrogens after the heavy atoms.
    Parameters
    ----------
    molecule: rdkit mol
    h_last: bool, optional, default is True

    Returns
    -------
    molecule: rdkit molecule with map indices that correspond to the atom canonical rank
    """

    # Check if molecule already has map. If it does, remove map because Chem.CanonicalRankAtoms uses map indices in
    # ranking
    if has_atom_map(molecule):
        remove_atom_map(molecule)

    # Add explicit hydrogen
    molecule = Chem.AddHs(molecule)
    heavy_atoms = 0
    hydrogens = 0
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    for i, j in enumerate(ranks):
        atom = molecule.GetAtomWithIdx(i)
        atom.SetAtomMapNum(j+1)
        if atom.GetAtomicNum() != 1:
            # heavy atom
            heavy_atoms +=1
        else:
            # hydrogen
            hydrogens +=1

    if h_last:
        # reorder map to put hydrogen last
        for atom in molecule.GetAtoms():
            map_idx = atom.GetAtomMapNum()
            if atom.GetAtomicNum() !=1:
                atom.SetAtomMapNum(map_idx - hydrogens)
            else:
                atom.SetAtomMapNum(map_idx + heavy_atoms)
    return molecule


def mol_to_smiles(molecule, isomeric=True, explicit_hydrogen=True, mapped=True):
    """
    Generate canonical SMILES with RDKit

    Parameters
    ----------
    molecule: RDKit Chem.rdchem.Mol instance
        The molecule to generate SMILES for
    isomeric: bool
        If True, SMILES will have isomeric information. If molecule already has isomeric information, this will be retained.
        If no isomeric information exists, this function will perceive it and assign the CW (clockwise) flag for chiral
        centers and the E-isomer for stereo bonds.
    explicit_hydrogen: bool
        If True, SMILES will have explicit hydrogens
    mapped: bool
        If True, SMILES will have map indices. (+1 because the map is 1 indexed)

    Returns
    -------
    smiles: str
        The canonical SMILES

    """
    if mapped and not explicit_hydrogen:
        raise Warning("Tagged SMILES must include hydrogens to retain order")

    if mapped and not isomeric:
        raise Warning("Tagged SMILES must include stereochemistry ")

    if explicit_hydrogen:
        # Add explicit hydrogens
        molecule = Chem.AddHs(molecule)
    if not explicit_hydrogen:
        molecule = Chem.RemoveHs(molecule)

    try:
        json_geometry = int(molecule.GetProp("_json_geometry"))
    except KeyError:
        json_geometry = False

    if isomeric and not has_stereo_defined(molecule):
        raise ValueError("Some stereochemistry is not defined")

    # Get canonical order for map
    if mapped:
        if json_geometry:
            # keep original ordering
            #ToDo this looks like a potential bug that only json_geometry gets atom maps
            for i in range(molecule.GetNumAtoms()):
                molecule.GetAtomWithIdx(i).SetAtomMapNum(i+1)
        else:
            molecule = canonical_order_atoms(molecule)

    smiles = Chem.MolToSmiles(molecule, allHsExplicit=explicit_hydrogen, isomericSmiles=isomeric, canonical=True)
    return smiles

def generate_index(mol, labeled_atoms):
    """
    Generate SMILES with labeled atoms

    Parameters
    ----------
    mol : rdmol
        Must have atom indices that correspond to labeled atoms
    labeled_atoms : tuple of ints
        ints should be -1 of atom map

    Returns
    -------
    labeled SMILES

    """
    # Remove explicit H
    mol = Chem.RemoveHs(mol)
    for a in mol.GetAtoms():
        if not a.GetAtomMapNum()-1 in labeled_atoms:
            a.SetAtomMapNum(0)

    for a in mol.GetAtoms():
        m = a.GetAtomMapNum() -1
        if m in labeled_atoms:
            i = labeled_atoms.index(m)
            a.SetAtomMapNum(i+1)
    return Chem.MolToSmiles(mol)

def get_connectivity_table(molecule, inverse_map):
    """

    Parameters
    ----------
    molecule
    inverse_map

    Returns
    -------

    """
    # Remove aromaticity
    Chem.Kekulize(molecule)
    connectivity_table = [[inverse_map[bond.GetBeginAtomIdx()]-1, inverse_map[bond.GetEndAtomIdx()]-1, bond.GetBondTypeAsDouble()]
                              for bond in molecule.GetBonds()]
    return connectivity_table



def get_atom_map(molecule, mapped_smiles):
    """

    Parameters
    ----------
    molecule
    mapped_smiles

    Returns
    -------

    """
    # Check for mapping in mapped_smiles
    mapped_mol = Chem.MolFromSmiles(mapped_smiles)
    if not has_atom_map(mapped_mol):
        raise ValueError("mapped SMILES must have map for every heavy atom and hydrogen")
    # Check if molecule has explicit H
    if not has_explicit_hydrogen(molecule):
        molecule = Chem.AddHs(molecule)

    mapped_pattern = Chem.MolFromSmarts(mapped_smiles)
    if molecule.HasSubstructMatch(mapped_pattern):
        idx_pattern_order = molecule.GetSubstructMatch(mapped_pattern)
    else:
        raise RuntimeError("Substrucure match failed for {}, SMARTS: {}".format(Chem.MolToSmiles(molecule), mapped_smiles))
    atom_map = {}
    for i, j in enumerate(idx_pattern_order):
        atom_map[mapped_pattern.GetAtomWithIdx(i).GetAtomMapNum()] = j

    return atom_map


def get_map_ordered_geometry(molecule, atom_map):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    try:
        if not molecule.GetConformer().Is3D():
            raise ValueError("Molecule must have 3D coordinates for generating QCSchema molecule")
    except ValueError:
        raise ValueError("Molecule must have 3D coordinates for generating QCSchema molecule")

    symbols = []
    geometry = []
    for mapping in range(1, molecule.GetNumAtoms()+1):
        idx = atom_map[mapping]
        atom = molecule.GetAtomWithIdx(idx)
        syb = atom.GetSymbol()
        symbols.append(syb)
        pos = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        geometry.extend(getattr(pos, c) * ANGSROM_2_BOHR for c in ['x', 'y', 'z'])

    return symbols, geometry


def has_explicit_hydrogen(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    explicit = True
    for a in molecule.GetAtoms():
        if a.GetImplicitValence() > 0 or a.GetNumExplicitHs() > 0:
            explicit = False

    return explicit


def add_explicit_hydrogen(molecule):
    molecule = Chem.AddHs(molecule)
    return molecule


def get_symbols(molecule):
    return [a.GetSymbol() for a in molecule.GetAtoms()]


def has_stereo_defined(molecule):
    #ToDo Fix this function for carbons with explicit hydrogen and imine with implicit hydrogens
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()
    # remove map indices and store. Then create molecule with SMILES without map indices and check for stereo on that.
    # This step is needed because even when map indices are stored in the data of the atoms, it is used for assigning potential
    # stereo centers. If a molecule does not have map indices but has data on the atoms, it will get flagged
    had_atom_map = False
    if has_atom_map(molecule):
        had_atom_map = True
        remove_atom_map(molecule)
    a = Chem.rdmolfiles.SmilesParserParams()
    a.removeHs = False
    mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(molecule), a)
    # restore map indices
    if had_atom_map:
        restore_atom_map(molecule)
    chiral_centers = Chem.FindMolChiralCenters(mol_copy, includeUnassigned=True)
    for center in chiral_centers:
        atom_id = center[0]
        if center[-1] == '?':
            unspec_chiral = True
            problematic_atoms.append((atom_id, mol_copy.GetAtomWithIdx(atom_id).GetSmarts()))

    # Find potential stereo bonds that are unspecified
    Chem.FindPotentialStereoBonds(mol_copy)
    for bond in mol_copy.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            unspec_db = True
            problematic_bonds.append((bond.GetBeginAtom().GetSmarts(), bond.GetSmarts(),
                                                bond.GetEndAtom().GetSmarts()))
    if unspec_chiral or unspec_db:
        warnings.warn("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms, problematic_bonds))
        return False
    else:
        return True


def has_atom_map(molecule):
    """
    Checks if any atom has map indices. Will return True if even only one atom has a map index
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    IS_MAPPED = False
    for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum() != 0:
                IS_MAPPED = True
                return IS_MAPPED
    return IS_MAPPED


def is_missing_atom_map(molecule):
    """
    Checks if any atom in molecule is missing a map index. If even only one atom is missing a map index will return True

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    MISSING_ATOM_MAP = False
    for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                MISSING_ATOM_MAP = True
                return MISSING_ATOM_MAP
    return MISSING_ATOM_MAP


def remove_atom_map(molecule, keep_map_data=True):
    """
    Remove atom map but store it in atom data.
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    for atom in molecule.GetAtoms():
        if atom.GetAtomMapNum() != 0:
            if keep_map_data:
                atom.SetProp('_map_idx', str(atom.GetAtomMapNum()))
            atom.SetAtomMapNum(0)


def restore_atom_map(molecule):
    """
    Restore atom map from atom data
    Parameters
    ----------
    molecule: OEMol
        Must have 'MapIdx' in atom data dictionary
    """
    for atom in molecule.GetAtoms():
        if atom.HasProp('_map_idx'):
            atom.SetAtomMapNum(int(atom.GetProp('_map_idx')))

def add_atom_map(molecule, in_place=False):
    """
    Add canonical ordered atom map to existing molecule

    Parameters
    ----------
    molecule : rdkit.Chem.Mol
    in_place : bool, default False

    Returns
    -------
    rdkit.Chem.Mol with atom map
    """
    # First canonical order atoms
    molecule =  canonical_order_atoms(molecule)

    for i in range(molecule.GetNumAtoms()):
        molecule.GetAtomWithIdx(i).SetAtomMapNum(i+1)
    return molecule


def is_map_canonical(molecule):
    """
    Check if map indices on molecule is in canonical order.
    Note:
    This only checks map indices on heavy atoms.

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    molcopy = copy.deepcopy(molecule)
    # reorder molcopy
    canonical_ordered_mol = canonical_order_atoms(molcopy)

    # check that both mapped SMILES are equal
    smiles_1 = Chem.MolToSmiles(molecule, canonical=True)
    smiles_2 = Chem.MolToSmiles(canonical_ordered_mol, canonical=True)
    return smiles_1 == smiles_2
