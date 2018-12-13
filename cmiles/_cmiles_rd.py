"""

"""
from rdkit import Chem
from .utils import _symbols, ANGSROM_2_BOHR


def mol_from_json(symbols, connectivity, geometry):
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


def has_stereo_defined(molecule):
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
    chiral_centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    for center in chiral_centers:
        atom_id = center[0]
        if center[-1] == '?':
            unspec_chiral = True
            problematic_atoms.append((atom_id, molecule.GetAtomWithIdx(atom_id).GetSmarts()))

    # Find potential stereo bonds that are unspecified
    Chem.FindPotentialStereoBonds(molecule)
    for bond in molecule.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            if not _ignore_stereo_flag(bond):
                unspec_db = True
                problematic_bonds.append((bond.GetBeginAtom().GetSmarts(), bond.GetSmarts(),
                                                bond.GetEndAtom().GetSmarts()))
    if unspec_chiral or unspec_db:
        raise ValueError("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms, problematic_bonds))
    else:
        return True


def _ignore_stereo_flag(bond):
    ignore = False
    beg = bond.GetBeginAtom()
    end = bond.GetEndAtom()

    if (beg.GetAtomicNum() == 7) and (end.GetAtomicNum() == 6) and (bond.GetBondType() == Chem.BondType.DOUBLE):
        for a in beg.GetNeighbors():
            if a != end and a.GetAtomicNum() == 1:
                # This is a C=NH bond and should be ignored when flagged
                ignore = True
                break

    if (beg.GetAtomicNum() == 6) and (end.GetAtomicNum() == 7) and (bond.GetBondType() == Chem.BondType.DOUBLE):
        for a in end.GetNeighbors():
            if a != beg and a.GetAtomicNum() == 1:
                # This is a C=NH bond and should be ignored when flagged
                ignore = True
                break
    return ignore


def has_atom_map(molecule):
    IS_MAPPED = True
    for atom in molecule.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            IS_MAPPED = False
    return IS_MAPPED


def remove_atom_map(molecule):
    for a in molecule.GetAtoms():
        a.SetAtomMapNum(0)
