"""

"""
import openeye as toolkit
from openeye import oechem
import time
import copy
import warnings
from .utils import ANGSROM_2_BOHR, _symbols


def mol_from_json(symbols, connectivity, geometry, permute_xyz=False):
    """
    Generate OEMol from QCSchema molecule specs
    Parameters
    ----------
    inp_molecule: dict
        Must have symbols and connectivity and/or geometry
    permute_xyz: bool, optional, default False
        If False, the molecule will have a tag indicating that the goemetry came from QCSchema. This
        will ensure that the order of the atoms and configuration is not change for generation of mapped SMILES and
        isomeric SMILES.
        If True, atoms will be reordered when generating mapped SMILES to canonical order and mapped SMILES might be
        different than xyz order.

    Returns
    -------
    molecule: OEMol

    """

    molecule = oechem.OEMol()
    for s in symbols:
        molecule.NewAtom(_symbols[s])

    # Add connectivity
    for bond in connectivity:
        a1 = molecule.GetAtom(oechem.OEHasAtomIdx(bond[0]))
        a2 = molecule.GetAtom(oechem.OEHasAtomIdx(bond[1]))
        molecule.NewBond(a1, a2, bond[-1])

    # Add geometry
    if molecule.NumAtoms() != geometry.shape[0]/3:
        raise ValueError("Number of atoms in molecule does not match length of position array")

    molecule.SetCoords(oechem.OEFloatArray(geometry))
    molecule.SetDimension(3)

    geom_tag = oechem.OEGetTag("json_geometry")
    if permute_xyz:
        # Add tag that the geometry is from JSON but can be changed.
        molecule.SetData(geom_tag, False)
    else:
        molecule.SetData(geom_tag, True)

    oechem.OEDetermineConnectivity(molecule)
    oechem.OEFindRingAtomsAndBonds(molecule)
    oechem.OEPerceiveBondOrders(molecule)
    oechem.OEAssignImplicitHydrogens(molecule)
    oechem.OEAssignFormalCharges(molecule)
    oechem.OEAssignAromaticFlags(molecule)
    oechem.OEPerceiveChiral(molecule)
    oechem.OE3DToAtomStereo(molecule)
    oechem.OE3DToBondStereo(molecule)

    return molecule


def canonical_order_atoms(molecule, in_place=True):
    """
    Canonical order of atom indices. This ensures the map indices are always in the same order.
    Parameters
    ----------
    molecule: oechem.OEMol
    in_place: bool, optional, default True
        If True, the order of atom indices will happen in place. If False, a copy of the molecule with reordered atom
        indices will be returned.

    Returns
    -------
    molecule: OEMol with canonical ordered atom indices.

    """

    if not in_place:
        molecule = copy.deepcopy(molecule)

    oechem.OECanonicalOrderAtoms(molecule)
    oechem.OECanonicalOrderBonds(molecule)

    vatm = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != oechem.OEElemNo_H:
            vatm.append(atom)
    molecule.OrderAtoms(vatm)

    vbnd = []
    for bond in molecule.GetBonds():
        if bond.GetBgn().GetAtomicNum() != oechem.OEElemNo_H and bond.GetEnd().GetAtomicNum() != oechem.OEElemNo_H:
            vbnd.append(bond)
    molecule.OrderBonds(vbnd)

    molecule.Sweep()

    for bond in molecule.GetBonds():
        if bond.GetBgnIdx() > bond.GetEndIdx():
            bond.SwapEnds()

    if not in_place:
        return molecule


def mol_to_smiles(molecule, isomeric=True, explicit_hydrogen=True, mapped=True):
    """
    Generate canonical SMILES with OpenEye.
    Parameters
    ----------

    molecule: oechem.OEMol
    isomeric: bool
        If True, SMILES will include chirality and stereo bonds
    explicit_hydrogen: bool
        If True, SMILES will include explicit hydrogen
    mapped: bool
        If True, will include map indices (In order of OpenEye omega canonical ordering)
    generate_conformer: bool, optional. Default True
        Generating conformer is needed to infer stereochemistry if SMILES does not have stereochemistry specified. Sometimes,
        however, this can be very slow because the molecule has many rotatable bonds. Then it is recommended to turn
        off generate_conformer but the stereochemistry might not be specified in the isomeric SMILES

    Returns
    -------
    smiles str

    """

    molecule = oechem.OEMol(molecule)

    if has_atom_map(molecule):
        #ToDo remove?
        raise ValueError("Why oh why does the molecule still have map indices if it was already removed???")

    if explicit_hydrogen:
        if not has_explicit_hydrogen(molecule):
            oechem.OEAddExplicitHydrogens(molecule)

    # First check if geometry from JSON exists
    try:
        JSON_geometry = molecule.GetData('json_geometry')
    except ValueError:
        JSON_geometry = False

    if isomeric:
        if not has_stereo_defined(molecule):
            raise ValueError("Smiles must have stereochemistry defined.")

    if not explicit_hydrogen and not mapped and isomeric:
        return oechem.OEMolToSmiles(molecule)
    if not explicit_hydrogen and not mapped and not isomeric:
        return oechem.OECreateSmiString(molecule, oechem.OESMILESFlag_Canonical | oechem.OESMILESFlag_RGroups)

    if not mapped and explicit_hydrogen and isomeric:
        return oechem.OECreateSmiString(molecule, oechem.OESMILESFlag_Hydrogens | oechem.OESMILESFlag_ISOMERIC)

    if not mapped and explicit_hydrogen and not isomeric:
        return oechem.OECreateSmiString(molecule, oechem.OESMILESFlag_Hydrogens | oechem.OESMILESFlag_Canonical |
                                        oechem.OESMILESFlag_RGroups)

    if not JSON_geometry:
        # canonical order of atoms if input was SMILES or permute_xyz is true
        canonical_order_atoms(molecule)

    for atom in molecule.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)

    if mapped and not explicit_hydrogen:
        raise Warning("Tagged SMILES must include hydrogens to retain order")

    if mapped and not isomeric:
        raise Warning("Tagged SMILES must include stereochemistry ")

    return oechem.OEMolToSmiles(molecule)


def get_connectivity_table(molecule):
    """

    Parameters
    ----------
    molecule
    inverse_map

    Returns
    -------

    """
    # Check that molecule is in canonical order
    if not is_molecule_canonical_order(molecule):
        raise ValueError("Molecule atoms are not in canonical order")
    connectivity_table = [[bond.GetBgnIdx(), bond.GetEndIdx(), bond.GetOrder()] for bond
                          in molecule.GetBonds()]

    return connectivity_table


def get_atom_map(molecule, mapped_smiles):
    """
    Map tag in mapped SMILES to atom idx
    This uses a substructure search. For a symmetrical molecule, there are as many matches as symmetries and atom maps can flip.
    It should only be used if the mapped SMILES was not generated with openeye.

    Parameters
    ----------
    molecule
    mapped_smiles
    strict: bool, optional, default True
        If True, will only return atom map if map on molecule is canonical and if mapped_smiles is provided, is also
        canonical.
        If False, will return atom map that is on molecule even if it's not canonical. If a mapped SMILES was provided,
        it will return atom map from substructure search on that SMILES.

    Returns
    -------

    """
    # Check if molecule has explicit hydrogen
    if not has_explicit_hydrogen(molecule):
        raise ValueError("Molecule must have explicit hydrogen")

    # # Check that map SMILES has map indices on all atoms and that it's in canonical order.
    # mapped_mol = oechem.OEMol()
    # oechem.OESmilesToMol(mapped_mol, mapped_smiles)
    # # Check for explicit hydrogen
    # if not has_explicit_hydrogen(mapped_mol):
    #     raise ValueError("Mapped SMILES must include explicit hydrogen")
    #
    # # Check that all atoms have map indices
    # if is_missing_atom_map(mapped_mol):
    #     # mapped SMILES must have map index on every atom
    #     raise ValueError("Mapped SMILES must have map indices for all atoms and hydrogen")
    #
    # # check canonical order of atom map
    # if not is_map_canonical(mapped_mol):
    #     if strict:
    #         raise ValueError("Map indices on mapped SMILES are not in openeye canonical order. If you set strict to False, "
    #                          "you can get an atom map but only do this if you understand what you are doing.")
    #     # map in mapped SMILES is not canonical - it might come from RDKit
    #     can_mapped_smiles = False
    # else:
    #     can_mapped_smiles = True
    #
    # if can_mapped_smiles:
    #     # Mapped SMILES is canonical. Get atom map from canonical ordered molecule
    #     molcopy = oechem.OEMol(molecule)
    #     # store atom indices in molecule data
    #     for atom in molcopy.GetAtoms():
    #         atom.SetData('atom_idx', atom.GetIdx())
    #     canonical_order_atoms(molcopy)
    #     atom_map = {}
    #     for atom in molcopy.GetAtoms():
    #         atom_map[atom.GetIdx()+1] = atom.GetData('atom_idx')
    # elif not can_mapped_smiles and not strict:
    #     warnings.warn("Using substructure search to get atom map. If your molecule is symmetric, more than one atom mapping"
    #                   " is possible and the order of the geometry can flip.")
        # Use substructure search
    ss = oechem.OESubSearch(mapped_smiles)
    oechem.OEPrepareSearch(molecule, ss)
    ss.SetMaxMatches(1)

    atom_map = {}
    matches = [m for m in ss.Match(molecule)]
    if not matches:
        raise RuntimeError("MCSS failed for {}, smiles: {}".format(oechem.OEMolToSmiles(molecule), mapped_smiles))
    for match in matches:
        for ma in match.GetAtoms():
            atom_map[ma.pattern.GetMapIdx()] = ma.target.GetIdx()

    # sanity check
    mol = oechem.OEGraphMol()
    oechem.OESubsetMol(mol, match, True)
    print("Match SMILES: {}".format(oechem.OEMolToSmiles(mol)))

    return atom_map


def get_map_ordered_geometry(molecule, mapped_smiles):
    """

    Parameters
    ----------
    molecule: OEMol with 1 conformer or OEConf
    atom_map: dict
        map_idx:atom_idx

    Returns
    -------
    symbols: list of str
        list of symbols
    geometry: list of ints
        xyz geometry. List is length N*3
    """

    if not molecule.GetDimension() == 3:
        raise RuntimeError("Molecule must have 3D coordinates for generating a QCSchema molecule")

    if isinstance(molecule, oechem.OEMol):
        if molecule.GetMaxConfIdx() != 1:
            raise Warning("The molecule must have at least and at most 1 conformation")

    # Check for JSON geometry
    try:
        json_geometry = molecule.GetData('json_geometry')
    except ValueError:
        json_geometry = None

    # Check that map SMILES and order in geometry all match up
    if not is_mapped_smiles_canonical(mapped_smiles):
        raise ValueError("Map indices on SMILES are not canonical")
    if not is_goemetry_canonical_order(molecule):
        if json_geometry is None:
            raise ValueError("Geometry was not generated with omega canonical order set to True")
    if not is_molecule_canonical_order(molecule):
        if json_geometry is None:
            raise ValueError("molecule atoms are not in canonical order")
        elif json_geometry:
            pass
        elif json_geometry is False:
             canonical_order_atoms(molecule)

    symbols = []
    geometry = []
    #for mapping in range(1, molecule.NumAtoms()+1):
    for atom in molecule.GetAtoms():
        #atom = molecule.GetAtom(oechem.OEHasAtomIdx(idx))
        syb = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
        symbols.append(syb)
        for i in range(3):
            geometry.append(molecule.GetCoords()[atom.GetIdx()][i]*ANGSROM_2_BOHR)

    return symbols, geometry


def has_explicit_hydrogen(molecule):
    #ToDo test this function well
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    return not oechem.OEHasImplicitHydrogens(molecule)


def add_explicit_hydrogen(molecule):
    oechem.OEAddExplicitHydrogens(molecule)
    return molecule


def get_symbols(molecule):
    return [oechem.OEGetAtomicSymbol(a.GetAtomicNum()) for a in molecule.GetAtoms()]


def has_stereo_defined(molecule):
    """
    Check if any stereochemistry in undefined.
    Parameters
    ----------
    molecule: OEMol

    Returns
    -------
    bool: True if all stereo chemistry is defined.
        If any stereochemsitry is undefined, raise and exception.

    """

    # perceive stereochemistry
    oechem.OEPerceiveChiral(molecule)

    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()
    for atom in molecule.GetAtoms():
        if atom.IsChiral() and not atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
            # Check if handness is specified
            v = []
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, oechem.OEAtomStereo_Tetrahedral)
            if stereo == oechem.OEAtomStereo_Undefined:
                unspec_chiral = True
                problematic_atoms.append((atom.GetIdx(), oechem.OEGetAtomicSymbol(atom.GetAtomicNum())))
    for bond in molecule.GetBonds():
        if bond.IsChiral() and not bond.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            v = []
            for neigh in bond.GetBgn().GetAtoms():
                if neigh != bond.GetEnd():
                    v.append(neigh)
                    break
            for neigh in bond.GetEnd().GetAtoms():
                if neigh != bond.GetBgn():
                    v.append(neigh)
                    break
            stereo = bond.GetStereo(v, oechem.OEBondStereo_CisTrans)

            if stereo == oechem.OEBondStereo_Undefined:
                unspec_db = True
                a1 = bond.GetBgn()
                a2 = bond.GetEnd()
                a1_idx = a1.GetIdx()
                a2_idx = a2.GetIdx()
                a1_s = oechem.OEGetAtomicSymbol(a1.GetAtomicNum())
                a2_s = oechem.OEGetAtomicSymbol(a2.GetAtomicNum())
                bond_order = bond.GetOrder()
                problematic_bonds.append((a1_idx, a1_s, a2_idx, a2_s, bond_order))
    if unspec_chiral or unspec_db:
        warnings.warn("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms,
                problematic_bonds))
        return False
    else:
        return True


def has_atom_map(molecule):
    """
    Checks if any atom has map indices. Will return True if only one atom has a map index
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    IS_MAPPED = False
    for atom in molecule.GetAtoms():
            if atom.GetMapIdx() != 0:
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
            if atom.GetMapIdx() == 0:
                MISSING_ATOM_MAP = True
                return MISSING_ATOM_MAP
    return MISSING_ATOM_MAP


def remove_atom_map(molecule, keep_map_data=True):
    """
    Remove atom map but store it in atom data. In place.
    Parameters
    ----------
    molecule: OEMol with map indices
    keep_map_data: bool, optional, default True
        If True, will store map indices in 'MapIdx' field in atom data

    """
    for atom in molecule.GetAtoms():
        if atom.GetMapIdx() != 0:
            if keep_map_data:
                atom.SetData('MapIdx', atom.GetMapIdx())
            atom.SetMapIdx(0)


def restore_atom_map(molecule):
    """
    Restore atom map from atom data. In place.
    Parameters
    ----------
    molecule: OEMol
        Must have 'MapIdx' in atom data dictionary
    """
    for atom in molecule.GetAtoms():
        if atom.HasData('MapIdx'):
            atom.SetMapIdx(atom.GetData('MapIdx'))


# def is_map_canonical(molecule):
#     """
#     Check if map indices on molecule is in canonical order.
#     Reorder atoms to be in canonical order and compare the atom order to the map indices.
#
#     Parameters
#     ----------
#     molecule: OEMol created from the mapped SMILES
#
#     Returns
#     -------
#     True if canonical ordered atoms match map indices. False otherwise.
#
#     """
#     molcopy = oechem.OEMol(molecule)
#     # reorder molcopy
#     canonical_order_atoms(molcopy, in_place=True)
#     # Now check that map indices are +1 on atom indices
#     for a in molcopy.GetAtoms():
#         if a.GetMapIdx() != a.GetIdx() + 1:
#             return False
#     return True


def is_mapped_smiles_canonical(mapped_smiles):
    """

    Parameters
    ----------
    mapped_smiles

    Returns
    -------

    """
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, mapped_smiles)
    # romove map
    remove_atom_map(mol)
    # Generate canonical mapped SMILES
    if not mol_to_smiles(mol, mapped=True, explicit_hydrogen=True, isomeric=True) == mapped_smiles:
        # This mapped SMILES was not generated by CMILES with openeye.
        return False
    return True
    #


def is_goemetry_canonical_order(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    # Check that molecule has geometry
    if not molecule.GetDimension() == 3:
        raise ValueError("Molecule must have 3D coordinates")
    molecule_copy = oechem.OEMol(molecule)
    canonical_order_atoms(molecule_copy)
    if not molecule.GetCoords() == molecule_copy.GetCoords():
        # geometry was not generated with canonical order set to True
        warnings.warn("Geometry was not generated with canonical order set to True")
        return False
    return True


def is_molecule_canonical_order(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    molcopy = oechem.OEMol(molecule)
    if has_atom_map(molcopy):
        remove_atom_map(molcopy)
    # Check for JSON geometry
    try:
        json_geometry = molcopy.GetData('json_geometry')
        if json_geometry:
            molcopy.SetData('json_geometry', False)
    except ValueError:
        pass
    canonical_mapped_smiles = mol_to_smiles(molcopy, isomeric=True, explicit_hydrogen=True, mapped=True)

    # Add map indices to molcopy
    for atom in molcopy.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)
    mapped_smiles = oechem.OEMolToSmiles(molcopy)
    if not canonical_mapped_smiles == mapped_smiles:
        return False
    return True
