"""

"""
from openeye import oechem
import time
import copy
from .utils import ANGSROM_2_BOHR, _symbols


def mol_from_json(symbols, connectivity, geometry):
    """
    Generate OEMol from QCSchema molecule specs
    Parameters
    ----------
    inp_molecule: dict
        Must have symbols and connectivity and/or geometry
        Note: If geometry is given, the molecule will have a tag indicating that the goemetry came from QCSchema. This
        will ensure that the order of the atoms and configuration is not change for generation of mapped SMILES and
        isomeric SMILES.

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

    # Add tag that the geometry is from JSON and shouldn't be changed.
    geom_tag = oechem.OEGetTag("json_geometry")
    molecule.SetData(geom_tag, True)
    oechem.OEDetermineConnectivity(molecule)
    oechem.OEFindRingAtomsAndBonds(molecule)
    oechem.OEPerceiveBondOrders(molecule)
    oechem.OEAssignImplicitHydrogens(molecule)
    oechem.OEAssignFormalCharges(molecule)
    oechem.OEAssignAromaticFlags(molecule)
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


def get_connectivity_table(molecule, inverse_map):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    connectivity_table = [[inverse_map[bond.GetBgnIdx()]-1, inverse_map[bond.GetEndIdx()]-1, bond.GetOrder()] for bond
                          in molecule.GetBonds()]

    return connectivity_table


def get_atom_map(molecule, mapped_smiles):
    """
    Map tag in mapped SMILES to atom idx

    Parameters
    ----------
    molecule
    mapped_smiles

    Returns
    -------

    """
    ss = oechem.OESubSearch(mapped_smiles)
    oechem.OEPrepareSearch(molecule, ss)
    ss.SetMaxMatches(1)

    atom_map = {}
    t1 = time.time()
    matches = [m for m in ss.Match(molecule)]
    t2 = time.time()
    seconds = t2-t1
    print("CSS took {} seconds".format(seconds))
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


def get_map_ordered_geometry(molecule, atom_map):
    """

    Parameters
    ----------
    molecule
    atom_map

    Returns
    -------

    """

    if not molecule.GetDimension() == 3:
        raise RuntimeError("Molecule must have 3D coordinates for generating a QCSchema molecule")

    if molecule.GetMaxConfIdx() != 1:
        raise Warning("The molecule must have at least and at most 1 conformation")

    symbols = []
    geometry = []
    for mapping in range(1, molecule.NumAtoms()+1):
        idx = atom_map[mapping]
        atom = molecule.GetAtom(oechem.OEHasAtomIdx(idx))
        syb = oechem.OEGetAtomicSymbol(atom.GetAtomicNum())
        symbols.append(syb)
        for i in range(3):
            geometry.append(molecule.GetCoords()[atom.GetIdx()][i]*ANGSROM_2_BOHR)

    return symbols, geometry


def has_explicit_hydrogen(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    return not oechem.OEHasImplicitHydrogens(molecule)


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
            if not _ignore_stereo_flag(bond):
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
        raise ValueError("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms,
                problematic_bonds))
    else:
        return True


def _ignore_stereo_flag(bond):

    ignore = False
    beg = bond.GetBgn()
    end = bond.GetEnd()

    if (beg.GetAtomicNum() == 7) and (end.GetAtomicNum() == 6) and (bond.GetOrder() == 2):
        for i, a in enumerate(beg.GetAtoms()):
            if a != end and a.GetAtomicNum() == 1:
                # This is a C=NH bond and should be ignored when flagged
                ignore = True
                break
        if i == 0 and beg.GetImplicitHCount() == 1:
            # This is a C=NH bond with implicit H
            ignore = True

    if (beg.GetAtomicNum() == 6) and (end.GetAtomicNum() == 7) and (bond.GetOrder() == 2):
        for i, a in enumerate(end.GetAtoms()):
            if a != beg and a.GetAtomicNum() == 1:
                # This is a C=NH bond and should be ignored when flagged
                ignore = True
                break
        if i == 0 and end.GetImplicitHCount() == 1:
            # This is a C=NH bond with implicit H
            ignore = True
    return ignore


def has_atom_map(molecule):
    IS_MAPPED = True
    for atom in molecule.GetAtoms():
            if atom.GetMapIdx() == 0:
                IS_MAPPED = False
    return IS_MAPPED


def remove_atom_map(molecule):
    for a in molecule.GetAtoms():
        a.SetMapIdx(0)
