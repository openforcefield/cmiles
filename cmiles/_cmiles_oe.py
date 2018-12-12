"""

"""
from openeye import oechem
import time
from .utils import ANGSROM_2_BOHR


def set_aromaticity_mdl(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    oechem.OEClearAromaticFlags(molecule)
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModel_MDL)
    oechem.OEAssignHybridization(molecule)


def get_connectivity_table(molecule, inverse_map):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    connectivity_table = []
    for bond in molecule.GetBonds():
        order = bond.GetOrder()
        if bond.IsAromatic():
            order = 1.5
        connectivity_table.append([inverse_map[bond.GetBgnIdx()]-1, inverse_map[bond.GetEndIdx()]-1, order])

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


def to_map_ordered_geometry(molecule, atom_map):
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
    return oechem.OEHasImplicitHydrogens(molecule)