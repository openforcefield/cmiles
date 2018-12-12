"""

"""
from rdkit import Chem


def set_aromaticity_mdl(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    Chem.SanitizeMol(molecule, Chem.SANITIZE_ALL^Chem.SANITIZE_SETAROMATICITY)
    Chem.SetAromaticity(molecule, Chem.AromaticityModel.AROMATICITY_MDL)


def get_connectivity_table(molecule, inverse_map):
    """

    Parameters
    ----------
    molecule
    inverse_map

    Returns
    -------

    """
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
