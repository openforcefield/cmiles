"""
Utility functions for cmiles generator
"""
import numpy as np

try:
    from rdkit import Chem
    has_rdkit = True
except ImportError:
    has_rdkit = False

try:
    from openeye import oechem
    if not oechem.OEChemIsLicensed():
        has_openeye = False
    has_openeye = True
except ImportError:
    has_openeye = False

if not has_openeye and not has_rdkit:
    raise ImportError("Must have openeye or rdkit installed")

_symbols = {'H': 1, 'He': 2,
            'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13,' Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51,
            'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62,
            'Eu': 63,' Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
            'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po':84,
            'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,' Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
            'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109}
BOHR_2_ANGSTROM = 0.529177210
ANGSROM_2_BOHR = 1. / BOHR_2_ANGSTROM


def load_molecule(inp_molecule, toolkit='openeye', **kwargs):
    """
    Load molecule. Input is restrictive. Can use and isomeric SMILES or a JSON serialized molecule

    Parameters
    ----------
    inp_molecule: input molecule
        This can be either a SMILES with stereochemistry (isomeric SMILES) or a JSON serialized molecules.
        for the JSON molecule, the minimum fields needed are symbols, connectivity and geometry.
    toolkit: str, optional, default openeye.
        Which cheminformatics toolkit to use

    Returns
    -------
    molecule: oemol or rdkit molecule
    """

    # Check input
    if isinstance(inp_molecule, dict):
        # This is a JSON molecule.
        molecule = mol_from_json(inp_molecule, toolkit=toolkit, **kwargs)

    elif isinstance(inp_molecule, str):
        if toolkit == 'openeye' and has_openeye:
            molecule = oechem.OEMol()
            if not oechem.OESmilesToMol(molecule, inp_molecule):
                raise ValueError("The supplied SMILES {} could not be parsed".format(inp_molecule))
        elif toolkit == 'rdkit' and has_rdkit:
            molecule = Chem.MolFromSmiles(inp_molecule)
            if not molecule:
                raise ValueError("The supplied SMILES {} could not be parsed".format(inp_molecule))
        else:
            raise ValueError("Only openeye and rdkit toolkits are supported")
    else:
        raise ValueError("Only QCSchema serialized molecule or an isomric SMILES are valid inputs")

    return molecule


def mol_from_json(inp_molecule, toolkit='openeye', **kwargs):
    """
    Load a molecule from QCSchema
    The input JSON should use QCSchema specs (https://molssi-qc-schema.readthedocs.io/en/latest/index.html#)
    Required fields to generate CMILES identiifers are symbols, connectivity and geometry.

    Parameters
    ----------
    inp_molecule: dict
       Required keys are symbols, connectivity and/or geometry. If using RDKit as backend, must have connectivity.
    toolkit: str, optional. Default openeye
        Specify which cheminformatics toolkit to use. Options are openeye and rdkit.
    permute_xyz: bool, optional, default False
        If False, will add flag to molecule such that the mapped SMILES retains the order of serialized geometry. If True,
        mapped SMILES will be in canonical order and serialized geometry will have to be reordered.

    Returns
    -------
    molecule: Either OEMol or rdkit.Chem.Mol

    """
    # Check fields
    required_fields = ['symbols', 'geometry', 'connectivity']
    for key in required_fields:
        if key not in inp_molecule:
            raise KeyError("input molecule must have {}".format(key))

    symbols = inp_molecule['symbols']
    connectivity = inp_molecule['connectivity']

    # convert to Angstrom.
    geometry = np.asarray(inp_molecule['geometry'], dtype=float)*BOHR_2_ANGSTROM
    if len(symbols) != geometry.shape[0]/3:
        raise ValueError("Number of atoms in molecule does not match length of position array")

    if toolkit == 'openeye' and has_openeye:
        import cmiles._cmiles_oe as mol_toolkit
    elif toolkit == 'rdkit' and has_rdkit:
        import cmiles._cmiles_rd as mol_toolkit
    else:
        raise ValueError("Only openeye and rdkit backends are supported")

    molecule = mol_toolkit.mol_from_json(symbols, connectivity, geometry, **kwargs)

    return molecule


def to_map_ordered_qcschema(molecule, molecule_ids, multiplicity=1):
    """

    Parameters
    ----------
    molecule
    molecule_ids
    multiplicity

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    mapped_smiles = molecule_ids['canonical_isomeric_explicit_hydrogen_mapped_smiles']
    atom_map = toolkit.get_atom_map(molecule, mapped_smiles)

    connectivity = get_connectivity_table(molecule, atom_map)
    symbols, geometry = toolkit.get_map_ordered_geometry(molecule, atom_map)
    charge = get_charge(molecule)

    qcschema_mol = {'symbols':symbols, 'geometry':geometry, 'connectitity_table': connectivity,
                    'molecular_charge': charge, 'molecular_multiplicity': multiplicity, 'identifiers': molecule_ids}

    return qcschema_mol


def get_atom_map(molecule, mapped_smiles):
    """

    Parameters
    ----------
    molecule
    mapped_smiles

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    atom_map = toolkit.get_atom_map(molecule, mapped_smiles)
    return atom_map


def get_connectivity_table(molecule, atom_map):
    """

    Parameters
    ----------
    molecule
    atom_map

    Returns
    -------

    """

    toolkit = _set_toolkit(molecule)
    inverse_map = dict(zip(atom_map.values(), atom_map.keys()))
    return toolkit.get_connectivity_table(molecule, inverse_map)


def permute_qcschema(json_mol, molecule_ids, toolkit='openeye'):
    """

    Parameters
    ----------
    json_mol
    molecule_ids

    Returns
    -------

    """
    molecule = mol_from_json(json_mol, toolkit=toolkit)
    ordered_qcschema = to_map_ordered_qcschema(molecule, molecule_ids, json_mol['molecular_multiplicity'])

    return ordered_qcschema


def has_atom_map(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    return toolkit.has_atom_map(molecule)


def remove_atom_map(molecule):
    """
    Remove atom map from molecule.
    This is done for several reasons such as the
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    toolkit.remove_atom_map(molecule)


def has_stereo_defined(molecule):
    """
    ToDo check for incorrect steroe (OEMDLHasIncorrectBondStereo)

    Parameters
    ----------
    molecule
    backend

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    return toolkit.has_stereo_defined(molecule)


def has_explicit_hydrogen(molecule):
    """
    Check if molecule has explicit hydrogen. In OpenEye, molecules created from explicit hydrogen SMILES will have
    explicit hydrogen.
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    toolkit = _set_toolkit(molecule)
    return toolkit.has_explicit_hydrogen(molecule)


def get_charge(molecule):

    charge = 0
    for atom in molecule.GetAtoms():
        charge += atom.GetFormalCharge()
    return charge


def _set_toolkit(molecule):

    if has_openeye and isinstance(molecule, (oechem.OEMol, oechem.OEMol, oechem.OEGraphMol)):
        import cmiles._cmiles_oe as toolkit
    elif has_rdkit and isinstance(molecule, Chem.rdchem.Mol):
        import cmiles._cmiles_rd as toolkit
    else:
        raise RuntimeError("Must have openeye or rdkit installed")
    return toolkit
