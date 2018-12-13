"""
Generate canonical, isomeric, explicit hydrogen, mapped SMILES

"""
from copy import deepcopy
import cmiles
from .utils import has_openeye, has_rdkit

if has_openeye:
    import openeye as oe
if has_rdkit:
    import rdkit as rd


def to_molecule_id(molecule_input, canonicalization='openeye', strict=True, **kwargs):
    """
    Generate a dictionary of canonical SMILES.

    This dictionary contains:
    1. canonical SMILES
    2. canonical, isomeric SMILES
    3. canonical, explicit hydrogen SMILES
    4. canonical, isomeric, explicit hydrogen SMILES
    5. canonical, isomeric, explicit hydrogen, mapped SMILES
    6. Standard InChI
    7. Standard InChI key
    8. unique protomer. This is generated with OpenEye so will only be returned if the user has openeye installed and
    an openeye license.

    The map index on the mapped SMILES is the rank order of the atoms. This SMILES can be used as a SMARTS query for
    a molecule generated from any SMILES representing the same molecule. Using a substrucutre search, you can find the
    mapping of the atom index to the map index in the mapped SMILES. This can be used to ensure all atoms in the same
    molecule have the same map indices.

    For example, methanol (`CO`) will become `[H:3][C:1]([H:4])([H:5])[O:2][H:6]`. Using substructure search on any
    methanol openeye or rdkit molecule, you will get a mapping from the atom map to the atom index.

    Below is some code using OpenEye to get the mapping:

    from openeye import oechem

    # Set up the substructure search
    ss = oechem.OESubSearch(mapped_smiles)
    oechem.OEPrepareSearch(molecule, ss)
    ss.SetMaxMatches(1)

    atom_map = {}
    for match in matches:
        for ma in match.GetAtoms():
            atom_map[ma.pattern.GetMapIdx()] = ma.target.GetIdx()


    The default option uses OpenEye canonicalization to generate these SMILES, but you can also use rdkit.


    Parameters
    ----------
    molecule: The input molecule can be a json serialized molecule or an isomeric SMILES.
              The serialized molecule must contain the following fields: symbols, xyz coordinates and a connectivity table.
              If a SMILES string is provied, it must contain stereochemistry information.
    canonicalization: str, optional, default 'openeye'
        The canonicalization backend to use for generating SMILES. Choice of 'openeye' or 'rdkit'.
        The canonicalization algorithms are different so the output will be different.
        The mapping will also be different.
    strict: bool, optional. Default True
        If true, will raise an exception if SMILES is missing explicit H.

    Returns
    -------
    cmiles: dict of identifiers

        The provenance key maps to the cmiles version and openeye or rdkit version used.

    """
    # check input and convert to oe or rdkit mol
    molecule = cmiles.utils.load_molecule(molecule_input, toolkit=canonicalization, **kwargs)

    # check for map. If map exists, remove. We only want maps generated with cmiles
    if cmiles.utils.has_atom_map(molecule):
        cmiles.utils.remove_atom_map(molecule)

    # check for explicit hydrogen
    if strict and not cmiles.utils.has_explicit_hydrogen(molecule):
        raise ValueError("Input molecule is missing explicit hydrogen")

    # Check for fully defined stereochemsitry
    if not cmiles.utils.has_stereo_defined(molecule):
        raise ValueError("Input SMILES must have full stereochemistry defined")

    molecule_ids = {}
    toolkit = cmiles.utils._set_toolkit(molecule)
    molecule_ids['canonical_smiles'] = toolkit.mol_to_smiles(molecule,
                                                             isomeric=False,
                                                             explicit_hydrogen=False,
                                                             mapped=False)
    molecule_ids['canonical_isomeric_smiles'] = toolkit.mol_to_smiles(molecule,
                                                                      isomeric=True,
                                                                      explicit_hydrogen=False,
                                                                      mapped=False)
    molecule_ids['canonical_explicit_hydrogen_smiles'] = toolkit.mol_to_smiles(molecule,
                                                                               isomeric=False,
                                                                               explicit_hydrogen=True,
                                                                               mapped=False)
    molecule_ids['canonical_isomeric_explicit_hydrogen_smiles'] = toolkit.mol_to_smiles(molecule,
                                                                                        isomeric=True,
                                                                                        explicit_hydrogen=True,
                                                                                        mapped=False)
    molecule_ids['canonical_isomeric_explicit_hydrogen_mapped_smiles'] = toolkit.mol_to_smiles(molecule, isomeric=True,
                                                                                               explicit_hydrogen=True,
                                                                                               mapped=True)

    molecule_ids['standard_inchi'], molecule_ids['inchi_key'] = to_inchi_and_key(molecule)
    molecule_ids['molecular_formula'] = cmiles.utils.mol_to_hill_molecular_formula(molecule)

    if cmiles.utils.has_rdkit:
        molecule_ids['unique_tautomer_representation'] = standardize_tautomer(molecule_ids['canonical_isomeric_smiles'])
    if cmiles.utils.has_openeye:
        molecule_ids['unique_protomer_representation'] = get_unique_protomer(molecule)

    molecule_ids['provenance'] = 'cmiles_' + cmiles.__version__ + '_{}_'.format(canonicalization) + \
                                 toolkit.toolkit.__version__

    try:
        if kwargs['permute_xyz']:
            permuted_json_mol = cmiles.utils.permute_qcschema(molecule_input, molecule_ids, canonicalization)
            return permuted_json_mol
    except KeyError:
        return molecule_ids


def to_inchi_and_key(molecule):

    # Todo can use the InChI code directly here
    # Make sure molecule is rdkit mol
    if not isinstance(molecule, rd.Chem.Mol):
        molecule = rd.Chem.MolFromSmiles(oe.oechem.OEMolToSmiles(molecule))

    inchi = rd.Chem.MolToInchi(molecule)
    inchi_key = rd.Chem.MolToInchiKey(molecule)
    return inchi, inchi_key


def to_iupac(molecule):
    if not has_openeye:
        raise ImportError("OpenEye is not installed. You can use the canonicalization='rdkit' to use the RDKit backend"
                           "The Conda recipe for cmiles installs rdkit")

    from openeye import oeiupac
    if not oeiupac.OEIUPACIsLicensed():
        raise ImportError("Must have OEIUPAC license!")
    return oeiupac.OECreateIUPACName(molecule)


def get_unique_protomer(molecule):

    molecule = deepcopy(molecule)
    # This only works for OpenEye
    # Todo There might be a way to use different layers of InChI for this purpose.
    # Not all tautomers are recognized as the same by InChI so it won't capture all tautomers.
    # But if we use only the formula and connectivity level we might be able to capture a larger subset.
    #
    if has_openeye:
        from openeye import oequacpac, oechem
    else:
        raise RuntimeError("Must have OpenEye for unique protomer feature")
    if not oechem.OEChemIsLicensed():
        raise ImportError("Must have OEChem license!")
    if not oequacpac.OEQuacPacIsLicensed():
        raise ImportError("Must have OEQuacPac license!")

    if has_rdkit:
        if isinstance(molecule, rd.Chem.rdchem.Mol):
            # convert to openeye molecule
            # Maybe we shouldn't do this.
            smiles = rd.Chem.MolToSmiles(molecule)
            molecule = oechem.OEMol()
            oechem.OESmilesToMol(molecule, smiles)

    molecule_copy = deepcopy(molecule)
    oequacpac.OEGetUniqueProtomer(molecule_copy, molecule)
    return oechem.OEMolToSmiles(molecule_copy)


def standardize_tautomer(iso_can_smi):
    """
    Standardize tautomer to one universal tautomer. Does not standardize for ionization states.
    In some cases preforms better than oequacpac.OEGetUniqueProtomer. See examples/tautomers.ipynb
    Parameters
    ----------
    iso_can_smi: str

    Returns
    -------
    std_tautomer: str
    """
    if has_rdkit:
        from rdkit.Chem import MolStandardize
    else:
        raise ImportError("Must have rdkit installed to use this function")

    std_tautomer = MolStandardize.canonicalize_tautomer_smiles(iso_can_smi)
    return std_tautomer
