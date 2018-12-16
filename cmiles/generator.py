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
    Generate a dictionary of canonical identifiers

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
    9. standardized tautomer. This is generated with RDKit.
    10. Hill molecular formula
    11. provenance:  the version of cmiles and toolkits used.

    The map index on the mapped SMILES is the rank order of the atoms. This SMILES can be used as a SMARTS query for
    a molecule generated from any SMILES representing the same molecule. Using a substrucutre search, you can find the
    mapping of the atom index to the map index in the mapped SMILES. This can be used to ensure all atoms in the same
    molecule have the same map indices.

    The utils moldule include functions, such as get_atom_map and mol_to_mapped_ordered_qcschema to get the atom map and
    generate a qcschema() molecule with the geometry and symbols ordered according to the mapped SMILES.

    The default option uses OpenEye canonicalization to generate these SMILES, but you can also use rdkit.


    Parameters
    ----------
    molecule: The input molecule can be a json serialized molecule or an isomeric SMILES.
              The serialized molecule must contain the following fields: symbols, geometry and a connectivity table.
              see QCSchema specs: https://molssi-qc-schema.readthedocs.io/en/latest/index.html#
              If a SMILES string is provided, it must contain stereochemistry information.
    canonicalization: str, optional, default 'openeye'
        Different toolkits have different canonicalization algorithms so the canonical SMILES will be different depending
        on the toolkit used. Currently, cmiles supports `openeye` and `rdkit`
        The mapping will also be different.
    strict: bool, optional. Default True
        If true, will raise an exception if SMILES is missing explicit H.
    ** permute_xyz: bool, optional, default False
        Only use if input molecule is in QCSchema format.
        If True, the geometry will be permuted to reflect the canononical
        atom order in the mapped SMILES. This function will return the permuted QCSchema. cmiles identifiers will be in
        the `identifiers` field
        If False, the map indices in the mapped SMILES will reflect the order of the atoms in the input QCSchema.

    Returns
    -------
    cmiles: dict of identifiers

        If `permute_xyz`, will return permuted qcschema with cmiles identifiers in `identifiers` field.

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
            permuted_json_mol = cmiles.utils.permute_qcschema(molecule_input, molecule_ids, toolkit=canonicalization)
            return permuted_json_mol
        else:
            return molecule_ids

    except KeyError:
        return molecule_ids


def to_inchi_and_key(molecule):
    """
    Generate inchi and inchikey. Uses RDKit which uses the inchi API

    Parameters
    ----------
    molecule: rdkit.Chem.Mol
        If an oechem.OEMol is given, will convert it to an rdkit.Chem.Mol

    Returns
    -------
    tuple (inchi, inchi_key)

    """

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
    """
    Generate unique protomer for all tuatomers and charge states of the moelcule.
    Requires openeye license.

    Parameters
    ----------
    molecule: oechem.OEMol
        Will convert rdkit.Chem.Mol to oechem.OEMol if openeye is installed and license is valid

    Returns
    -------
    unique_protomer: str

    """

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
