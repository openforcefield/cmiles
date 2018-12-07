"""
Generate canonical, isomeric, explicit hydrogen, mapped SMILES

"""
import warnings
import collections
from copy import deepcopy
import cmiles

HAS_OPENEYE = True
try:
    import openeye as oe
except ImportError:
    HAS_OPENEYE = False

HAS_RDKIT = True
try:
    import rdkit as rd
except ImportError:
    HAS_RDKIT = False


def to_molecule_id(molecule, canonicalization='openeye'):
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

    Returns
    -------
    cmiles: dict of identifiers

        The provenance key maps to the cmiles version and openeye or rdkit version used.

    """
    # check input and convert to oe or rdkit mol
    molecule = cmiles.utils.load_molecule(molecule, backend=canonicalization)

    # check for map. If map exists, remove. We only want maps generated with cmiles
    if cmiles.utils.is_mapped(molecule):
        cmiles.utils.remove_map(molecule)
    # check for explicit hydrogen
    if not cmiles.utils.has_explicit_hydrogen(molecule, backend=canonicalization):
        warnings.warn("Input molecule is missing explicit hydrogen. In some cases this can lead to hydrogen placement "
                      "that are different than you intended")

    # check for fully defined stereochemistry


    smiles = {}
    if canonicalization == 'openeye':
        if not HAS_OPENEYE:
            raise RuntimeError("You do not have OpenEye installed or you are missing the license.")
        smiles['canonical_smiles'] = to_canonical_smiles_oe(molecule, isomeric=False, explicit_hydrogen=False,
                                                             mapped=False)
        smiles['canonical_isomeric_smiles'] = to_canonical_smiles_oe(molecule, isomeric=True, explicit_hydrogen=False,
                                                                      mapped=False)
        smiles['canonical_isomeric_explicit_hydrogen_smiles'] = to_canonical_smiles_oe(molecule, isomeric=True,
                                                                                       explicit_hydrogen=True,
                                                                                        mapped=False)
        smiles['canonical_explicit_hydrogen_smiles'] = to_canonical_smiles_oe(molecule, isomeric=False,
                                                                               explicit_hydrogen=True, mapped=False)
        smiles['canonical_isomeric_explicit_hydrogen_mapped_smiles'] = to_canonical_smiles_oe(molecule, isomeric=True,
                                                                                               explicit_hydrogen=True,
                                                                                               mapped=True)
        smiles['unique_protomer_representation'] = get_unique_protomer(molecule)
        smiles['provenance'] = 'cmiles_' + cmiles.__version__ + '_openeye_' + oe.__version__
    elif canonicalization == 'rdkit':
        if not HAS_RDKIT:
           raise RuntimeError("You do not have RDKit installed")
        smiles['canonical_smiles'] = to_canonical_smiles_rd(molecule, isomeric=False, explicit_hydrogen=False,
                                                             mapped=False)
        smiles['canonical_isomeric_smiles'] = to_canonical_smiles_rd(molecule, isomeric=True, explicit_hydrogen=False,
                                                                      mapped=False)
        smiles['canonical_isomeric_explicit_hydrogen_smiles'] = to_canonical_smiles_rd(molecule, isomeric=True,
                                                                                        explicit_hydrogen=True,
                                                                                        mapped=False)
        smiles['canonical_explicit_hydrogen_smiles'] = to_canonical_smiles_rd(molecule, isomeric=False,
                                                                               explicit_hydrogen=True, mapped=False)
        smiles['canonical_isomeric_explicit_hydrogen_mapped_smiles'] = to_canonical_smiles_rd(molecule, isomeric=True,
                                                                                               explicit_hydrogen=True,
                                                                                               mapped=True)
        if HAS_OPENEYE:
            smiles['unique_protomer_representation'] = get_unique_protomer(molecule)
        smiles['provenance'] = 'cmiles_' + cmiles.__version__ + '_rdkit_' + rd.__version__
    else:
        raise TypeError("canonicalization must be either 'openeye' or 'rdkit'")

    smiles['standard_inchi'], smiles['inchi_key'] = to_inchi_and_key(molecule)
    smiles['molecular_formula'] = molecular_formula(molecule)

    return smiles


def to_inchi_and_key(molecule):

    # Todo can use the InChI code directly here
    # Make sure molecule is rdkit mol
    if not isinstance(molecule, rd.Chem.Mol):
        molecule = rd.Chem.MolFromSmiles(oe.oechem.OEMolToSmiles(molecule))

    inchi = rd.Chem.MolToInchi(molecule)
    inchi_key = rd.Chem.MolToInchiKey(molecule)
    return inchi, inchi_key


def to_canonical_smiles_rd(molecule, isomeric, explicit_hydrogen, mapped):
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

    molecule = deepcopy(molecule)
    # Check molecule instance
    if not isinstance(molecule, (rd.Chem.rdchem.Mol, rd.Chem.Mol)):
        warnings.warn("molecule should be rdkit molecule. Converting to rdkit molecule", UserWarning)
        # Check if OpenEye Mol and convert to RDKit molecule
        if HAS_OPENEYE:
            if not oe.OEChemIsLicensed():
                raise ImportError("Need license for oechem!")
            if isinstance(molecule, oe.oechem.OEMol):
                # Convert OpenEye Mol to RDKit molecule
                molecule = rd.Chem.MolFromSmiles(oe.oechem.OEMolToSmiles(molecule))
            if molecule is None:
                raise RuntimeError("RDKit could not parse SMILES")
        else:
            raise RuntimeError('Molecule needs to be an RDKit molecule or you must have OpenEye installed')

    if explicit_hydrogen:
        # Add explicit hydrogens
        molecule = rd.Chem.AddHs(molecule)
    if not explicit_hydrogen:
        molecule = rd.Chem.RemoveHs(molecule)

    try:
        json_geometry = int(molecule.GetProp("_json_geometry"))
    except KeyError:
        json_geometry = False

    if isomeric and not cmiles.utils.has_stereo_defined(molecule, backend='rdkit'):
        raise ValueError("Some stereochemistry is not defined")

    # Get canonical order for map
    if mapped:
        if json_geometry:
            # keep original ordering
            for i in range(molecule.GetNumAtoms()):
                molecule.GetAtomWithIdx(i).SetAtomMapNum(i+1)
        else:
            molecule = cmiles.utils.canonical_order_atoms_rd(molecule)

    smiles = rd.Chem.MolToSmiles(molecule, allHsExplicit=explicit_hydrogen, isomericSmiles=isomeric, canonical=True)
    return smiles


def to_canonical_smiles_oe(molecule, isomeric, explicit_hydrogen, mapped):
    """
    Generate canonical SMILES with OpenEye.
    If Isomeric is True, this function will check if a conformer exists. If there is no conformer, oeomega will be used
    to generate a conformer so that stereochemistry can be perceived from the 3D conformation.

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
    molecule = deepcopy(molecule)
    if not HAS_OPENEYE:
        raise ImportError("OpenEye is not installed. You can use the canonicalization='rdkit' to use the RDKit backend"
                           "The Conda recipe for cmiles installs rdkit")

    from openeye import oechem
    if not oechem.OEChemIsLicensed():
        raise ImportError("Need Licence for OEChem!")

    # check molecule
    if not isinstance(molecule, (oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase)):
        raise TypeError("Must use oemol for openeye backend")
    molecule = oechem.OEMol(molecule)

    if explicit_hydrogen:
        oechem.OEAddExplicitHydrogens(molecule)

    # Generate conformer for canonical order
    # First check if geometry from JSON exists
    try:
        JSON_geometry = molecule.GetData('json_geometry')
    except ValueError:
        JSON_geometry = False

    if isomeric:
        oechem.OEPerceiveChiral(molecule)
        oechem.OE3DToAtomStereo(molecule)
        oechem.OE3DToBondStereo(molecule)
        if not cmiles.utils.has_stereo_defined(molecule, backend='openeye'):
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

    # Add tags to molecule
    if not JSON_geometry:
        cmiles.utils.canonical_order_atoms_oe(molecule)
        # canonical order of atoms if input was SMILES. For JSON serialized molecule, keep the original order.
    for atom in molecule.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)

    if mapped and not explicit_hydrogen:
        raise Warning("Tagged SMILES must include hydrogens to retain order")

    if mapped and not isomeric:
        raise Warning("Tagged SMILES must include stereochemistry ")

    return oechem.OEMolToSmiles(molecule)


def to_iupac(molecule):
    if not HAS_OPENEYE:
        raise ImportError("OpenEye is not installed. You can use the canonicalization='rdkit' to use the RDKit backend"
                           "The Conda recipe for cmiles installs rdkit")

    from openeye import oeiupac
    if not oeiupac.OEIUPACIsLicensed():
        raise ImportError("Must have OEIUPAC license!")
    return oeiupac.OECreateIUPACName(molecule)


def molecular_formula(molecule):
    """
    Generate Hill sorted empirical formula. Hill sorted first lists C and H and then all other symbols in alphabetical
    order
    Parameters
    ----------
    molecule: openeye or rdkit molecule

    Returns
    -------
    hill sorted empirical formula
    """

    # check molecule
    if HAS_OPENEYE and isinstance(molecule, (oe.OEMol, oe.OEGraphMol, oe.OEMolBase )):
        # use openeye
        if not oe.oechem.OEHasExplicitHydrogens(molecule):
            molecule = deepcopy(molecule)
            #ToDo This only checks for at least one Hydrogen so it might not be adequate.
            oe.oechem.OEAddExplicitHydrogens(molecule)
        symbols = [oe.oechem.OEGetAtomicSymbol(a.GetAtomicNum()) for a in molecule.GetAtoms()]

    elif HAS_RDKIT and isinstance(molecule, rd.Chem.Mol):
        # use rdkit
        molecule = rd.Chem.AddHs(deepcopy(molecule))
        symbols = [a.GetSymbol() for a in molecule.GetAtoms()]
    else:
        raise TypeError("Only openeye and rdkit molecules are supported molecules")

    count = collections.Counter(x.title() for x in symbols)

    hill_sorted = []
    for k in ['C', 'H']:
        # remove C and H from count
        if k in count:
            c = count.pop(k)
            hill_sorted.append(k)
            if c > 1:
                hill_sorted.append(str(c))

    for k in sorted(count.keys()):
        c = count[k]
        hill_sorted.append(k)
        if c > 1:
            hill_sorted.append(str(c))

    return "".join(hill_sorted)


def get_unique_protomer(molecule):

    molecule = deepcopy(molecule)
    # This only works for OpenEye
    # Todo There might be a way to use different layers of InChI for this purpose.
    # Not all tautomers are recognized as the same by InChI so it won't capture all tautomers.
    # But if we use only the formula and connectivity level we might be able to capture a larger subset.
    #
    if HAS_OPENEYE:
        from openeye import oequacpac, oechem
    else:
        raise RuntimeError("Must have OpenEye for unique protomer feature")
    if not oechem.OEChemIsLicensed():
        raise ImportError("Must have OEChem license!")
    if not oequacpac.OEQuacPacIsLicensed():
        raise ImportError("Must have OEQuacPac license!")

    if isinstance(molecule, rd.Chem.rdchem.Mol):
        # convert to openeye molecule
        smiles = rd.Chem.MolToSmiles(molecule)
        molecule = oechem.OEMol()
        oechem.OESmilesToMol(molecule, smiles)

    molecule_copy = deepcopy(molecule)
    oequacpac.OEGetUniqueProtomer(molecule_copy, molecule)
    return oechem.OEMolToSmiles(molecule_copy)
