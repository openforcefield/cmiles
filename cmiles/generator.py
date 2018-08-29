"""
Generate canonical, isomeric, explicit hydrogen, mapped SMILES

"""
import warnings
from copy import deepcopy
import cmiles as c
import rdkit as rd

HAS_OPENEYE = True
try:
    import openeye
except ImportError:
    HAS_OPENEYE = False


def to_canonical_smiles(molecule, canonicalization='openeye'):
    """
    Generate a dictionary of canonical SMILES.

    This dictionary contains:
    1. canonical SMILES
    2. canonical, isomeric SMILES
    3. canonical, explicit hydrogen SMILES
    4. canonical, isomeric, explicit hydrogen SMILES
    5. canonical, isomeric, explicit hydrogen, mapped SMILES

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
    molecule: openeye or rdkit molecule to generate canonical SMILES.
        Use an openeye molecule if using openey canonicalization and rdkit molecule if you are using rdkit canonicalization

    canonicalization: str, optional, default 'openeye'
        The canonicalization backend to use for generating SMILES. Choice of 'openeye' or 'rdkit'.
        The canonicalization algorithms are different so the output will be different.

    Returns
    -------
    cmiles: dict
        The canonical SMILES
        The provenance key maps to the cmiles version and openeye or rdkit version used.

    """
    # check input and convert to oe or rdkit mol
    molecule = c.utils.load_molecule(molecule, backend=canonicalization)
    smiles = {}
    if canonicalization == 'openeye':
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
        smiles['provenance'] = 'cmiles_' + c.__version__ + '_openeye_' + openeye.__version__
    elif canonicalization == 'rdkit':
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
        smiles['provenance'] = 'cmiles_' + c.__version__ + '_rdkit_' + rd.__version__
    else:
        raise TypeError("canonicalization must be either 'openeye' or 'rdkit'")

    return smiles


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
    if not isinstance(molecule, rd.Chem.rdchem.Mol):
        warnings.warn("molecule should be rdkit molecule. Converting to rdkit molecule", UserWarning)
        # Check if OpenEye Mol and convert to RDKit molecule
        if HAS_OPENEYE and isinstance(molecule, openeye.OEMol):
            # Convert OpenEye Mol to RDKit molecule
            molecule = rd.Chem.MolFromSmiles(openeye.oechem.OEMolToSmiles(molecule))
            if molecule is None:
                raise RuntimeError("RDKit could not parse SMILES")
        else:
            raise RuntimeError('Molecule needs to be an RDKit molecule or you must have OpenEye installed')

    if explicit_hydrogen:
        # Add explicit hydrogens
        molecule = rd.Chem.AddHs(molecule)

    if isomeric:
        # Make sure molecule has isomeric information
        # If molecule already has isomeric information, keep it.

        # First find chiral centers
        chiral_centers = rd.Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
        for center in chiral_centers:
            atom_id = center[0]
            if center[-1] == '?':
                # If chirality is unspecified, assign clockwise chiral flag
                molecule.GetAtomWithIdx(atom_id).SetChiralTag(rd.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)

        # Find potential stereo bonds and set to E-isomer if unspecified
        rd.Chem.FindPotentialStereoBonds(molecule)
        for bond in molecule.GetBonds():
            if bond.GetStereo() == rd.Chem.BondStereo.STEREOANY:
                bond.SetStereo(rd.Chem.BondStereo.STEREOE)

    # Get canonical order for map
    if mapped:
        ranks = list(rd.Chem.CanonicalRankAtoms(molecule, breakTies=True))
        for i, j in enumerate(ranks):
            molecule.GetAtomWithIdx(i).SetAtomMapNum(j+1)

    smiles = rd.Chem.MolToSmiles(molecule, allHsExplicit=explicit_hydrogen, isomericSmiles=isomeric, canonical=True)
    return smiles


def to_canonical_smiles_oe(molecule, isomeric, explicit_hydrogen, mapped, generate_conformer=True):
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
    if not HAS_OPENEYE:
        raise ImportError("OpenEye is not installed. You can use the canonicalization='rdkit' to use the RDKit backend"
                           "The Conda recipe for cmiles installs rdkit")

    from openeye import oechem

    # check molecule
    if not isinstance(molecule, (oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase)):
        warnings.warn("molecule must be OEMol. Converting to OEMol", UserWarning)
        rd_mol = molecule
        molecule = oechem.OEMol()
        oechem.OEParseSmiles(molecule, rd.Chem.MolToSmiles(rd_mol))
    molecule = oechem.OEMol(molecule)

    if explicit_hydrogen:
        oechem.OEAddExplicitHydrogens(molecule)

    # Generate conformer for canonical order
    if generate_conformer:
        try:
            molecule = c.utils.generate_conformers(molecule, max_confs=1, strict_stereo=False, strict_types=False)
        except RuntimeError:
            warnings.warn("Omega failed to generate a conformer. Smiles may be missing stereochemistry and the map index will"
                          "not be in canonical order.")

    if isomeric:
        oechem.OEPerceiveChiral(molecule)
        oechem.OE3DToAtomStereo(molecule)
        oechem.OE3DToBondStereo(molecule)

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
    for atom in molecule.GetAtoms():
        atom.SetMapIdx(atom.GetIdx() + 1)

    if mapped and not explicit_hydrogen:
        raise Warning("Tagged SMILES must include hydrogens to retain order")

    if mapped and not isomeric:
        raise Warning("Tagged SMILES must include stereochemistry ")

    # add tag to data
    tag = oechem.OEGetTag("has_map")
    molecule.SetData(tag, bool(True))

    return oechem.OEMolToSmiles(molecule)

