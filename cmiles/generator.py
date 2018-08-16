"""
generator.py
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
    Generate canonical SMILES. The default options will generate isomeric, explicit hydrogen mapped SMILES with
    OpenEye's canonicalization algorithm.

    When using rdkit's canonicalization, chiral centers will always be assigned the CW flag (Clockwise) and stereo bonds
    will be assigned E-isomer.

    Parameters
    ----------
    molecule:
    isomeric: bool, optional, default True
        If true, SMILES will have stereochemistry specified if the molecule has stereochemical information
    explicit_h: bool, optional, default True
        If True, SMILES will have explicit hydrogen.
    mapped: bool, optional, default True
        If True, SMILES will have map of index of atom (+1 because the map is 1 indexed)
    canonicalization: str, optional, default 'openeye'
        The canonicalization backend to use for generating SMILES. Choice of 'openeye' or 'rdkit'

    Returns
    -------
    cmiles: str
        The canonical SMILES

    """
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
        If True, SMILES will have isomeric information if molecule has isomeric information.
    explicit_h: bool
        If True, SMILES will have explicit hydrogens
    mapped: bool
        If True, SMILES will have map indices

    Returns
    -------
    smiles: str
        The canonical SMILES

    """
    # Check RDKit version
    import rdkit
    if rdkit.__version__ != '2018.03.3':
        raise RuntimeError("RDKit version must be 2018.03.3")

    molecule = deepcopy(molecule)
    # Check molecule instance
    if not isinstance(molecule, rd.Chem.rdchem.Mol):
        warnings.warn("molecule should be rdkit molecule. Converting to rdkit molecule", UserWarning)
        # Check if OpenEye Mol and convert to RDKit molecule
        if HAS_OPENEYE and isinstance(molecule, openeye.OEMol):
            # Convert OpenEye Mol to RDKit molecule
            molecule = rd.Chem.MolFromSmiles(openeye.oechem.OEMolToSmiles(molecule))
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


def to_canonical_smiles_oe(molecule, isomeric, explicit_hydrogen, mapped):
    """
    Generate canonical SMILES with OpenEye.
    If Isomeric is True, this function will check if a conformer exists. If there is not conformer, oeomega will be used
    to generate a conformer so that stereochemistry can be perceived from the 3D conformation.

    Parameters
    ----------
    molecule
    isomeric
    explicit_hydrogen
    mapped

    Returns
    -------

    """
    if not HAS_OPENEYE:
        raise ImportError("OpenEye is not installed. You can use the canonicalization='rdkit' to use the RDKit backend"
                           "The Conda recipe for cmiles installs rdkit")
    if openeye.__version__ != '2018.Feb.1':
        raise RuntimeError("Must use OpeneEye version 2018.Feb.1")
    from openeye import oechem

    # check molecule
    if not isinstance(molecule, oechem.OEMol):
        warnings.warn("molecule must be OEMol. Converting to OEMol", UserWarning)
        rd_mol = molecule
        molecule = oechem.OEMol()
        oechem.OEParseSmiles(molecule, rd.Chem.MolToSmiles(rd_mol))
    molecule = oechem.OEMol(molecule)

    if explicit_hydrogen:
        oechem.OEAddExplicitHydrogens(molecule)

    # Generate conformer for canonical order
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

