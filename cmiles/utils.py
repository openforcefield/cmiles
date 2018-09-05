"""
Utility functions for cmiles generator
"""
import os
import copy
import warnings

try:
    from rdkit import Chem
    has_rdkit = True
except ImportError:
    has_rdkit = False

try:
    from openeye import oechem
    has_openeye = True
except ImportError:
    has_openeye = False


def generate_conformers(molecule, max_confs=800, strict_stereo=True, ewindow=15.0, rms_threshold=1.0, strict_types=True,
                        copy=True, canon_order=True):
    """Generate conformations for the supplied molecule
    Parameters
    ----------
    molecule : OEMol
        Molecule for which to generate conformers
    max_confs : int, optional, default=800
        Max number of conformers to generate.  If None, use default OE Value.
    strict_stereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
    strict_types : bool, optional, default=True
        If True, requires that Omega have exact MMFF types for atoms in molecule; otherwise, allows the closest atom
        type of the same element to be used.
    Returns
    -------
    molcopy : OEMol
        A multi-conformer molecule with up to max_confs conformers.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """
    try:
        from openeye import oechem, oeomega
    except ImportError:
        raise Warning("Could not import OpenEye. Need license for OpenEye!")
    if copy:
        molcopy = oechem.OEMol(molecule)
    else:
        molcopy = molecule
    omega = oeomega.OEOmega()

    # These parameters were chosen to match http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(True)
    omega.SetCanonOrder(canon_order)

    omega.SetSampleHydrogens(True)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetEnergyWindow(ewindow)
    omega.SetRMSThreshold(rms_threshold)  # Word to the wise: skipping this step can lead to significantly different charges!

    omega.SetStrictStereo(strict_stereo)
    omega.SetStrictAtomTypes(strict_types)

    omega.SetIncludeInput(False)  # don't include input
    if max_confs is not None:
        omega.SetMaxConfs(max_confs)

    status = omega(molcopy)  # generate conformation
    if not status:
        raise(RuntimeError("omega returned error code %d" % status))

    return molcopy


def load_molecule(inp_molecule, backend='openeye'):
    """
    Load molecule. Input is very permissive. Can use SMILES, SMARTS, and file formats that OpenEye or RDKit can parse.

    Parameters
    ----------
    inp_molecule: input molecule
        Can be SMILES, filename, OpenEye or RDKit molecule

    Returns
    -------
    molecule: output molecule
        If has license to OpenEye, will return an OpenEye molecule. Otherwise will return a RDKit molecule if input can
        be parsed with RDKit.
    """

    if backend == 'rdkit':
        if not has_rdkit:
            raise RuntimeError("You need to have RDKit installed to use the RDKit backend")
        molecule = load_mol_rd(inp_molecule)

    if backend == 'openeye':
        if not has_openeye:
            raise RuntimeError("You need to have OpenEye installed or an up-to-date license to use the openeye backend")
        molecule = load_mol_oe(inp_molecule)
    return molecule


def load_mol_rd(inp_molecule):

    _EXT_DISPATCH_TABLE = {'.pdb': Chem.MolFromPDBFile, '.mol2': Chem.MolFromMol2File, '.tpl': Chem.MolFromTPLFile}
    if isinstance(inp_molecule, str):
        # First check if it has a file extension
        ext = _get_extension(inp_molecule)

        if not ext:
            # This is probably a SMILES
            molecule = Chem.MolFromSmiles(inp_molecule)
            if not molecule:
                raise Warning("Could not parse molecule")

        # Try loading string as file
        try:
            molecule = _EXT_DISPATCH_TABLE[ext](inp_molecule)
        except KeyError:
            raise KeyError("Could not parse {}".format(ext))

    if isinstance(inp_molecule, Chem.Mol):
        molecule = copy.deepcopy(inp_molecule)
    if has_openeye and not isinstance(inp_molecule, Chem.Mol):
        # Check if OpenEye molecule
        if isinstance(inp_molecule, ((oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase))):
            warnings.warn("Cannot use RDKit backend with OpenEye molecule. Converting oemol to rdkit mol")
            molecule = Chem.MolFromSmiles(oechem.OEMolToSmiles(inp_molecule))

    return molecule


def load_mol_oe(inp_molecule):

    molecule = oechem.OEMol()
    if isinstance(inp_molecule, str):
        # First check if it has a file extension
        ext = _get_extension(inp_molecule)

        if not ext:
            # This is probably a SMILES
            oechem.OEParseSmiles(molecule, inp_molecule)
            if not molecule:
                raise Warning("Could not parse molecule")
            return molecule

        ifs = oechem.oemolistream()
        if not ifs.open(inp_molecule):
            raise Warning("OpenEye could not open File")
        for mol in ifs.GetOEMols():
            molecule = oechem.OEMol(mol)

    if isinstance(inp_molecule, (oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase)):
        molecule = copy.deepcopy(inp_molecule)

    return molecule


def _get_extension(filename):
    (base, extension) = os.path.splitext(filename)
    if extension == '.gz':
        extension2 = os.path.splitext(base)[1]
        return extension2 + extension
    return extension

