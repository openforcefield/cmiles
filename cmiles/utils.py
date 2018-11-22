"""
Utility functions for cmiles generator
"""
import os
import copy
import warnings
import numpy as np

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

_symbols = {'H':1,'He':2,
            'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F:':9,'Ne':10,
            'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
            'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
            'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,
            'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,
            'Lu':71,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,
            'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,'Sg':106,'Bh':107,'Hs':108,'Mt':109}
BOHR_2_ANGSTROM = 0.529177210
ANGSROM_2_BOHR = 1. / BOHR_2_ANGSTROM

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
        Can be SMILES, filename, OpenEye or RDKit molecule, JSON molecule
        for the JSON molecule, the minimum fields needed are symbols and geometry or symbols and connectivity

    Returns
    -------
    molecule: output molecule
        If has license to OpenEye, will return an OpenEye molecule. Otherwise will return a RDKit molecule if input can
        be parsed with RDKit.
    """
    if isinstance(inp_molecule, dict):
        # This is a JSON molecule
        molecule = load_mol_json(inp_molecule, backend=backend)

    elif backend == 'rdkit':
        if not has_rdkit:
            raise RuntimeError("You need to have RDKit installed to use the RDKit backend")
        molecule = _load_mol_rd(inp_molecule)

    elif backend == 'openeye':
        if not has_openeye:
            raise RuntimeError("You need to have OpenEye installed or an up-to-date license to use the openeye backend")
        molecule = _load_mol_oe(inp_molecule)
    else:
        raise RuntimeError("You must have either RDKit or OpenEye installed")
    return molecule


def _load_mol_rd(inp_molecule):

    _EXT_DISPATCH_TABLE = {'.pdb': Chem.MolFromPDBFile, '.mol2': Chem.MolFromMol2File, '.tpl': Chem.MolFromTPLFile}
    if isinstance(inp_molecule, str):
        # First check if it has a file extension
        ext = _get_extension(inp_molecule)

        if not ext:
            # This is probably a SMILES
            molecule = Chem.MolFromSmiles(inp_molecule)
            if not molecule:
                raise Warning("Could not parse molecule")
            return molecule

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


def _load_mol_oe(inp_molecule):

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


def load_mol_json(inp_molecule, backend='openeye'):
    """
    Load a molecule from JSON
    If using openeye backend, connectivity is not needed.
    The input JSON must include symbols and geometries or symbols and connectivity.
    Parameters
    ----------
    inp_molecule
    backend

    Returns
    -------

    """
    # Check fields
    if 'symbols' not in inp_molecule:
        raise KeyError("JSON input molecule must have symbols")
    if 'connectivity' not in inp_molecule:
        # Must use OpenEye.
        if not has_openeye:
            raise RuntimeError("Cannot create molecular graph from geometry without OpenEye")
        molecule = _mol_from_geometry(inp_molecule)
    elif 'connectivity' in inp_molecule:
        molecule = _mol_from_connectivity(inp_molecule, backend=backend)
    else:
        raise KeyError("Not enough information to generate a molecular graph. Input JSON molecule must have either"
                       "geometry or connectivity defined")


def _mol_from_geometry(inp_molecule):
    """
    Must have OpenEye to create molecular graph from only geometry without connectivity
    Parameters
    ----------
    inp_molecule

    Returns
    -------

    """
    if not has_openeye:
        raise RuntimeError("Must have OpenEye to generate molecular graph from geometry")
    symbols = inp_molecule['symbols']
    geometry = inp_molecule['geometry']

    molecule = oechem.OEMol()
    for s in symbols:
        molecule.NewAtom(_symbols[s])
    geometry_3n = np.array(geometry, dtype=float).reshape(int(len(geometry)/3), 3)
    if molecule.NumAtoms() != len(geometry_3n):
        raise ValueError("Number of atoms in molecule does not match length of position array")
    # convert to Angstroms
    geometry_3n = geometry_3n*BOHR_2_ANGSTROM
    molecule.SetCoords(oechem.OEFloatArray(geometry_3n))

    oechem.OEDetermineConnectivity(molecule)
    oechem.OEFindRingAtomsAndBonds(molecule)
    oechem.OEPerceiveBondOrders(molecule)
    oechem.OEAssignImplicitHydrogens(molecule)
    oechem.OEAssignFormalCharges(molecule)
    return molecule


def _mol_from_connectivity(inp_molecule, backend='openeye'):
    pass

def _get_extension(filename):
    (base, extension) = os.path.splitext(filename)
    if extension == '.gz':
        extension2 = os.path.splitext(base)[1]
        return extension2 + extension
    return extension


def is_mapped(molecule, backend='openeye'):
    IS_MAPPED = True
    for atom in molecule.GetAtoms():
        if backend == 'openeye':
            if atom.GetMapIdx() == 0:
                IS_MAPPED = False
        elif backend == 'rdkit':
            if atom.GetAtomMapNum() == 0:
                IS_MAPPED = False
        else:
            raise TypeError("Only openeye or rdkit are supported backends")
    return IS_MAPPED


def remove_map(molecule, backend='openeye'):
    """

    Parameters
    ----------
    molecule
    backend

    Returns
    -------

    """
    for a in molecule.GetAtoms():
        if backend == 'openeye':
            a.SetMapIdx(0)
        elif backend == 'rdkit':
            a.SetAtomMapNum(0)
        else:
            raise TypeError("Only openeye and rdkit are supported backends")


