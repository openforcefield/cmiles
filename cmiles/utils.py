"""
Utility functions for cmiles generator
"""
import copy
import numpy as np
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


def load_molecule(inp_molecule, backend='openeye'):
    """
    Load molecule. Input restrictive. Can use and isomeric SMILES or a JSON serialized molecule

    Parameters
    ----------
    inp_molecule: input molecule
        This can be either a SMILES with stereochemistry (isomeric SMILES) or a JSON serialized molecules.
        for the JSON molecule, the minimum fields needed are symbols, connectivity and geometry.

    Returns
    -------
    molecule: output molecule
        If has license to OpenEye, will return an OpenEye molecule. Otherwise will return a RDKit molecule if input can
        be parsed with RDKit.
    """
    # Check input
    if isinstance(inp_molecule, dict):
        # This is a JSON molecule.
        molecule = mol_from_json(inp_molecule, backend=backend)

    elif isinstance(inp_molecule, str):
        # Check for explicit H. This is not an exhaustive check but will catch many cases. It will also catch false
        # negatives so
        if inp_molecule.find('H') == -1:
            warnings.warn("{} might be missing explicit hydrogen. Double-check your input".format(inp_molecule))

        if backend == 'rdkit':
            if not has_rdkit:
                raise RuntimeError("You need to have RDKit installed to use the RDKit backend")
            molecule = Chem.MolFromSmiles(inp_molecule)
            if not molecule:
                raise ValueError("The supplied SMILES {} could not be parsed".format(inp_molecule))

        elif backend == 'openeye':
            if not has_openeye:
                raise RuntimeError("You need to have OpenEye installed or an up-to-date license to use the openeye "
                                   "backend")
            molecule = oechem.OEMol()
            if not oechem.OESmilesToMol(molecule, inp_molecule):
                raise ValueError("The supplied SMILES {} could not be parsed".format(inp_molecule))
        else:
            raise RuntimeError("You must have either RDKit or OpenEye installed")
    else:
        raise TypeError("Input must be either a SMILES string or a JSON serialized molecule")

    return molecule


def mol_from_json(inp_molecule, backend='openeye'):
    """
    Load a molecule from QCSchema
    The input JSON should use QCSchema specs (https://molssi-qc-schema.readthedocs.io/en/latest/index.html#)
    Required fields to generate CMILES identiifers are symbols, connectivity and geometry.

    Parameters
    ----------
    inp_molecule: dict
       Required keys are symbols, connectivity and/or geometry. If using RDKit as backend, must have connectivity.
    backend: str, optional. Default openeye
        Specify which cheminformatics toolkit to use. Options are openeye and rdkit.

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

    if backend == 'openeye':
        molecule = _mol_from_json_oe(symbols, connectivity, geometry)
    elif backend == 'rdkit':
        molecule = _mol_from_json_rd(symbols, connectivity, geometry)
    else:
        raise ValueError("Only openeye and rdkit backends are supported")

    return molecule


def _mol_from_json_oe(symbols, connectivity, geometry):
    """
    Generate OEMol from QCSchema molecule specs
    Parameters
    ----------
    inp_molecule: dict
        Must have symbols and connectivity and/or geometry
        Note: If geometry is given, the molecule will have a tag indicating that the goemetry came from QCSchema. This
        will ensure that the order of the atoms and configuration is not change for generation of mapped SMILES and
        isomeric SMILES.

    Returns
    -------
    molecule: OEMol

    """

    if not has_openeye:
        raise RuntimeError("You do not have OpenEye installed or do not have license to use it. Use the RDKit backend")

    molecule = oechem.OEMol()
    for s in symbols:
        molecule.NewAtom(_symbols[s])

    # Add connectivity
    for bond in connectivity:
        a1 = molecule.GetAtom(oechem.OEHasAtomIdx(bond[0]))
        a2 = molecule.GetAtom(oechem.OEHasAtomIdx(bond[1]))
        molecule.NewBond(a1, a2, bond[-1])

    # Add geometry
    if molecule.NumAtoms() != geometry.shape[0]/3:
        raise ValueError("Number of atoms in molecule does not match length of position array")

    molecule.SetCoords(oechem.OEFloatArray(geometry))
    molecule.SetDimension(3)

    # Add tag that the geometry is from JSON and shouldn't be changed.
    geom_tag = oechem.OEGetTag("json_geometry")
    molecule.SetData(geom_tag, True)
    oechem.OEDetermineConnectivity(molecule)
    oechem.OEFindRingAtomsAndBonds(molecule)
    oechem.OEPerceiveBondOrders(molecule)
    oechem.OEAssignImplicitHydrogens(molecule)
    oechem.OEAssignFormalCharges(molecule)
    oechem.OEAssignAromaticFlags(molecule)

    return molecule


def _mol_from_json_rd(symbols, connectivity, geometry):
    """
    Generate RDkit.Chem.Mol from QCSchema molecule specs.
    Parameters
    ----------
    inp_molecule: dict
        Must include symbols and connectivity. Geometry is optional. If geometry is given, stereochemistry will be taken
        from coordinates

    Returns
    -------
    molecule: rdkit.Chem.Mol
    """

    if not has_rdkit:
        raise RuntimeError("Must have RDKit installed when using the RDKit backend")
    from rdkit.Geometry.rdGeometry import Point3D

    _BO_DISPATCH_TABLE = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

    geometry = geometry.reshape(int(len(geometry)/3), 3)
    conformer = Chem.Conformer(len(symbols))
    has_geometry = True

    molecule = Chem.Mol()
    em = Chem.RWMol(molecule)
    for i, s in enumerate(symbols):
        atom = em.AddAtom(Chem.Atom(_symbols[s]))
        atom_position = Point3D(geometry[i][0], geometry[i][1], geometry[i][2])
        conformer.SetAtomPosition(atom, atom_position)

    # Add connectivity
    for bond in connectivity:
        bond_type = _BO_DISPATCH_TABLE[bond[-1]]
        em.AddBond(bond[0], bond[1], bond_type)

    molecule = em.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except:
        raise RuntimeError("Could not sanitize molecule")

    # Add coordinates
    if has_geometry:
        initial_conformer_id = molecule.AddConformer(conformer, assignId=True)
        # Assign stereochemistry from coordinates
        from rdkit.Chem import rdmolops
        rdmolops.AssignStereochemistryFrom3D(molecule, confId=initial_conformer_id, replaceExistingTags=True)
        # Add a tag to keep current order
        molecule.SetProp("_json_geometry", '1')

    return molecule

# ToDo: ordered geometry by map indices.
# Inputs can be:
#  1. qcschema and map smile
#  2. oe or rd mol with geometry.
#     a. mol can have map on the indices
#     b. If mol does not have map, must supply mapped SMILES


def is_mapped(molecule):

    if has_rdkit:
        if isinstance(molecule, Chem.Mol):
            backend = 'rdkit'
    if has_openeye:
        if isinstance(molecule, (oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase)):
            backend = 'openeye'

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


def remove_map(molecule):
    """
    Remove atom map from molecule.
    This is done for several reasons such as the
    Parameters
    ----------
    molecule

    Returns
    -------

    """
    if has_rdkit:
        if isinstance(molecule, Chem.Mol):
            backend = 'rdkit'
    if has_openeye:
        if isinstance(molecule, (oechem.OEMol, oechem.OEGraphMol, oechem.OEMolBase)):
            backend = 'openeye'
    for a in molecule.GetAtoms():
        if backend == 'openeye':
            a.SetMapIdx(0)
        elif backend == 'rdkit':
            a.SetAtomMapNum(0)
        else:
            raise TypeError("Only openeye and rdkit are supported backends")


def is_stereo_defined(molecule, backend='openeye'):
    """

    Parameters
    ----------
    molecule
    backend

    Returns
    -------

    """
    if backend == 'openeye' and has_openeye:
        if not isinstance(molecule, (oechem.OEMol, oechem.OEMolBase, oechem.OEGraphMol)):
            raise TypeError("If using openeye must have an oemol")
        stereo = _is_stereo_defined_oe(molecule)

    elif backend == 'rdkit' and has_rdkit:
        if not isinstance(molecule, Chem.Mol):
            raise TypeError("If using rdkit, must provide an rdkit.Chem.Mol")
        stereo = _is_stereo_defined_rd(molecule)

    else:
        raise TypeError("only openeye and rdkit are supported")
    return stereo


def _is_stereo_defined_oe(molecule):
    """
    Check if any stereochemistry in undefined.
    Parameters
    ----------
    molecule: OEMol

    Returns
    -------
    bool: True if all stereo chemistry is defined.
        If any stereochemsitry is undefined, raise and exception.

    """

    if has_openeye:
        if not oechem.OEChemIsLicensed():
            raise ImportError("Must have oechem License!")
    else:
        raise ImportError("Must have openeye installed")

    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()
    for atom in molecule.GetAtoms():
        if atom.IsChiral() and not atom.HasStereoSpecified(oechem.OEAtomStereo_Tetrahedral):
            # Check if handness is specified
            v = []
            for nbr in atom.GetAtoms():
                v.append(nbr)
            stereo = atom.GetStereo(v, oechem.OEAtomStereo_Tetrahedral)
            if stereo == oechem.OEAtomStereo_Undefined:
                unspec_chiral = True
                problematic_atoms.append((atom.GetIdx(), oechem.OEGetAtomicSymbol(atom.GetAtomicNum())))
    for bond in molecule.GetBonds():
        if bond.IsChiral() and not bond.HasStereoSpecified(oechem.OEBondStereo_CisTrans):
            v = []
            for neigh in bond.GetBgn().GetAtoms():
                if neigh != bond.GetEnd():
                    v.append(neigh)
                    break
            for neigh in bond.GetEnd().GetAtoms():
                if neigh != bond.GetBgn():
                    v.append(neigh)
                    break
            stereo = bond.GetStereo(v, oechem.OEBondStereo_CisTrans)

            if stereo == oechem.OEBondStereo_Undefined:
                unspec_db = True
                a1 = bond.GetBgn()
                a2 = bond.GetEnd()
                a1_idx = a1.GetIdx()
                a2_idx = a2.GetIdx()
                a1_s = oechem.OEGetAtomicSymbol(a1.GetAtomicNum())
                a2_s = oechem.OEGetAtomicSymbol(a2.GetAtomicNum())
                bond_order = bond.GetOrder()
                problematic_bonds.append((a1_idx, a1_s, a2_idx, a2_s, bond_order))
    if unspec_chiral or unspec_db:
        raise ValueError("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms,
                problematic_bonds))
    else:
        return True


def _is_stereo_defined_rd(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """

    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()
    chiral_centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    for center in chiral_centers:
        atom_id = center[0]
        if center[-1] == '?':
            unspec_chiral = True
            problematic_atoms.append((atom_id, molecule.GetAtomWithIdx(atom_id).GetSmarts()))

    # Find potential stereo bonds that are unspecified
    Chem.FindPotentialStereoBonds(molecule)
    for bond in molecule.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            unspec_db = True
            problematic_bonds.append((bond.GetBeginAtom().GetSmarts(), bond.GetSmarts(),
                                                    bond.GetEndAtom().GetSmarts()))
    if unspec_chiral or unspec_db:
        raise ValueError("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms, problematic_bonds))
    else:
        return True


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
    explicit = True
    if has_openeye:
        if isinstance(molecule, (oechem.OEMol, oechem.OEMolBase, oechem.OEGraphMol)):
            backend = 'openeye'
    if has_rdkit:
        if isinstance(molecule, Chem.Mol):
            backend = 'rdkit'
    if backend == 'openeye':
        for a in molecule.GetAtoms():
            if a.GetImplicitHCount() > 0:
                # The molecule was generated from an implicit hydrogen SMILES
                explicit = False

    if backend == 'rdkit':
        # Not implemented because I do not know how to check for this in RDKit.
        pass

    return explicit


def canonical_order_atoms_oe(molecule, in_place=True):
    """
    Canonical order of atom indices. This ensures the map indices are always in the same order.
    Parameters
    ----------
    molecule: oechem.OEMol
    in_place: bool, optional, default True
        If True, the order of atom indices will happen in place. If False, a copy of the molecule with reordered atom
        indices will be returned.

    Returns
    -------
    molecule: OEMol with canonical ordered atom indices.

    """

    if not in_place:
        molecule = copy.deepcopy(molecule)

    oechem.OECanonicalOrderAtoms(molecule)
    oechem.OECanonicalOrderBonds(molecule)

    vatm = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != oechem.OEElemNo_H:
            vatm.append(atom)
    molecule.OrderAtoms(vatm)

    vbnd = []
    for bond in molecule.GetBonds():
        if bond.GetBgn().GetAtomicNum() != oechem.OEElemNo_H and bond.GetEnd().GetAtomicNum() != oechem.OEElemNo_H:
            vbnd.append(bond)
    molecule.OrderBonds(vbnd)

    molecule.Sweep()

    for bond in molecule.GetBonds():
        if bond.GetBgnIdx() > bond.GetEndIdx():
            bond.SwapEnds()

    if not in_place:
        return molecule


def canonical_order_atoms_rd(molecule, h_last=True):
    """
    Canonical order atoms in RDKit molecule. Eaach atom in the molecule is given a map index that corresponds to the RDkit
    rank for that atom (+1). RDKit atom ranking ranks hydrogens first and then the heavy atoms. When
    h_last is set to True, the map indices are reordered to put hydrogens after the heavy atoms.
    Parameters
    ----------
    molecule: rdkit mol
    h_last: bool, optional, default is True

    Returns
    -------
    molecule: rdkit molecule with map indices that correspond to the atom canonical rank
    """

    # Check if molecule already has map. If it does, remove map because Chem.CanonicalRankAtoms uses map indices in
    # ranking
    if is_mapped(molecule):
        remove_map(molecule)

    # Add explicit hydrogen
    molecule = Chem.AddHs(molecule)
    heavy_atoms = 0
    hydrogens = 0
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    for i, j in enumerate(ranks):
        atom = molecule.GetAtomWithIdx(i)
        atom.SetAtomMapNum(j+1)
        if atom.GetAtomicNum() != 1:
            # heavy atom
            heavy_atoms +=1
        else:
            # hydrogen
            hydrogens +=1

    if h_last:
        # reorder map to put hydrogen last
        for atom in molecule.GetAtoms():
            map_idx = atom.GetAtomMapNum()
            if atom.GetAtomicNum() !=1:
                atom.SetAtomMapNum(map_idx - hydrogens)
            else:
                atom.SetAtomMapNum(map_idx + heavy_atoms)
    return molecule
