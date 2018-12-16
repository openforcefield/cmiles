cmiles
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/openforcefield/cmiles.png)](https://travis-ci.org/openforcefield/cmiles)
[![codecov](https://codecov.io/gh/openforcefield/cmiles/branch/master/graph/badge.svg)](https://codecov.io/gh/openforcefield/cmiles/branch/master)

Generate canonical identifiers for chemical databases, specifically quantum
chemical data. 
 
cmiles seeks to address several issues:

1. *Link the QC molecule to its cheminformatics molecular graph to make QC data
useful to the force field and machine learning communities.*
    A QC molecule is identified by its elements and geometry. Each conformer
    is considered a separate molecule. However, from a cheminformatics perspective,
    the connectivity graph is the identity of the molecule. 
2. *Canonical identifiers.*  
    When indexing a chemical database, it is crucial that the identifier
    is canonical to reduce redundancy and search failures.
    **Problem**:
    Each toolkit has its own canonicalization algorithm. This algorithm may
    change in different versions of the toolkit
    **Solution**:
    Distribute cmiles as a Docker image with pinned dependencies.
  
3. *Canonical order of QC geometry.*
    Since QC molecules are identified by their elements and geometry, canonical
    atomic order can reduce redundancy.
    **Problem**:
    In cheminformatics toolkits, the order of atoms in a molecular graph are arbitrary.
     Therefore, every time you generate a new molecular graph, the order of the atoms can change.
    **Solution**:
    Use the toolkits canonical atomic ranking to generate a SMILES string that has
    explicit hydrogens and map indices that correspond to the canonical atomic ranking.
    Then, this mapped SMILES can be used as a SMARTS pattern to find the mapping of
    a molecular graph's these map indices to atomic indices. This map is then used
    to generate canonical order for the xyz coordinates and symbols in the QC molecule.
    `cmiles.utils` provides functions that will do this. 
    In addition, the mapped SMILES ensure that all molecular graphs generated can be mapped
    to QC geometries if those geometries are in the order of the map. 
    
4. Standardize compounds
    Provide an index to find all protomers of a molecule.
    **Problem**:
    Different protomeric states are different QC molecules. However, sometimes all
    protomers and / tautomers of a molecule are needed. SMILES strings are different
    for each protomer. While InChI standardizes molecular charge states, it only
    standardize the tautomers it recognizes and not others. Some tautomers
    it does not capture are keto-enol and enamine-imine.
    
    cmiles does not yet offer a full solution. Currently it provides the InChI, the
    [unique protomer from openeye](https://docs.eyesopen.com/toolkits/python/quacpactk/OEProtonFunctions/OEGetUniqueProtomer.html)
    and `MolStandardize` from rdkit. rdkit's solution only addresses tautomers of the 
    same charge states and openeye's solution does not capture indoles, isoindoles and 
    some mesomers. However, the union of these identifiers captures more than each individual solution. 
    
    
#### How to use cmiles
`cmiles` has one main function, `to_molecule_id(molecule)` This 
function returns a dictionary with 10 identifiers generated with either OpenEye
or RDKit. 

1. Canonical SMILES
2. Canonical, isomeric SMILES
3. Canonical, isomeric, explicit hydrogen SMILES
4. Canonical, explicit hydrogen SMILES
5. Canonical, isomeric, explicit hydrogen, mapped SMILES.
6. Molecular formula in Hill notation
7. InChI
8. InChIKey
9. unique protomer SMILES
10. Standardized tautomer SMILES

`cmiles.utils` provides functions to generate atom maps and map ordered geometries.

### Dependencies
One of the following cheminformatics toolkits:  
* openeye  
* rdkit  

### Copyright

Copyright (c) 2018, Chaya D. Stern


#### Acknowledgements
 
Project based on the 
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
