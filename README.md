cmiles
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/openforcefield/cmiles.png)](https://travis-ci.org/openforcefield/cmiles)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/cmiles/branch/master)
[![codecov](https://codecov.io/gh/openforcefield/cmiles/branch/master/graph/badge.svg)](https://codecov.io/gh/openforcefield/cmiles/branch/master)

Generate canonical, isomeric, explicit hydrogen, mapped SMILES.

This package pins an OpenEye and RDKit version to SMILES canonicalization.
This is to ensure that the same SMILES is always generated for a molecule.
This is crucial when curating databases of molecule. 

In addition, cmiles will generate explicit hydrogen mapped SMILES. The map
indices gives the order of the atoms in the molecule. 
######  Why is this important? 
Indices in molecular graphs are arbitrary. Therefore, every time you 
generate a new molecular graph, the order of the atoms can change. The 
mapped SMILES can be used as a SMARTS pattern to do a substructure search
on to get a mapping from the atom indices to the atom map in the SMARTS.
This mapping can be used to ensure the same ordering of any molecular 
graph. Below is some example code using OpenEye:

```
from openeye import oechem

# Set up the substructure search
ss = oechem.OESubSearch(mapped_smiles)
oechem.OEPrepareSearch(molecule, ss)
ss.SetMaxMatches(1)

atom_map = {}
for match in matches:
    for ma in match.GetAtoms():
        atom_map[ma.pattern.GetMapIdx()] = ma.target.GetIdx()
        
```

The `atom_map` dictionary can then be used to order the atoms in the xyz
coordinates. If the same mapped SMILES is used, the order of will always
be the same for that molecule.

#### How to use cmiles
`cmiles` has one main function, `to_canonical_smiles(molecule)`. This 
function returns a dictionary with 5 SMILES generated with either OpenEye
or RDKit. The 5 strings are:

1. Canonical SMILES
2. Canonical, isomeric SMILES
3. Canonical, isomeric, explicit hydrogen SMILES
4. Canonical, explicit hydrogen SMILES
5. Canonical, isomeric, explicit hydrogen, mapped SMILES.

If you only want one SMILES, you can use `to_canonical_smiles_oe` or 
`to_canonical_smiles_rd` with the appropriate options to get the SMILES
with either OpenEye or RDKit canonicalization.

### Copyright

Copyright (c) 2018, Chaya D. Stern


#### Acknowledgements
 
Project based on the 
[Computational Chemistry Python Cookiecutter](https://github.com/choderalab/cookiecutter-python-comp-chem)
