import cmiles
import rdkit
version = rdkit.__version__
print(version)
# Regenerate with molecule names (makes it easier to search later)

# Make sure to add SMILES to input smi file. RDKit expects that line there and just skips the first line
mols = rdkit.Chem.SmilesMolSupplier('drug_bank_stereo.smi', titleLine=False)
mapped_str = ''
inchi_srt = ''
inchi_key = ''
for m in mols:
    name = m.GetProp('_Name')
    mapped_sm = cmiles.utils.mol_to_smiles(m)
    inchi = cmiles.get_inchi_and_key(m)
    mapped_str += mapped_sm
    mapped_str += ' '
    mapped_str += name
    mapped_str +='\n'
    inchi_srt += inchi[0]
    inchi_srt += ' '
    inchi_srt += name
    inchi_srt += '\n'
    inchi_key += inchi[-1]
    inchi_key += ' '
    inchi_key += name
    inchi_key += '\n'

with open('drug_bank_mapped_smi_rd_{}.smi'.format(version), 'w') as f:
    f.write(mapped_str)

with open('drug_bank_inchi_rd_{}.txt'.format(version), 'w') as f:
    f.write(inchi_srt)

with open('drug_bank_inchikey_rd_{}.txt'.format(version), 'w') as f:
    f.write(inchi_key)