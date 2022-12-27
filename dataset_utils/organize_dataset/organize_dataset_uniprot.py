#!/usr/bin/env python3
import json
import os

DATA_FOLDER = 'data/uniprot_dump_19_11_2022_16_05'

json_files = os.listdir(DATA_FOLDER)
json_files.sort(key=lambda x: int(x.split('_')[0]))

def generate_fasta(i, protein_dict):
    sequence = protein_dict['sequence']['value']
    header = f'> {i}\n'
    fasta = f'{header}{sequence}\n'
    return fasta

metadata_table = []
features_table = []

output_fasta = open('data/dataset_uniprot/sequences.fasta', 'w')
fasta_pos = 0
n_files = len(json_files)
features = {}
for i, filename in enumerate(json_files):
    with open(f'{DATA_FOLDER}/{filename}') as json_file:
        uniprot_dict = json.load(json_file)

    fasta_write_buffer = ''

    for protein_dict in uniprot_dict['results']:
        uniprot_accession = protein_dict['primaryAccession']
        uniprot_ID = protein_dict['uniProtkbId']
        seq_length = protein_dict['sequence']['length']
        org_name = protein_dict['organism']['scientificName']
        org_taxonID = protein_dict['organism']['taxonId']
        org_domain = protein_dict['organism']['lineage'][0]
        
        fasta_seq = generate_fasta(uniprot_accession, protein_dict)
        fasta_len = len(fasta_seq)
        fasta_write_buffer += fasta_seq

        table_line = f'{uniprot_accession}\t{uniprot_ID}\t{org_name}\t{org_domain}\t{org_taxonID}\t{seq_length}'
        metadata_table.append(table_line)

        for feature_dict in protein_dict['features']:
            feature_description = feature_dict['description']
            feature_type = feature_dict['type']
            feature_location = feature_dict['location']['start']['value']
            features_table.append(f'{uniprot_accession}\t{feature_type}\t{feature_description}\t{feature_location}')

    output_fasta.write(fasta_write_buffer)

    if i % 10 == 0:
        print(f'{i + 1}/{n_files}', end='\r')

print()

output_fasta.close()

print ('Writing metadata table...')
with open('data/dataset_uniprot/metadata.tsv', 'w') as metadata_file:
    header = 'uniprot_accession\tuniprot_ID\torg_name\torg_domain\torg_taxonID\tseq_length\n'
    metadata_file.write(header)
    metadata_file.write('\n'.join(metadata_table))
    print(f'Written {len(metadata_table)} rows of metadata with fields {header[:-1]}')

print ('Writing features table...')
with open('data/dataset_uniprot/features.tsv', 'w') as features_file:
    header = 'uniprot_accession\tfeature_type\tfeature_description\tfeature_location\n'
    features_file.write(header)
    features_file.write('\n'.join(features_table))
    print(f'Written {len(features_table)} rows of features with fields {header[:-1]}')
