import sys, os
sys.path.append(os.getcwd())
from dataset_utils.fasta_utils import read_fasta

class Table:
    def __init__(self) -> None:
        self.header: list[str] = []
        self.rows: list[list[str]] = []
    
    def get(self, row_i: int, col: str):
        col_i = self.header.index(col)
        return self.rows[row_i][col_i]

def simplify_kinase(kinase):
    '''
    Group kinase into kinase family
    '''
    simplified_kinase = kinase

    if kinase.startswith('PKA'):
        simplified_kinase = 'PKA'

    if kinase.startswith('PKB'):
        simplified_kinase = 'PKB'

    if kinase.startswith('PKC'):
        simplified_kinase = 'PKC'

    return simplified_kinase


with open('data/phosphosite_dump/Kinase_Substrate_Dataset.tsv', 'r') as tsv:
    table = Table()
    for i, line in enumerate(tsv.readlines()):
        if i < 3:
            continue

        line_split = line[:-1].split('\t')

        if i == 3:
            table.header = line_split
            continue

        table.rows.append(line_split)

phosphosite_fasta = read_fasta('data/phosphosite_dump/Phosphosite_PTM_seq.fasta', format=dict)
phosphosite_fasta = {name.split('|')[-1].strip() : seq for name, seq in phosphosite_fasta.items() if name.split('|')[-1].strip() != ''}

filtered_fasta = ''
metadata_table = 'seq_id\tkinase\torganism\taa_phosphosite\tposition\n'

saved_sequences = set()
for row in table.rows:
    _, kinase, _, _, _, _, seq_id, _, organism, aa, _, _, _, _, _, _  = row

    if seq_id not in phosphosite_fasta.keys():
        print(f'Skipping sequence {seq_id} (not in FASTA)')
        continue
    
    if seq_id not in saved_sequences:
        sequence = phosphosite_fasta[seq_id]
        filtered_fasta += f'> {seq_id}\n{sequence.upper()}\n'
        saved_sequences.add(seq_id)

    metadata_table += f'{seq_id}\t{kinase}\t{organism}\t{aa}\t{aa[1:]}\n'
    
with open('data/dataset_phosphosite/phosphosite_metadata.tsv', 'w') as metadata:
    metadata.write(metadata_table)

with open('data/dataset_phosphosite/phosphosite_sequences.fasta', 'w') as fasta:
    fasta.write(filtered_fasta)
