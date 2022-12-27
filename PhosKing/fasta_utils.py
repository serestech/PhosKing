from typing import Union

def read_fasta(file: str, format: type = list) -> Union[list[tuple], dict]:
    '''
    Reads a fasta file into a list of tuples ready for ESM or, optionally, a dict.
    '''
    with open(file, 'r') as fastafile:
        raw_fasta = fastafile.read()

    fasta_lines = raw_fasta.splitlines()

    # Trim comments and empty liens at the beginning
    for i, line in enumerate(fasta_lines):
        if line.startswith('>'):
            first_entry_line = i
            break

    fasta_lines = fasta_lines[first_entry_line:]

    assert fasta_lines[0].startswith('>'), "Fasta file after trimming doesn't start with '>'"

    fasta_list = []
    sequence = ''
    for i, line in enumerate(fasta_lines):
        next_line = fasta_lines[i + 1] if i + 1 < len(fasta_lines) else None

        if line.startswith('>'):
            current_header = line[1:].strip()
        else:
            sequence += line

        if next_line is None or next_line.startswith('>'):
            sequence = sequence.replace('\n', '')
            fasta_list.append((current_header, sequence))
            current_header = ''
            sequence = ''
    
    if format == list:
        return fasta_list
    elif format == dict:
        return {name : sequence for name, sequence in fasta_list}
