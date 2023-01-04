import torch
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from random import sample
import time as t 
from random import shuffle
import torch.nn.functional as F
import esm
import sys
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


def phosphorilable_aas(seq, interesting_aas={'S', 'T', 'Y'}):
    '''
    Take a sequence and return the index (1-indexed) of the phosphorilable amino acids
    '''
    phosphorilable_positions = list()

    for i, aa in enumerate(seq):
        if aa in interesting_aas:
            phosphorilable_positions.append(i + 1)  # 1-indexed positions

    return phosphorilable_positions


class ESM_Embeddings(Dataset):
    '''
    PyTorch Dataset for phosphorilations. 
    '''
    def __init__(self, fracc_non_phospho=0.5, aa_window=0, small_data=False, flatten_window:bool=False, add_dim=False, mode='phospho', mappings_dir:str=None, verbose_init=False):
        self.start = t.perf_counter()
        self.verbose = verbose_init
        self._log('Initializing...')
        
        if mode not in ('phospho', 'kinase'):
            raise NotImplementedError(f'Mode {mode} not recognized')
        else:
            self.mode = mode
                    
        assert 0 < fracc_non_phospho < 1, 'Fraction of non phosphorilated amino acids must be between 0 and 1'
        self.fracc_non_phospho = fracc_non_phospho
        self.fracc_phospho = 1 - self.fracc_non_phospho 
        
        self.aa_window = aa_window
        self.flatten_window = flatten_window
        self.add_dim = add_dim

        data_dir = os.path.dirname(__file__) + '/../data'
        self.pickles_dir = data_dir + '/embedding_pickles'
        self.metadata_table = data_dir + '/train/features_kinase.tsv'
        
        self.embeddings_dict: dict[torch.Tensor] = {}
        self._load_pickles(small_data)
        
        self.sequence_names = list(self.embeddings_dict.keys())
        self.fasta = read_fasta(data_dir + '/train/comb_seq_0.85.fasta', format=dict)

        # Keep only pickled sequences
        before = len(self.fasta)
        self.fasta = {seqname : seq for seqname, seq in self.fasta.items()
                      if seqname in self.sequence_names}
        self._log(f'Discarded {before - len(self.fasta)} sequences that were not in the pickles')
        
        if self.mode == 'kinase':
            self.mapping = {'AMPK': 0, 'ATM': 1, 'Abl': 2, 'Akt1': 3, 'AurB': 4, 'CAMK2': 5, 'CDK1': 6, 'CDK2': 7, 'CDK5': 8, 'CKI': 9, 'CKII': 10, 'DNAPK': 11, 'EGFR': 12, 'ERK1': 13, 'ERK2': 14, 'Fyn': 15, 'GSK3': 16, 'INSR': 17, 'JNK1': 18, 'MAPK': 19, 'P38MAPK': 20, 'PKA': 21, 'PKB': 22, 'PKC': 23, 'PKG': 24, 'PLK1': 25, 'RSK': 26, 'SRC': 27, 'mTOR': 28}
            self.reverse_mapping = {i: kinase for kinase, i in self.mapping.items()}
        
        self.data = [] # List of tuples, each containing sequence, position and phosphorilation (True/False)
        self._load_metadata()
        
        self.true  = torch.Tensor([1])
        self.false = torch.Tensor([0])
        
        self._log('Finished initialization')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, pos, out_data = self.data[idx]
        
        embedding = self.embeddings_dict[seq]
        
        bound = embedding.size()[1] - 1
        
        if (pos + self.aa_window) <= bound and (pos - self.aa_window) > 0:  # Normal case
            out_tensor = embedding[0, pos - self.aa_window : pos + self.aa_window + 1]
        elif (pos +  self.aa_window) > bound:  # Overflow over the sequence. Stack sequence end
            if (pos - self.aa_window) <= 0:  # position too low
                out_tensor = embedding[0, 1: bound + 1]
                extra_tensors_1 = self.aa_window - pos + 1
                extra_tensors_2 = pos - (bound - self.aa_window)
                out_tensor = F.pad(out_tensor, pad=(0, 0 , extra_tensors_1, extra_tensors_2), value=0)
            else:
                out_tensor = embedding[0, pos - self.aa_window : bound + 1]
                extra_tensors = pos - (bound - self.aa_window)
                out_tensor = F.pad(out_tensor, pad=(0, 0 , 0, extra_tensors), value=0)
        elif (pos - self.aa_window) <= 0:  # position too low
            out_tensor = embedding[0, 1 : pos + self.aa_window + 1]
            extra_tensors = self.aa_window - pos + 1
            out_tensor = F.pad(out_tensor, pad=(0, 0 , extra_tensors, 0), value=0)
        else:
            raise RuntimeError('Error cogiendo los tensores. Habla con Dani')  
        
        if self.flatten_window or self.aa_window == 0:
            out_tensor = torch.flatten(out_tensor)
        
        if self.add_dim:
            out_tensor = out_tensor[None, : ]
        
        if self.mode == 'phospho':
            out_label = self.true if out_data else self.false
        elif self.mode == 'kinase':
            out_label = out_data
        else:
            raise NotImplementedError
        
        return out_tensor, out_label
    
    def _load_pickles(self, small_dataset: bool):
        self._log('Loading pickles...')
        pickle_files = [file for file in os.listdir(self.pickles_dir) if file.endswith('.pickle')]
        if small_dataset:
            pickle_files = pickle_files[:1]
        for filename in pickle_files:
            self._log(f'Loading pickle {filename}')
            with open(f'{self.pickles_dir}/{filename}', 'rb') as pickle_file:
                self.embeddings_dict = {**self.embeddings_dict, **pickle.load(pickle_file)}
        self._log(f'Pickles contain {len(self.embeddings_dict)} sequences')
    
    def _load_metadata(self):
        '''
        Load the Phosphosite metadata, filtering by sequences present in the pickles. 
        '''
        self._log('Loading metadata')
        
        with open(self.metadata_table, 'r') as meta_table:
            table_lines = meta_table.readlines()
        PHOSPHORILABLE_AAS = {'S', 'T', 'Y'}
            
        data_dict = {}  # Metadata will be saved here and ocnverted to list at the very end
        n_discarded_missing = 0
        n_discarded_not_phospho = 0
        n_discarded_wrong = 0
        n_unknown_kinase = 0
        for table_line in table_lines[1:]:
            seq_id, aa, position, kinases = table_line.strip().split('\t')
            aminoacid: tuple = (seq_id, int(position))  # aa identified by seq and position (1-indexed)
            
            if seq_id not in self.sequence_names:
                n_discarded_missing += 1
                # self._log(f'Discarding phosphorilation {aminoacid} (seq not in pickles)')
                continue
            
            try:
                if self.fasta[seq_id][int(position) - 1] != aa:
                    n_discarded_wrong += 1
                    # self._log(f'Discarding phosphorilation {aminoacid} (amino acid does not correspond with FASTA)')
                    continue
            except IndexError:
                    n_discarded_wrong += 1
                    # self._log(f'Discarding phosphorilation {aminoacid} (amino acid does not correspond with FASTA, out of range)')
                    continue

            if aa not in PHOSPHORILABLE_AAS:
                n_discarded_not_phospho += 1
                # self._log(f'Discarding phosphorilation {aminoacid} (amino acid not phosphorilable)')
                continue
            
            if self.mode == 'phospho':
                data_dict[aminoacid] = True
            elif self.mode == 'kinase':
                if 'Unknown' in kinases:
                    n_unknown_kinase += 1
                    continue
                kinases = kinases.strip().split(',')
                label = torch.zeros(len(self.mapping))
                for kinase in kinases:
                    i = self.mapping[kinase]
                    label[i] = 1
                data_dict[aminoacid] = label
        
        n_phosphorilations = len(data_dict)
        self._log(f'Loaded {n_phosphorilations} phosphorilations. {n_discarded_missing + n_discarded_not_phospho + n_discarded_wrong + n_unknown_kinase} dsicarded ({n_discarded_missing} not in pickles, {n_discarded_not_phospho} not in {PHOSPHORILABLE_AAS}, {n_discarded_wrong} wrongly documented{"" if self.mode == "phospho" else (", " + str(n_unknown_kinase) + " unknown kinase")})')
        
        if self.mode == 'phospho':
            self._log(f'Getting all non-phosphorilated phosphorilable amino acids')
            
            not_phospho_aas = []
            for seq_id, sequence in self.fasta.items():
                for position in phosphorilable_aas(sequence, PHOSPHORILABLE_AAS):
                    aminoacid: tuple = (seq_id, position)
                    if aminoacid not in data_dict.keys():
                        not_phospho_aas.append(aminoacid)
            
            self._log(f'Found {len(not_phospho_aas)} non-phosphorilated phosphorilable amino acids. Subsetting')

            population_size = int((self.fracc_non_phospho * n_phosphorilations) / self.fracc_phospho)
            self._log(f'Sampling {population_size} non-phosphorilated amino acids') 
            not_phospho_sample = sample(population=not_phospho_aas, k=population_size)
            dataset_size = n_phosphorilations + len(not_phospho_sample)
            self._log(f'Sampled {len(not_phospho_sample)} amino acids. Total dataset: {(n_phosphorilations / dataset_size) * 100:.2f}% phosphorilated amino acids, {(len(not_phospho_sample) / dataset_size) * 100:.2f}% non-phosphorilated')
            
            for aminoacid in not_phospho_sample:
                data_dict[aminoacid] = False
            
        self.data = [(*aminoacid, label) for aminoacid, label in data_dict.items()]
        
        self._log(f'Checking that all extracted positions actually correspond to a phosphorilable amino acid ({PHOSPHORILABLE_AAS}) in the FASTA ')
        phos_delete = []
        for i, (seq, position, _) in enumerate(self.data):
            try:
                if self.fasta[seq][position - 1] not in PHOSPHORILABLE_AAS:
                    phos_delete.append(i)
                    self._log(f'WARNING: Found amino acid \'{self.fasta[seq][position - 1]}\' in data (seq {seq}, position {position})')
            except IndexError as exc:
                    phos_delete.append(i)
                    self._log(f'WARNING: Found position {position} in sequence of {len(self.fasta[seq])}: {exc}')
                
        self.data = [phos for i, phos in enumerate(self.data) if i not in phos_delete]
        
        shuffle(self.data)
        
        self._log(f'Generated data list of length {len(self.data)}. Some examples:\n{sample(self.data, 50 if self.mode == "phospho" else 5)}')
        
    def _log(self, msg: str):
        if self.verbose:
            now = t.perf_counter()
            print(f'Dataset [{int((now - self.start) // 60):02d}:{(now - self.start) % 60:0>6.3f}]:', msg)


class ESM_Embeddings_test:
    '''
    Dataset for finding phosphorylations using a previously trained model
    '''
    def __init__(self, fasta_file, params, device, aa_window=0, two_dims=False, mode='phospho'):
        print('Reading fasta...')
        try:
            self.seq_data = read_fasta(fasta_file, format=list)
        except FileNotFoundError:
            print('Fasta file not found, aborting...')
            sys.exit(1)
        
        # # In case we take the whole big fasta and only want a small sample for testing
        #import random
        #self.seq_data = random.sample(self.seq_data, k=50)

        print(f'Found {len(self.seq_data)} sequences!')

        print('Computing embeddings...')
        self._compute_embeddings(params, aa_window, two_dims, device)        


    def _compute_embeddings(self, params, aa_window, two_dims, device):
        if params == 1280:
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            n_layers = 33
        elif params == 320:
            esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            n_layers = 6
        else:
            print(f'Embedding with {params} parameters not available, aborting...')
            sys.exit(1)

        batch_converter = alphabet.get_batch_converter()
        esm_model.to(device)
        esm_model.eval()

        self.idxs = dict()
        self.tensors = dict()
        with torch.no_grad():
            k = 0
            for seq_ID, seq in self.seq_data:
                idxs = phosphorilable_aas(seq)
                if len(idxs) == 0:
                    print(f'Sequence {seq_ID} has no phosphorilable aminoacids, omitting...')
                    continue

                _, _, batch_tokens = batch_converter([(seq_ID, seq)])
                batch_tokens = batch_tokens.to(device)
                try:
                    representations = esm_model(batch_tokens, repr_layers=[n_layers], return_contacts=False)["representations"][n_layers]
                except:
                    print(f'It seems sequence {seq_ID} is too long and esm could not embeddings! (out of memory)')
                    continue

                self.idxs[seq_ID] = idxs
                if two_dims and aa_window == 0:
                    x = representations[0]
                    out_tensor = torch.index_select(x, 0, torch.tensor(self.idxs[seq_ID], dtype=torch.int32, device=device))

                else:
                    bound = len(seq) + 1
                    out_tensor = torch.empty((len(self.idxs[seq_ID]), aa_window*2+1, params))
                    x = representations

                    for i, pos in enumerate(self.idxs[seq_ID]):
                        if (pos + aa_window) <= bound and (pos - aa_window) > 0:  # Normal case
                            aa_tensor = x[0, pos - aa_window : pos + aa_window + 1]
                        elif (pos +  aa_window) > bound:  # Overflow over the sequence. Stack sequence end
                            if (pos - aa_window) <= 0:  # both position too low and overflow
                                extra_tensors_1 = aa_window - pos + 1
                                extra_tensors_2 = pos - (bound - aa_window)
                                aa_tensor = F.pad(x[0, 1: bound + 1], pad=(0, 0 , extra_tensors_1, extra_tensors_2), value=0)
                            else:
                                extra_tensors = pos - (bound - aa_window)
                                aa_tensor = F.pad(x[0, pos - aa_window : bound + 1], pad=(0, 0 , 0, extra_tensors), value=0)
                        elif (pos - aa_window) <= 0:  # position too low
                            extra_tensors = aa_window - pos + 1
                            aa_tensor = F.pad(x[0, 1 : pos + aa_window + 1], pad=(0, 0 , extra_tensors, 0), value=0)
                        else:
                            raise RuntimeError('Error cogiendo los tensores. Habla con Dani')  
                        
                        if not two_dims or aa_window > 0:
                            aa_tensor = aa_tensor[None, :]

                        out_tensor[i] = aa_tensor

                self.tensors[seq_ID] = out_tensor
                k += 1
                print(f'{k} embeddings computed!', end='\r')
            
            
    def __getitem__(self, seq_ID):
        return self.idxs[seq_ID], self.tensors[seq_ID]
    
    def IDs(self):
        return list(self.idxs.keys())



if __name__ == '__main__':
    ds = ESM_Embeddings(fracc_non_phospho=0.6,
                        mode='phospho',
                        aa_window=2,
                        verbose_init=True)
    
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    
    inputs, labels = next(iter(loader))
    
    print(f'Getting first batch produced tensors of sizes {inputs.size()} (inputs) and {labels.size()} (labels)')

    ds = ESM_Embeddings(fracc_non_phospho=0.6,
                        mode='kinase',
                        aa_window=2,
                        verbose_init=True)
    
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    
    inputs, labels = next(iter(loader))
    
    print(f'Getting first batch produced tensors of sizes {inputs.size()} (inputs) and {labels.size()} (labels)')
