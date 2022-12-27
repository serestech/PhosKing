#!/usr/bin/env python3
import torch
import esm
import pickle
import os


data_dir = os.getcwd() + '/data'

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 1280 parameters
layer = 33
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 320 parameters
# layer = 6
batch_converter = alphabet.get_batch_converter()
model.eval()

# CHANGE THIS IF CUDA IS NOT AVAILABLE
# By default, embeddings are computed using cuda, but stored in cpu for better compatibility
device = torch.device('cuda')
device_cpu = torch.device('cpu')
model = model.cuda(device)

# # Get only proteins with kinase (old)
# phospho_set = set()
# with open(data_dir + '/train/phospho_proteins_with_kinase.txt') as phospho_file:
#     for line in phospho_file:
#         phospho_set.add(line.strip())


# Read fasta file
fasta_file_name = data_dir + '/train/comb_seq_0.85.fasta'
data = list()
with open(fasta_file_name) as infile:
    seq = ''
    for line in infile:
        if line[0] == '>':
            if seq != '':
                data.append((seqID, seq))
            seq = ''
            seqID = line[1:].strip()
        else:
            seq += line.strip()
    if seq != '':
        data.append((seqID, seq))

print('Data length:', len(data))
n_pick = 1
n_skipped = 0
representations = dict()

# Extract per-residue representations (on CPU)
with torch.no_grad():
    for i, tup in enumerate(data):

        #Write dict to pickle every 1000 sequences (minus too long sequences)
        if i % 1000 == 0 and representations:
            with open(f'{data_dir}/embedding_pickles/representations_{n_pick}.pickle', 'wb') as f:
                pickle.dump(representations, f)
            representations = dict()
            n_pick += 1

        batch_labels, batch_strs, batch_tokens = batch_converter([tup])
        batch_tokens = batch_tokens.to(device)

        try:
            x = model(batch_tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
            x = x.to(device_cpu)
            representations[(tup[0])] = x
            print('Processed sequences:', i, '; Written files:', n_pick-1, '; Skipped sequences:', n_skipped, end='\r')
        except:
            n_skipped += 1 #Skip too long sequences

    if representations:
        with open(f'{data_dir}/embedding_pickles/representations_{n_pick}.pickle', 'wb') as f:
            pickle.dump(representations, f)
            n_pick += 1
        
print('Processed sequences:', i, '; Written files:', n_pick-1, '; Skipped sequences:', n_skipped)
print('The end :D')