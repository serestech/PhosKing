from typing import List
from models.CNN_RNN import CNN_RNN_FFNN
from models.FFNN_kin import FFNN
import torch
import numpy as np
import torch.nn.functional as F
print(f'Using PyTorch version {torch.__version__}')
import esm
try:
    from phosking.dataset import phosphorilable_aas
except ModuleNotFoundError:
    from dataset import phosphorilable_aas
from os.path import isfile

AA_WINDOW = 16
KINASE_MAPPING = {'AMPK': 0, 'ATM': 1, 'Abl': 2, 'Akt1': 3, 'AurB': 4, 'CAMK2': 5, 'CDK1': 6, 'CDK2': 7, 'CDK5': 8, 'CKI': 9, 'CKII': 10, 'DNAPK': 11, 'EGFR': 12, 'ERK1': 13, 'ERK2': 14, 'Fyn': 15, 'GSK3': 16, 'INSR': 17, 'JNK1': 18, 'MAPK': 19, 'P38MAPK': 20, 'PKA': 21, 'PKB': 22, 'PKC': 23, 'PKG': 24, 'PLK1': 25, 'RSK': 26, 'SRC': 27, 'mTOR': 28}
KINASE_MAPPING_REV = {i: kinase for kinase, i in KINASE_MAPPING.items()}

def predict(sequences: List[tuple], force_cpu=False) -> tuple:
    device = torch.device('cuda' if not force_cpu and torch.cuda.is_available() else 'cpu')
    print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')
    
    embeddings = compute_embeddings(sequences, device)
    
    print('Loading PhosKing model (phosphorylation)')
    model = CNN_RNN_FFNN(1280, 512, 1024)
    model = model.to(device)
    state_dict = torch.load('states_dicts/CNN_RNN.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print('Computing phosphorylation predictions')
    phospho_predictions = {}
    for seq_ID, seq in sequences:
        if seq_ID not in embeddings.keys():
            print(f'Skipping sequence {seq_ID}. No embeddings found')
            continue
        
        aa_positions = phosphorilable_aas(seq)
        
        with torch.no_grad():
            inputs: torch.Tensor = embeddings[seq_ID]
            inputs = inputs.to(device)
            outputs: torch.Tensor = model(inputs)
            outputs: np.ndarray = outputs.detach().cpu().numpy().flatten()
        

        phospho_predictions[seq_ID] = {}
        for i, aa_pos in enumerate(aa_positions):
            phospho_predictions[seq_ID][aa_pos] = outputs[i]
    
    if not isfile('states_dicts/kinase_model.pth'):
        print('Kinase model not found. Finished')
        return phospho_predictions, None
    
    print('Loading PhosKing model (kinase)')
    model_kin = FFNN(6400, 12800, 29)
    model_kin = model_kin.to(device)
    state_dict = torch.load('states_dicts/kinase_model.pth', map_location=device)
    model_kin.load_state_dict(state_dict)
    model_kin.eval()
    
    print('Computing kinase predictions')
    kinase_predictions = {}
    for seq_ID, seq in sequences:
        if seq_ID not in embeddings.keys():
            print(f'Skipping sequence {seq_ID}. No embeddings found')
            continue
        
        aa_positions = phosphorilable_aas(seq)
        
        with torch.no_grad():
            inputs: torch.Tensor = embeddings[seq_ID]
            inputs = inputs[:,14:19,:]
            inputs = torch.flatten(inputs, start_dim=1)
            inputs = inputs.to(device)
            outputs: torch.Tensor = model_kin(inputs)
            outputs: np.ndarray = outputs.detach().cpu().numpy()
        
        kinase_predictions[seq_ID] = {}
        for i, aa_pos in enumerate(aa_positions):
            kinase_predictions[seq_ID][aa_pos] = {KINASE_MAPPING_REV[j]: score for j, score in enumerate(outputs[i])}
    
    print('Finished')
        
    return phospho_predictions, kinase_predictions

def compute_embeddings(sequences, device) -> dict:
    print('Loading ESM-2...')
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.to(device)
    esm_model.eval()
    print('Loaded ESM-2. Computing embeddings...')
    
    idxs = dict()
    tensors = dict()
    with torch.no_grad():
        k = 0
        for seq_ID, seq in sequences:
            phospho_targets = phosphorilable_aas(seq)
            if len(phospho_targets) == 0:
                print(f'Sequence {seq_ID} has no phosphorilable aminoacids, omitting...')
                continue

            _, _, batch_tokens = batch_converter([(seq_ID, seq)])
            batch_tokens = batch_tokens.to(device)
            try:
                representations = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)["representations"][33]
            except Exception as exc:
                print(f'Error calculating the embeddings for {seq_ID}: {exc}')
                print(f'Skipping sequence {seq_ID}')
                continue

            idxs[seq_ID] = phospho_targets

            bound = len(seq) + 1
            out_tensor = torch.empty((len(idxs[seq_ID]), AA_WINDOW * 2 + 1, 1280))
            x = representations

            for i, pos in enumerate(idxs[seq_ID]):
                if (pos + AA_WINDOW) <= bound and (pos - AA_WINDOW) > 0:  # Normal case
                    aa_tensor = x[0, pos - AA_WINDOW : pos + AA_WINDOW + 1]
                elif (pos +  AA_WINDOW) > bound:  # Overflow over the sequence. Stack sequence end
                    if (pos - AA_WINDOW) <= 0:  # both position too low and overflow
                        extra_tensors_1 = AA_WINDOW - pos + 1
                        extra_tensors_2 = pos - (bound - AA_WINDOW)
                        aa_tensor = F.pad(x[0, 1: bound + 1], pad=(0, 0 , extra_tensors_1, extra_tensors_2), value=0)
                    else:
                        extra_tensors = pos - (bound - AA_WINDOW)
                        aa_tensor = F.pad(x[0, pos - AA_WINDOW : bound + 1], pad=(0, 0 , 0, extra_tensors), value=0)
                elif (pos - AA_WINDOW) <= 0:  # position too low
                    extra_tensors = AA_WINDOW - pos + 1
                    aa_tensor = F.pad(x[0, 1 : pos + AA_WINDOW + 1], pad=(0, 0 , extra_tensors, 0), value=0)
                else:
                    raise IndexError(f'Error loading the ESM-2 amino acid window. Unable to handle index {pos} in sequence {seq_ID}')  
                
                aa_tensor = aa_tensor[None, :]

                out_tensor[i] = aa_tensor

            tensors[seq_ID] = out_tensor
            k += 1
            print(f'{k} of {len(sequences)} embeddings computed', end='\r')
        print('Finished computing embeddings')
        
    return tensors

def format_predictions(sequences: List[tuple], predictions: tuple) -> None:
    sequences: dict = {seq_ID: seq for seq_ID, seq in sequences}
    
    phospho_predictions, kinase_predictions = predictions
    
    for seq_ID, preds in phospho_predictions.items():
        seq = sequences[seq_ID]
        dots = ''
        for pos in range(len(seq)):
            if pos + 1 in preds.keys():
                if preds[pos + 1] > 0.99:
                    dots += '*'
                elif preds[pos + 1] > 0.9:
                    dots += '+'
                elif preds[pos + 1] > 0.75:
                    dots += '.'
                else:
                    dots += ' '
            else:
                dots += ' '
        
        print('- ' * 41 + '\n > ' + seq_ID)
        for i in range(len(seq)//80+1):
            l = i*80
            print(dots[l:l+80])
            print(seq[l:l+80])
            print(' ' * 9 + '|' + '|'.join(list('{:<9}'.format(l + j * 10) for j in range(1, 9) if (l + j * 10) <= len(seq))))
        
        print()
        print('Pos.   Score       '*5)
        for i, pos in enumerate(preds.keys()):
            if i % 5 == 0:
                if i != 0:
                    print()
            else:
                print('|', end='  ')
            if preds[pos] > 0.99:
                dot = '*'
            elif preds[pos] > 0.9:
                dot = '+'
            elif preds[pos] > 0.75:
                dot = '.'
            else:
                dot = ' '
            print('{}{:<6}{:<5.3g} {}'.format(sequences[seq_ID][pos - 1], pos, round(preds[pos], 3), dot), end='  ')
        print('\n')
        
        if kinase_predictions is None:
            continue
        
        if seq_ID not in kinase_predictions.keys():
            print(f'Found no kinase predictions for sequence {seq_ID}')
            continue
        
        # Show kinase prediction only if phosphorylation score is > 0.75
        for pos, pred in preds.items():
            if pred < 0.75:
                kinase_predictions[seq_ID].pop(pos)
        
        print('Pos.   Likely kinases                               '*2)
        for i, (pos, kin_preds) in enumerate(kinase_predictions[seq_ID].items()):
            if i % 2 == 0:
                if i != 0:
                    print()
            else:
                print('|', end='  ')
        
            likely_kinases = [(kin, score) for kin, score in kin_preds.items() if score > 0.1]
            if len(likely_kinases) == 0:
                kinase_string = 'Unknown'
            else:
                likely_kinases.sort(key=lambda tup: tup[1], reverse=True)
                likely_kinases = likely_kinases[:3]
                kinase_string = ', '.join(f'{kinase} ({score:.3f})' for kinase, score in likely_kinases)
            
            if len(kinase_string) <= 40:
                kinase_string += ' ' * (40 - len(kinase_string))
            else:
                kinase_string = kinase_string[:40]
            
            print('{}{:<6}{}'.format(sequences[seq_ID][pos - 1], pos, kinase_string), end='  ')
        
        print('\n')