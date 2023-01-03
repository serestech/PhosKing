import torch
import torch.nn.functional as F
from models import CNN_RNN


def predict(seq, representations):
    # Set model parameters
    aa_window = 16
    n_inputs = 1280
    n_hidden_1 = 512
    n_hidden_2 = 1024

    # Load model
    model = CNN_RNN.CNN_RNN_FFNN(n_inputs, n_hidden_1, n_hidden_2)
    model.load_state_dict(torch.load("states_dicts/CNN_RNN.pth"))

    # Find phosphorylatable potential aminoacid 
    phosphorilable_aas_idxs = phosphorilable_aas(seq,  interesting_aas={'S', 'T', 'Y'})

    bound = len(seq) + 1
    x = representations 
    # Obtain aminoacid window for each phosphorylatable potential aminoacid
    for pos in phosphorilable_aas_idxs:
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
        
        aa_tensor = aa_tensor[None, :]
        
        # Get model prediction
        preds = model(aa_tensor)
        preds = preds.detach().cpu().numpy().flatten()
        if preds>0.75:
            print("Aa: ", pos, "Probability: ",preds)
        
def phosphorilable_aas(seq, interesting_aas={'S', 'T', 'Y'}):
    '''
    Take a sequence and return the index (1-indexed) of the phosphorilable amino acids
    '''
    phosphorilable_positions = list()

    for i, aa in enumerate(seq):
        if aa in interesting_aas:
            phosphorilable_positions.append(i + 1)  # 1-indexed positions

    return phosphorilable_positions

