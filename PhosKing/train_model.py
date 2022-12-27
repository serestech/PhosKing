#! /usr/bin/env python3

import argparse
import sys, os

parser = argparse.ArgumentParser(prog='ESMTrain',
                                 description='Train a phosphorilation prediction model with PyTorch using ESM Embeddings')

parser.add_argument('-m', '--model_file', action='store', dest='model_file', help='Model file (python file with PyTorch model)')
parser.add_argument('-n', '--model_name', action='store', dest='model_name', help='Model name (class name in the model file)')
parser.add_argument('-a', '--model_args', action='store', dest='model_args', help='Comma separated ints to pass to the model constructor (e.g. 1280,2560,1)', default='')
parser.add_argument('-e', '--num_epochs', action='store', dest='n_epochs', help='Number of epochs', default='20')
parser.add_argument('-md', '--mode', action='store', dest='mode', help='Prediction mode (phospho or kinase)', default='phospho')
parser.add_argument('-zg', '--zero_grad', action='store_true', dest='zero_grad', help='Wether to reset the gradients or not on each batch')
parser.add_argument('-l', '--loss_fn', action='store', dest='loss_fn', help='Loss function', default='BCE')
parser.add_argument('-f', '--frac_non_phos', action='store', dest='frac_non_phos', help='Fraction of non phosphorilated amino acids in dataset', default='0.5')
parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', help='Batch size', default='128')
parser.add_argument('-lr', '--learning_rate', action='store', dest='lr', help='Learning rate', default='1e-5')
parser.add_argument('-w', '--weight_decay', action='store', dest='wd', help='Weight decay (only compatible optimizers)', default='1e-5')
parser.add_argument('-o', '--optimizer', action='store', dest='optimizer', help='Optimizer to use', default='Adam')
parser.add_argument('-aaw', '--aa_window', action='store', dest='aa_window', help='Amino acid window for the tensors (concatenated tensor of the 5 amino acids)', default='0')
parser.add_argument('-fw', '--flatten_window', action='store_true', dest='flatten_window', help='Wether to flatten or not the amino acid window (only if -aaw > 0)')
parser.add_argument('-ad', '--add_dim', action='store_true', dest='add_dim', help='Wether to add an extra dimension to the tensor (needed for CNN)')
parser.add_argument('-sd', '--small_data', action='store_true', dest='small_data', help='Load a small dataset (for testing)')
parser.add_argument('-es', '--early_stopping', action='store_true', dest='early_stopping', help='Do early stopping')
parser.add_argument('-c', '--force_cpu', action='store_true', dest='force_cpu', help='Force CPU training')
args = parser.parse_args()

if args.model_file is None or args.model_name is None:
    parser.print_help()
    sys.exit()

print(f'Using python env in {sys.executable}')

import torch
print(f'Using torch version {torch.__version__}')
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from importlib import import_module
from dataset import ESM_Embeddings
from math import sqrt
from sklearn.model_selection import train_test_split
import time as t

mapping = {'AMPK': 0, 'ATM': 1, 'Abl': 2, 'Akt1': 3, 'AurB': 4, 'CAMK2': 5, 'CDK1': 6, 'CDK2': 7, 'CDK5': 8, 'CKI': 9, 'CKII': 10, 'DNAPK': 11, 'EGFR': 12, 'ERK1': 13, 'ERK2': 14, 'Fyn': 15, 'GSK3': 16, 'INSR': 17, 'JNK1': 18, 'MAPK': 19, 'P38MAPK': 20, 'PKA': 21, 'PKB': 22, 'PKC': 23, 'PKG': 24, 'PLK1': 25, 'RSK': 26, 'SRC': 27, 'mTOR': 28}
reverse_mapping = {i: kinase for kinase, i in mapping.items()}

def perf_scores(predictions, targets):
    batch_size, n_classes = predictions.size()
    total_predictions = batch_size * n_classes
    
    prediction_positives = torch.sum(predictions).item()
    prediction_negatives = total_predictions - prediction_positives
    
    target_positives = torch.sum(targets).item()
    target_negatives = total_predictions - target_positives
    
    mistakes = torch.sum(torch.logical_xor(predictions, targets)).item()
    corrects = total_predictions - mistakes
    
    true_positives = torch.sum(torch.logical_and(predictions, targets)).item()
    false_positives = prediction_positives - true_positives
    true_negatives = total_predictions - torch.sum(torch.logical_or(predictions, targets)).item()
    false_negatives = prediction_negatives - true_negatives
    
    accuracy = corrects / total_predictions
    sensitivity = true_positives / target_positives
    specificity = true_negatives / target_negatives
    
    mcc = (true_positives * true_negatives - false_positives * false_negatives)/(sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives)))
    
    try:
        precision = true_positives / prediction_positives
    except ZeroDivisionError as err:
        print(f'WARNING: {err} (model predicted no kinase)')
        precision = 0
    
    return accuracy, precision, sensitivity, specificity, mcc

def save_model(model: nn.Module, train_acc, valid_acc, train_loss, valid_loss, test_acc=None, test_sens=None, test_prec=None, test_spec=None, test_mcc=None, final=False):
    dirname = os.path.dirname(__file__) + '/../states_dicts/'
    filename = dirname + states_dict_name + ('_phos' if args.mode == 'phospho' else '_kin') + ('_final' if final else '_best')
    
    print(f'    Saving checkpoint: {filename}<.pth,.info>')
    
    torch.save(model.state_dict(), filename + '.pth')
    
    print('    Saving info file...')
    
    with open(filename + '.info', 'w') as info_file:
        info_file.write('### MODEL ###\n\n')
        
        with open(args.model_file, 'r') as model_file:
            info_file.write(model_file.read() + '\n\n')
            
        info_file.write('### PARAMS ###\n\n')
        info_file.write(f'{args}\n\n')
            
        info_file.write('### PERFORMANCE ###\n\n')
        info_file.write(f'Training accuracy:   {train_acc:8.5f}  Training loss:   {train_loss:8.5f}\n')
        info_file.write(f'Validation accuracy: {valid_acc:8.5f}  Validation loss: {valid_loss:8.5f}\n')
        if test_acc is not None:
            info_file.write(f'Test accuracy:    {test_acc:8.5f}\n')
            print(f'    Test accuracy:    {test_acc:8.5f}')
        if test_sens is not None:
            info_file.write(f'Test sensitivity: {test_sens:8.5f}\n')
            print(f'    Test sensitivity: {test_sens:8.5f}')
        if test_prec is not None:
            info_file.write(f'Test precision:   {test_prec:8.5f}\n')
            print(f'    Test precision:   {test_prec:8.5f}')
        if test_spec is not None:
            info_file.write(f'Test specificity: {test_spec:8.5f}\n')
            print(f'    Test specificity: {test_spec:8.5f}')
        if test_mcc is not None:
            info_file.write(f'Test MCC:         {test_spec:8.5f}\n')
            print(f'    Test MCC:         {test_mcc:8.5f}\n')
            
        info_file.write('\n')

        info_file.write('### VALIDATION SET ###\n\n')
        for i in validation_dataset.indices:
            aminoacid = dataset.data[i]
            if args.mode == 'kinase':
                seq_id, pos, label_tensor = aminoacid
                kinases = []
                for i, val in enumerate(label_tensor):
                    if val.item() == 1:
                        kinases.append(reverse_mapping[i])
                aminoacid = (seq_id, pos, ','.join(kinases))
            info_file.write(' '.join([str(elem) for elem in aminoacid]) + '\n')

        info_file.write('\n')

        info_file.write('### TEST SET ###\n\n')
        for i in test_dataset.indices:
            aminoacid = dataset.data[i]
            info_file.write(' '.join([str(elem) for elem in aminoacid]) + '\n')

        info_file.write('\n')
        
    print('    Saved')

# Hacky thing to import the model by storing the filename and model in strings
model_dir = os.path.dirname(args.model_file)
sys.path.append(model_dir)
model_module_name = os.path.basename(args.model_file)[:-3]
model_module = import_module(model_module_name)
model_class = getattr(model_module, args.model_name)

device = torch.device('cuda' if  not args.force_cpu and torch.cuda.is_available() else 'cpu')
print(f'Using torch device of type {device.type}{": " + torch.cuda.get_device_name(device) if device.type == "cuda" else ""}')

if args.model_args is None:
    model: torch.nn.Module = model_class()
else:
    model: torch.nn.Module = model_class(*[int(arg) for arg in args.model_args.split(',')])
model = model.to(device)

dataset = ESM_Embeddings(fracc_non_phospho=float(args.frac_non_phos),
                         aa_window=int(args.aa_window),
                         flatten_window=args.flatten_window,
                         small_data=args.small_data,
                         add_dim=args.add_dim,
                         mode=args.mode,
                         verbose_init=True)

# Train on 80% of the data. Split the other 20% between test and validation
indices = list(range(len(dataset)))
train_idx, rest_idx = train_test_split(indices, train_size=0.8, test_size=0.2, shuffle=True)
test_idx, validation_idx = train_test_split(rest_idx, train_size=0.5, test_size=0.5, shuffle=True)
print(f'Number of observations:\n   Training {len(train_idx)}  Validation: {len(validation_idx)}  Test: {len(test_idx)}')

assert set(train_idx).isdisjoint(set(rest_idx)) and set(test_idx).isdisjoint(set(validation_idx))  # Sanity check

# Create subsets of the data
train_dataset = data.Subset(dataset, train_idx)
test_dataset = data.Subset(dataset, test_idx)
validation_dataset = data.Subset(dataset, validation_idx)

batch_size = int(args.batch_size)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
valid_loader = data.DataLoader(dataset=validation_dataset,
                               batch_size=batch_size,
                               shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)

if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
else:
    raise NotImplementedError(f'Optimizer {args.optimizer} not implemented')

if args.loss_fn == 'BCE':
    loss_fn = nn.BCELoss()
elif args.loss_fn == 'CEL':
    loss_fn = nn.CrossEntropyLoss()
else:
    raise NotImplementedError(f'Didn\'t recognize loss function {args.loss_fn}')

num_epochs = int(args.n_epochs)
validation_every_steps = 100 if args.mode == 'phospho' else 10

early_stop = args.early_stopping
states_dict_name = f'{args.model_name}_{t.strftime("%d_%m_%y_%H_%M", t.localtime())}'
best_val_accuracy = 0
best_val_sensitivity = 0
best_val_precision = 0
step = 0
model.train()
steps = []
train_accuracies = []
valid_accuracies = []
train_precisions = []
valid_precisions = []
train_sensitivities = []
valid_sensitivities = []
train_specificities = []
valid_specificities = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1} of {num_epochs}')
    
    train_accuracies_batches  = []
    train_precisions_batches  = []
    train_sensitivity_batches = []
    train_specificity_batches = []
    for inputs, targets in iter(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if args.zero_grad:
            optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        step += 1
        
        predictions = torch.round(outputs)
  
        train_accuracy, train_precision, train_sensitivity, train_specificity, mcc = perf_scores(predictions, targets)
        train_accuracies_batches.append(train_accuracy)
        train_precisions_batches.append(train_precision)
        train_sensitivity_batches.append(train_sensitivity)
        train_specificity_batches.append(train_specificity)

        if step % validation_every_steps == 0:
            train_loss = float(loss)
            
            train_accuracies.append(sum(train_accuracies_batches) / len(train_accuracies_batches))
            train_precisions.append(sum(train_precisions_batches) / len(train_precisions_batches))
            train_sensitivities.append(sum(train_sensitivity_batches) / len(train_sensitivity_batches))
            train_specificities.append(sum(train_specificity_batches) / len(train_specificity_batches))
            train_accuracies_batches = []
            train_precisions_batches = []
            train_sensitivity_batches = []
            train_specificity_batches = []

            
            valid_accuracies_batches = []
            valid_precisions_batches = []
            valid_sensitivity_batches = []
            valid_specificity_batches = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in iter(valid_loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(inputs)
                    
                    loss = loss_fn(outputs, targets)
                    
                    predictions = torch.round(outputs)
                    
                    valid_accuracy, valid_precision, valid_sensitivity, valid_specificity, mcc = perf_scores(predictions, targets)
                    valid_accuracies_batches.append(valid_accuracy)
                    valid_precisions_batches.append(valid_precision)
                    valid_sensitivity_batches.append(valid_sensitivity)
                    valid_specificity_batches.append(valid_specificity)
                    
                model.train()
            
            valid_accuracies.append(sum(valid_accuracies_batches) / len(valid_accuracies_batches))
            valid_precisions.append(sum(valid_precisions_batches) / len(valid_precisions_batches))
            valid_sensitivities.append(sum(valid_sensitivity_batches) / len(valid_sensitivity_batches))
            valid_specificities.append(sum(valid_specificity_batches) / len(valid_specificity_batches))
            
            steps.append(step)
            print(f'  Step {step}')
            print(f'    Training accuracy:      {train_accuracies[-1]:8.5f}   Training loss:          {train_loss:8.5f}')
            print(f'    Validation accuracy:    {valid_accuracies[-1]:8.5f}   Validation loss:        {float(loss):8.5f}')
            print(f'    Training sensitivity:   {train_sensitivities[-1]:8.5f}   Training specificity:   {train_specificities[-1]:8.5f}   Training precision:   {train_precisions[-1]:8.5f}')
            print(f'    Validation sensitivity: {valid_sensitivities[-1]:8.5f}   Validation specificity: {valid_specificities[-1]:8.5f}   Validation precision: {valid_precisions[-1]:8.5f}')
            
            if early_stop and step > (400 if args.mode == 'phospho' else 1200) and \
               ((args.mode == 'phospho' and valid_accuracies[-1] > best_val_accuracy) or \
                ( args.mode == 'kinase' and (valid_sensitivities[-1] + 0.25 * valid_precisions[-1]) > (best_val_sensitivity + 0.25 * best_val_precision))):
                best_val_accuracy = valid_accuracies[-1]
                best_val_precision = valid_precisions[-1]
                best_val_sensitivity = valid_sensitivities[-1]
                best_val_specificity = valid_specificities[-1]

                best_train_accuracy = train_accuracies[-1]
                best_train_precision = train_precisions[-1]
                best_train_sensitivity = train_sensitivities[-1]
                best_train_specificity = train_specificities[-1]
                test_accuracies_batches = []
                test_precisions_batches = []
                test_sensitivity_batches = []
                test_specificity_batches = []
                test_mcc_batches = []
                with torch.no_grad():
                    model.eval()
                    
                    for inputs, targets in iter(test_loader):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        outputs = model(inputs)
                        predictions = torch.round(outputs)
                            
                        test_accuracy, test_precision, test_sensitivity, test_specificity, mcc = perf_scores(predictions, targets)
                        test_accuracies_batches.append(test_accuracy)
                        test_precisions_batches.append(test_precision)
                        test_sensitivity_batches.append(test_sensitivity)
                        test_specificity_batches.append(test_specificity)
                        test_mcc_batches.append(mcc)
                            
                    model.train()
                
                best_test_accuracy = sum(test_accuracies_batches) / len(test_accuracies_batches)
                best_test_precision = sum(test_precisions_batches) / len(test_precisions_batches)
                best_test_sensitivity = sum(test_sensitivity_batches) / len(test_sensitivity_batches)
                best_test_specificity = sum(test_specificity_batches) / len(test_specificity_batches)
                best_test_mcc = sum(test_mcc_batches) / len(test_mcc_batches)
                

                save_model(model, train_acc=train_accuracies[-1], train_loss=train_loss, valid_acc=valid_accuracies[-1], test_mcc=best_test_mcc,
                           test_acc=best_test_accuracy, test_spec=best_test_specificity, test_prec=best_test_precision, test_sens=best_test_sensitivity,
                           valid_loss=float(loss))

if early_stop:
    print('Accuracies of the best run:')
    print(f'  Train {best_train_accuracy:8.5f}   Valid {best_val_accuracy:8.5f}   Test {best_test_accuracy:8.5f}')
    print('Precisions of the best run:')
    print(f'  Train {best_train_precision:8.5f}   Valid {best_val_precision:8.5f}   Test {best_test_precision:8.5f}')
    print('Sensitivities of the best run:')
    print(f'  Train {best_train_sensitivity:8.5f}   Valid {best_val_sensitivity:8.5f}   Test {best_test_sensitivity:8.5f}')
    print('Specificities of the best run:')
    print(f'  Train {best_train_specificity:8.5f}   Valid {best_val_specificity:8.5f}   Test {best_test_specificity:8.5f}')


model.eval()
valid_accuracies_batches = []
valid_precisions_batches = []
valid_sensitivity_batches = []
valid_specificity_batches = []
with torch.no_grad():
    for inputs, targets in iter(valid_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predictions = torch.round(outputs)

        valid_accuracy, valid_precision, valid_sensitivity, valid_specificity, mcc = perf_scores(predictions, targets)
        valid_accuracies_batches.append(valid_accuracy)
        valid_precisions_batches.append(valid_precision)
        valid_sensitivity_batches.append(valid_sensitivity)
        valid_specificity_batches.append(valid_specificity)
                    
valid_accuracies.append(sum(valid_accuracies_batches) / len(valid_accuracies_batches))
valid_precisions.append(sum(valid_precisions_batches) / len(valid_precisions_batches))
valid_sensitivities.append(sum(valid_sensitivity_batches) / len(valid_sensitivity_batches))
valid_specificities.append(sum(valid_specificity_batches) / len(valid_specificity_batches))

test_accuracies_batches = []
test_precisions_batches = []
test_sensitivity_batches = []
test_specificity_batches = []
test_mcc_batches = []
with torch.no_grad():
    for inputs, targets in iter(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predictions = torch.round(outputs)

        test_accuracy, test_precision, test_sensitivity, test_specificity, mcc = perf_scores(predictions, targets)
        test_accuracies_batches.append(test_accuracy)
        test_precisions_batches.append(test_precision)
        test_sensitivity_batches.append(test_sensitivity)
        test_specificity_batches.append(test_specificity)
        test_mcc_batches.append(mcc)
        
            

test_acc = sum(test_accuracies_batches) / len(test_accuracies_batches)
test_precision = sum(test_precisions_batches) / len(test_precisions_batches)
test_sensitivity = sum(test_sensitivity_batches) / len(test_sensitivity_batches)
test_specificity = sum(test_specificity_batches) / len(test_specificity_batches)
test_mcc = sum(test_mcc_batches) / len(test_mcc_batches)


print('Accuracies of the last step:')
print(f'  Train {train_accuracies[-1]:8.5f}   Valid {valid_accuracies[-1]:8.5f}   Test {test_acc:8.5f}')
print('Precisions of the last step:')
print(f'  Train {train_precisions[-1]:8.5f}   Valid {valid_precisions[-1]:8.5f}   Test {test_precision:8.5f}')
print('Sensitivities of the last step:')
print(f'  Train {train_sensitivities[-1]:8.5f}   Valid {valid_sensitivities[-1]:8.5f}   Test {test_sensitivity:8.5f}')
print('Specificities of the last step:')
print(f'  Train {train_specificities[-1]:8.5f}   Valid {valid_specificities[-1]:8.5f}   Test {test_specificity:8.5f}')
print(f'{test_mcc=}')


save_model(model, train_acc=train_accuracies[-1], valid_acc=valid_accuracies[-1],
           test_acc=test_acc, train_loss=0.0, valid_loss=0.0, final=True, test_mcc=test_mcc,
           test_spec=test_specificity, test_prec=test_precision, test_sens=test_sensitivity)

