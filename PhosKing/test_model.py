#! /usr/bin/env python3

import argparse
import sys, os
from sklearn import metrics

parser = argparse.ArgumentParser(prog='ESMTest', description='Find phosphorilation predictions with PyTorch using ESM Embeddings a trained model')

parser.add_argument('-i', '--input_file', action='store', dest='fasta_file', help='Fasta file with sequence to read')
parser.add_argument('-o', '--output_file', action='store', dest='output_file', help='Output file with predictions (not implemented yet)', default='')
parser.add_argument('-p', '--params', action='store', dest='params', help='Parameters for the ESM Embedding (available: 320, 1280)', default='1280')
parser.add_argument('-m', '--model_file', action='store', dest='model_file', help='Model file (python file with PyTorch model)')
parser.add_argument('-n', '--model_name', action='store', dest='model_name', help='Model name (class name in the model file)')
parser.add_argument('-a', '--model_args', action='store', dest='model_args', help='Comma separated ints to pass to the model constructor (e.g. 1280,2560,1)')
parser.add_argument('-sd', '--state_dict', action='store', dest='state_dict', help='State dict file of the trained model. Must match with parameters in model_args', default='')
parser.add_argument('-aaw', '--aa_window', action='store', dest='aa_window', help='Amino acid window for the tensors (concatenated tensor of the 5 amino acids)', default='0')
parser.add_argument('-2d', '--two_dims', action='store_true', dest='two_dims', help='If input tensors to the model must have 2 dimensions')
parser.add_argument('-md', '--mode', action='store', dest='mode', help='Prediction mode (phospho or kinase)', default='phospho')
parser.add_argument('-c', '--force_cpu', action='store_true', dest='force_cpu', help='Force CPU training')
args = parser.parse_args()

if args.model_file is None or args.model_name is None or args.state_dict is None:
    parser.print_help()
    sys.exit(1)

print(f'Using python env in {sys.executable}')

import torch
print(f'Using torch version {torch.__version__}')
from torch import nn
import torch.utils.data as data
from importlib import import_module
from dataset import ESM_Embeddings_test
import time as t
import numpy as np

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

state_dict = torch.load(args.state_dict, map_location = device)
model.load_state_dict(state_dict)

model.eval()

dataset = ESM_Embeddings_test(fasta_file=args.fasta_file,
                              params=int(args.params),
                              device=device,
                              aa_window=int(args.aa_window),
                              two_dims=args.two_dims,
                              mode=args.mode
)


predictions = dict()
for seq_ID, seq in dataset.seq_data:
    if seq_ID in dataset.IDs():
        with torch.no_grad():
            idxs, inputs = dataset[seq_ID]
            inputs = inputs.to(device)

            preds = model(inputs)
            preds = preds.detach().cpu().numpy().flatten()

        predictions[seq_ID] = dict()
        for i,pos in enumerate(dataset.idxs[seq_ID]):
            predictions[seq_ID][pos] = preds[i]

        if args.output_file:
            pass #TODO
        else:
            dots = ''
            i = 0
            for pos in range(len(seq)):
                if pos + 1 in dataset.idxs[seq_ID]:
                    if preds[i] > 0.95:
                        dots += '*'
                    elif preds[i] > 0.75:
                        dots += '+'
                    elif preds[i] > 0.5:
                        dots += '.'
                    else:
                        dots += ' '
                    i += 1
                else:
                    dots += ' '
            
            print('- ' * 41 + '\n > ' + seq_ID)

            for i in range(len(seq)//80+1):
                l = i*80
                print(dots[l:l+80])
                print(seq[l:l+80])
                print(' '*9+'|'+'|'.join(list('{:<9}'.format(l+j*10) for j in range(1,9))))


# If features file available (with true phosphorylations): compute accuracy scores
feat_file = ''
if feat_file:
    with open(feat_file) as feat_file:
        feat_file.readline()
        features = dict()
        for line in feat_file:
            seq_ID, _, pos = line.strip().split('\t')
            if features.get(seq_ID) is None:
                features[seq_ID] = list()
            features[seq_ID].append(int(pos))

    print('\n{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format('Threshold', 'Accur.', 'Precis.', 'Sensit.', 'Specif.', 'MCC'))
    for threshold in (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.999):
        accuracy = list()
        precision = list()
        sensitivity = list()
        specificity = list()
        MCC = list()

        predictions_list = list()
        features_list = list()
        
        pred_neg = set()
        pred_pos = set()
        feat_pos = set()
        for seq_ID in dataset.IDs():
            if features.get(seq_ID):
                for pos in features[seq_ID]:
                    feat_pos.add((seq_ID, pos))
                for pos, pred in predictions[seq_ID].items():
                    if pred > threshold:
                        pred_pos.add((seq_ID, pos))
                    else:
                        pred_neg.add((seq_ID, pos))
                    predictions_list.append(pred)
                    if (seq_ID, pos) in feat_pos:
                        features_list.append(1)
                    else:
                        features_list.append(0)

        feat_neg = pred_neg.union(pred_pos).difference(feat_pos)

        TP = len(pred_pos.intersection(feat_pos))
        TN = len(pred_neg.intersection(feat_neg))
        FP = len(pred_pos.intersection(feat_neg))
        FN = len(pred_neg.intersection(feat_pos))

        try:
            accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
        except ZeroDivisionError:
            accuracy = 0
        try:
            precision = (TP) / (TP + FP) * 100
        except ZeroDivisionError:
            precision = 0
        try:
            sensitivity = (TP) / (TP + FN) * 100
        except ZeroDivisionError:
            sensitivity = 0
        try:
            specificity = (TN) / (TN + FP) * 100
        except ZeroDivisionError:
            specificity = 0
        try:
            MCC = (TP * TN - FP * FN)/(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))) * 100
        except ZeroDivisionError:
            MCC = 0

        print('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}'.format(threshold,
            round(accuracy, 2), round(precision, 2),
            round(sensitivity, 2), round(specificity, 2), round(MCC, 2)))

    fpr, tpr, thresholds = metrics.roc_curve(features_list, predictions_list, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('AUC:', auc)