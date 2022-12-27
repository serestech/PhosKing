#!/usr/bin/env python3
import torch
import pickle
import os

device = torch.device('cpu')
repr_path = 'data/representations/'
dir_list = os.listdir(repr_path)

for filename in dir_list:
    with open(repr_path + filename, 'rb') as f:
        d = pickle.load(f)

    for id in d.keys():
        d[id] = d.get(id).to(device)

    with open(repr_path + filename, 'wb') as f:
        pickle.dump(d, f)

    print(filename, 'done')