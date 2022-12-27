#! /usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $(basename $0) <dest>"
    exit 1
fi

dest=$1

module purge
module load python3/3.9.6 cuda/11.7

virtualenv -p=python3 --verbose --always-copy $dest

$dest/bin/python -m pip install --default-timeout=1000 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
$dest/bin/python -m pip install pandas scikit-learn fair-esm

module purge
