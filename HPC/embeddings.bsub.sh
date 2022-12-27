#!/bin/sh
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J precompute_embeddings
### -- ask for number of cores (default: 1) --
#BSUB -n 10
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:45
# -- request system memory --
#BSUB -R "rusage[mem=50GB]"
# -- request gpu memory --
#BSUB -R "select[gpu80gb]"
# -- request only 1 node --
#BSUB -R "span[hosts=1]"
### -- set the email address --
#BSUB -u danyia@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/52/c/174062/logs_o/
#BSUB -e /zhome/52/c/174062/logs_e/

module purge
module load python3/3.9.6 cuda/11.7

echo "Working directory: $(pwd)"
echo "User: $(whoami)"

export PATH=~/torch_pyenv/bin:${PATH}

/zhome/52/c/174062/DL/precompute_embeddings.py
