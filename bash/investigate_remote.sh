#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

python /homes/ms2314/dp-pvi-project/linreg/investigate_bias.py --output-base-dir /scratch/ms2314/ --tag check-bias --N-dp-seeds 10
