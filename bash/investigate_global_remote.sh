#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

python /homes/ms2314/dp-pvi-project/linreg/investigate_bias_global_ana.py --output-base-dir /scratch/ms2314/ --overwrite --tag anneal-damping-fixed --N-dp-seeds 10
