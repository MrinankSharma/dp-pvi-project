#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

python /homes/ms2314/dp-pvi-project/linreg/global_sample_invest.py --output-base-dir /scratch/ms2314/ --overwrite --tag global-param-choice-updated --N-seeds 50 --no-workers 20
