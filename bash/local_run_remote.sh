#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

python /homes/ms2314/dp-pvi-project/linreg/local_sample_invest.py --output-base-dir /scratch/ms2314/ --overwrite --tag final-noisy --N-seeds 50
python /homes/ms2314/dp-pvi-project/linreg/local_sample_invest.py --output-base-dir /scratch/ms2314/ --no-noise --overwrite --tag ficd nal-not-noisy --N-seeds 50