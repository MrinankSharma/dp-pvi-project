#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

# try different means
python /homes/ms2314/dp-pvi-project/linreg/grid_search_ana.py /scratch/ms2314/
python /homes/ms2314/dp-pvi-project/linreg/grid_search_global_ana.py --output-base-dir /scratch/ms2314/