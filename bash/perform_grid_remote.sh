#!/bin/bash

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

python /homes/ms2314/dp-pvi-project/linreg/grid_search_ana.py --output-base-dir /scratch/ms2314/ --tag damped
python /homes/ms2314/dp-pvi-project/linreg/grid_search_global_ana.py --output-base-dir /scratch/ms2314/ --tag more_workers_damped_two
python /homes/ms2314/dp-pvi-project/linreg/grid_search_ana.py --output-base-dir /scratch/ms2314/ --tag damped_more_workers --no-workers 40 --N-dp-seeds 4
