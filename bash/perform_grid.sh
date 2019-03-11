#!/bin/bash

# set the path directly ....
export PYTHONPATH='/Users/msharma/workspace/IIB/dp-pvi-project'
echo $PYTHONPATH

# try different means
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/grid_search_ana.py
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/grid_search_global_ana.py