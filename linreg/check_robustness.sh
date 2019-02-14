#!/bin/bash

# set the path directly ....
export PYTHONPATH='/Users/msharma/workspace/IIB/dp-pvi-project'
echo $PYTHONPATH

# try different means
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean -10 --noise-std 1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean -0.25 --noise-std 1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 0 --noise-std 1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 0.05 --noise-std 1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 6 --noise-std 1

# try different noise standard deviations.
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 1 --noise-std 0.1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 1 --noise-std 1
python /Users/msharma/workspace/IIB/dp-pvi-project/linreg/linreg_dpsgd_pvi_sync.py --num-workers=5 --data-type=homous --no-intervals 10 --damping 0.0 --mean 1 --noise-std 10