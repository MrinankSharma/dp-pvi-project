dependencies:

python 2.7
tensorflow 1.12.0
matplotlib
ray 0.5.3 <https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.5.3-cp27-cp27m-macosx_10_6_intel.whl>
jupyter


example commands to train and test on a toy 1d dataset

python linreg_pvi_sync.py --num-workers=10 --data-type=inhomous

python linreg_pvi_async.py --num-workers=5 --data-type=inhomous --interval-time=0.1 --no-intervals=20

python linreg_pvi_test.py --param-file=/tmp/distributed_training/pvi_async_toy_1d_data_inhomous_seed_42_no_workers_5_interval_time_0.10_damping_0.000/params_interval_19.npz