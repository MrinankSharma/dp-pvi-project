

# set the path directly ....
export PYTHONPATH='/homes/ms2314/dp-pvi-project'
echo $PYTHONPATH

#python /homes/ms2314/dp-pvi-project/linreg/global_robust.py --output-base-dir /scratch/ms2314/ --overwrite --tag global-robust --N-seeds 50
#python /homes/ms2314/dp-pvi-project/linreg/global_robust.py --output-base-dir /scratch/ms2314/ --ppw --overwrite --tag global-robust-ppw --N-seeds 50

python /homes/ms2314/dp-pvi-project/linreg/global_robust.py --output-base-dir /scratch/ms2314/ --overwrite --tag updated-noise-free --N-seeds 50  --exp 4