#!/bin/bash

rsync -rv ms2314@ulam:/scratch/ms2314/ /Users/msharma/workspace/IIB/dp-pvi-project/ulam-scratch
rsync -rv ms2314@hinton:/scratch/ms2314/ /Users/msharma/workspace/IIB/dp-pvi-project/hinton-scratch

rsync -rv ms2314@hinton:/scratch/ms2314/logs/gs_local_final_ana/ /Users/msharma/workspace/IIB/dp-pvi-project/hinton-scratch/logs/gs_local_final_ana/