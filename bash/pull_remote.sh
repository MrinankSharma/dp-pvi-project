#!/bin/bash

scp -rp ms2314@ulam:/scratch/ms2314/logs/gs_local_ana/ /Users/msharma/workspace/IIB/dp-pvi-project/remote-results-ulam/gs_local_ana
scp -rp ms2314@ulam:/scratch/ms2314/logs/gs_global_ana/ /Users/msharma/workspace/IIB/dp-pvi-project/remote-results-ulam/gs_global_ana
scp -rp ms2314@hinton:/scratch/ms2314/logs/gs_local_ana/ /Users/msharma/workspace/IIB/dp-pvi-project/remote-results-hinton/gs_local_ana
scp -rp ms2314@hinton:/scratch/ms2314/logs/gs_global_ana/ /Users/msharma/workspace/IIB/dp-pvi-project/remote-results-hinton/gs_global_ana
