#!/bin/sh
cd
cd /Users/cameron/Code/Python3.6/STIR_recon
python3 ./scripts/interfileToDICOM.py Input_Data/input.hv Input_Data/input.v  temp/unsmoothed.dcm
python3 ./scripts/gaussianSmoothing.py temp/unsmoothed.dcm temp/smoothed.dcm
python ./scripts/doChangCorrection.py temp/smoothed.dcm temp/smoothed_changMask.dcm
python3 ./scripts/applyChangCorrection.py temp/smoothed.dcm temp/smoothed_changMask.dcm temp/smoothed_changAC.dcm
python ./scripts/slicer.py temp/smoothed.dcm Output_Data/output.dcm 1000


