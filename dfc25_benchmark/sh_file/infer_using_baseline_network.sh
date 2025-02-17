#!/bin/bash

DATASET_PATH=${1:-'/public/dfc25_track2_trainval/val'}
PROJ_PATH=${2:-'/workspace/WorkSpaceCls/BRIGHT'}

MODEL_NAME=${3:-'MUHSI_20250217_101216'} # UNet_20250115_151030(0.644) SiamCRNN_20250118_091842(0.666)

cd ..

python script/infer_using_baseline_network.py  --val_dataset_path ${DATASET_PATH}  \
                                               --model_type MUHSI \
                                               --val_data_list_path ${PROJ_PATH}'/dfc25_benchmark/dataset/splitname/val_setlevel.txt' \
                                               --existing_weight_path ${PROJ_PATH}'/dfc25_benchmark/weights/'${MODEL_NAME}'/best_model.pth' \
                                               --inferece_saved_path ${PROJ_PATH}'/dfc25_benchmark/inference_results'