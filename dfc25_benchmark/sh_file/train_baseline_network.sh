#!/bin/bash

DATASET_PATH=${1:-'/public/dfc25_track2_trainval/train'}
PROJ_PATH=${2:-'/workspace/WorkSpaceCls/BRIGHT'}

cd ..

# python script/train_baseline_network.py  --dataset 'BRIGHT' \
#                                 --train_batch_size 8 \
#                                 --eval_batch_size 8 \
#                                 --num_workers 0 \
#                                 --crop_size 640 \
#                                 --max_iters 800000 \
#                                 --learning_rate 1e-4 \
#                                 --model_type 'UNet' \
#                                 --train_dataset_path ${DATASET_PATH} \
#                                 --train_data_list_path ${PROJ_PATH}'/dfc25_benchmark/dataset/splitname/train_setlevel.txt' \
#                                 --holdout_dataset_path ${DATASET_PATH} \
#                                 --holdout_data_list_path ${PROJ_PATH}'/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt' \
#                                 --model_param_path ${PROJ_PATH}'/dfc25_benchmark/weights'

python script/train_baseline_network.py  --dataset 'BRIGHT' \
                                --train_batch_size 1 \
                                --eval_batch_size 1 \
                                --val_internal 1000 \
                                --num_workers 2 \
                                --crop_size 640 \
                                --max_iters 100000 \
                                --learning_rate 1e-5 \
                                --model_type 'MUHSI' \
                                --train_dataset_path ${DATASET_PATH} \
                                --train_data_list_path ${PROJ_PATH}'/dfc25_benchmark/dataset/splitname/train_setlevel.txt' \
                                --holdout_dataset_path ${DATASET_PATH} \
                                --holdout_data_list_path ${PROJ_PATH}'/dfc25_benchmark/dataset/splitname/holdout_setlevel.txt' \
                                --model_param_path ${PROJ_PATH}'/dfc25_benchmark/weights'