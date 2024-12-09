#!/usr/bin/env bash

set -x
NPROC_PER_NODE=$1
MODEL_DIR=$2
HICO_PATH=$3
EPOCH_NUM=$4


python -m torch.distributed.launch \
        --master_port=12345 \
        --nproc_per_node=${NPROC_PER_NODE} \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --dataset_file hico \
        --hoi_path ${HICO_PATH} \
        --num_obj_classes 80 \
        --num_verb_classes 116 \
        --backbone focall \
        --batch_size 1 \
        --num_queries 200 \
        --use_nms_filter \
        --input_img_size 640 \
        --pretrained ${MODEL_DIR} \
        --eval \
        --use_distributed \
        --in_channels 192 384 768 1536 \
        --freeze_lang_encoder \
        --sam_model large \
        --num_workers 2 \
        --hoi \
        --test_epoch ${EPOCH_NUM} \
        --use_mask \
        --remove_no_interaction