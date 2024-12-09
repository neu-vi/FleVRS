#!/usr/bin/env bash

set -x
NPROC_PER_NODE=$1
MODEL_DIR=$2
VCOCO_PATH=$3

python -m torch.distributed.launch \
        --master_port=13456 \
        --nproc_per_node=${NPROC_PER_NODE} \
        --use_env \
        main.py \
        --output_dir ${EXP_DIR} \
        --dataset_file vcoco \
        --vcoco_path ${VCOCO_PATH} \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --backbone focall \
        --batch_size 1 \
        --num_queries 200 \
        --use_nms_filter \
        --input_img_size 640 \
        --pretrained ${MODEL_DIR} \
        --eval \
        --use_distributed \
        --in_channels 192 384 768 1536 \
        --sam_model large \
        --num_workers 0 \
        --hoi \
        --freeze_lang_encoder \
        --use_triplet