#!/usr/bin/env bash

set -x
NPROC_PER_NODE=$1
LOG_DIR=$2
COCO_PATH=$3
PSG_PATH=$4
PRETRAINED=$5


python -m torch.distributed.launch \
        --master_port=12345 \
        --nproc_per_node=${NPROC_PER_NODE} \
        --use_env \
         main.py \
        --output_dir ${LOG_DIR} \
        --dataset_file psg \
        --coco_path ${COCO_PATH} \
        --psg_folder ${PSG_PATH} \
        --num_obj_classes 133 \
        --num_verb_classes 56 \
        --backbone focall \
        --batch_size 8 \
        --num_queries 200 \
        --in_channels 192 384 768 1536 \
        --epochs 60 \
        --lr_drop 20 \
        --use_nms_filter \
        --input_img_size 640 \
        --freeze_lang_encoder \
        --use_distributed \
        --num_workers 4 \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --psg \
        --eos_coef 0.02 \
        --sam_model large \
        --pretrained ${PRETRAINED}
        