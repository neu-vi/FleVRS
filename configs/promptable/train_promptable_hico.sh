#!/usr/bin/env bash
set -x
NPROC_PER_NODE=$1
LOG_DIR=$2
HICO_PATH=$3
PRETRAINED=$4


python -m torch.distributed.launch \
        --master_port=12345 \
        --nproc_per_node=${NPROC_PER_NODE} \
        --use_env \
         main.py \
        --output_dir ${LOG_DIR} \
        --dataset_file hico \
        --hoi_path ${HICO_PATH} \
        --num_obj_classes 80 \
        --num_verb_classes 116 \
        --backbone focall \
        --batch_size 4 \
        --num_queries 200 \
        --in_channels 192 384 768 1536 \
        --epochs 30 \
        --lr_drop 20 \
        --use_nms_filter \
        --input_img_size 640 \
        --pretrained ${PRETRAINED} \
        --freeze_lang_encoder \
        --use_distributed \
        --sam_model large \
        --num_workers 4 \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --hoi \
        --flexible_grounding \
        --use_mask \
        --remove_no_interaction \
        --save_model_epoch 5
