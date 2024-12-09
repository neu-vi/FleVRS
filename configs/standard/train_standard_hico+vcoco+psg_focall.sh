#!/usr/bin/env bash

set -x
NPROC_PER_NODE=$1
LOG_DIR=$2
COCO_PATH=$3
PSG_PATH=$4
HICO_PATH=$5
VCOCO_PATH=$6
PRETRAINED=$7


python -m torch.distributed.launch \
        --master_port=12345 \
        --nproc_per_node=${NPROC_PER_NODE} \
        --use_env \
         main.py \
        --output_dir ${LOG_DIR} \
        --dataset_file hico+vcoco+psg \
        --vcoco_path ${VCOCO_PATH} \
        --coco_path ${COCO_PATH} \
        --psg_folder ${PSG_PATH} \
        --hoi_path ${HICO_PATH} \
        --num_obj_classes_psg 133 \
        --num_verb_classes_psg 56 \
        --num_obj_classes_hico 80 \
        --num_verb_classes_hico 116 \
        --num_obj_classes_vcoco 81 \
        --num_verb_classes_vcoco 29 \
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
        --use_triplet \
        --num_workers 4 \
        --lr 1e-5 \
        --lr_backbone 1e-6 \
        --psg \
        --hoi \
        --eos_coef_psg 0.02 \
        --eos_coef_hoi 0.1 \
        --sam_model large \
        --pretrained ${PRETRAINED}
        