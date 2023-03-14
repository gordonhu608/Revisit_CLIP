#!/bin/bash

#cd ../..

# custom config
DATA="/path/to/dataset/folder"
TRAINER=Ours

DATASET=$1
SEED=$2

CFG=vit_b16
TTA=True

for SHOTS in 1 2 4 8 16
do
    DIR=output/fewshot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Resuming..."
        python train_fewshot.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --tta ${TTA} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --tta ${TTA} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES all
    fi
done