#!/usr/bin/env bash

if [[ "${STANDARDIZE}" == "true" ]]; then
    STANDARDIZE="--standardize"
else
    STANDARDIZE=
fi

eval "python main.py \
    --dataset-dir ${DATASET_DIR} \
    --dataset-name ${DATASET_NAME} \
    --train-perc ${TRAIN_PERC} \
    --val-perc ${VAL_PERC} \
    --test-perc ${TEST_PERC} \
    --log-dir ${LOG_DIR} \
    --devices ${DEVICES} \
    --img-size ${IMAGE_SIZE} \
    --num-channels ${NUM_CHANNELS} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --save-top-k ${SAVE_TOP_K} \
    --batch-size ${BATCH_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr-scheduler ${LR_SCHEDULER} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --latent-dimensions ${LATENT_DIMENSIONS} \
    --num-workers ${NUM_WORKERS} \
    --experiment-name ${EXPERIMENT_NAME} \
    --use-mlflow \
    ${STANDARDIZE}"