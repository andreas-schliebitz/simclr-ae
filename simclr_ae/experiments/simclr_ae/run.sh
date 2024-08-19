#!/usr/bin/env bash

if [[ "${FREEZE_PROJECTION_HEAD}" == "true" ]]; then
    FREEZE_PROJECTION_HEAD="--freeze-projection-head"
else
    FREEZE_PROJECTION_HEAD=
fi

if [[ "${STANDARDIZE}" == "true" ]]; then
    STANDARDIZE="--standardize"
else
    STANDARDIZE=
fi

if [[ -n "${PROJECTION_HEAD_ACTIVATION}" ]]; then
    PROJECTION_HEAD_ACTIVATION="--projection-head-activation ${PROJECTION_HEAD_ACTIVATION}"
fi

eval "python main.py \
    --dataset-dir ${DATASET_DIR} \
    --dataset-name ${DATASET_NAME} \
    --train-perc ${TRAIN_PERC} \
    --val-perc ${VAL_PERC} \
    --test-perc ${TEST_PERC} \
    --log-dir ${LOG_DIR} \
    --devices ${DEVICES} \
    --weights ${WEIGHTS} \
    --backbone ${BACKBONE} \
    --projection-head ${PROJECTION_HEAD} \
    --projection-head-weights '${PROJECTION_HEAD_WEIGHTS}' \
    --projection-head-embedding-dim ${PROJECTION_HEAD_EMBEDDING_DIM} \
    --img-size ${IMAGE_SIZE} \
    --num-channels ${NUM_CHANNELS} \
    --epochs ${EPOCHS} \
    --patience ${PATIENCE} \
    --save-top-k ${SAVE_TOP_K} \
    --batch-size ${BATCH_SIZE} \
    --optimizer ${OPTIMIZER} \
    --lr-scheduler ${LR_SCHEDULER} \
    --temperature ${TEMPERATURE} \
    --learning-rate ${LEARNING_RATE} \
    --momentum ${MOMENTUM} \
    --weight-decay ${WEIGHT_DECAY} \
    --latent-dimensions ${LATENT_DIMENSIONS} \
    --num-workers ${NUM_WORKERS} \
    --eval-train-perc ${TRAIN_PERC} \
    --eval-val-perc ${VAL_PERC} \
    --eval-test-perc ${TEST_PERC} \
    --eval-epochs ${EVAL_EPOCHS} \
    --eval-learning-rate ${EVAL_LEARNING_RATE} \
    --eval-weight-decay ${EVAL_WEIGHT_DECAY} \
    --experiment-name ${EXPERIMENT_NAME} \
    --experiment-eval-name ${EXPERIMENT_EVAL_NAME} \
    --use-mlflow \
    ${PROJECTION_HEAD_ACTIVATION} \
    ${FREEZE_PROJECTION_HEAD} \
    ${STANDARDIZE}"