#!/usr/bin/env bash

_GPU_ID="${1}"
_LATENT_DIMENSIONS="${2}"

if [[ -z "${_GPU_ID}" ]]; then
    echo "No GPU ID provided."
    exit 1
fi

if [[ -z "${_LATENT_DIMENSIONS}" ]]; then
    echo "No latent dimensions provided for projection space."
    exit 2
fi

# Argument values to main.py
dataset_name=("Imagenette" "STL10" "CIFAR10" "FGVCAircraft" "CIFAR100")
latent_dimensions=("${_LATENT_DIMENSIONS}") # "32" "64" "128"
projection_head=("ae" "mlp")
standardize=("false" "true")
freeze_projection_head=("true")
projection_head_embedding_dimension=("128" "256" "512")
projection_head_activation=("ReLU" "SiLU" "Sigmoid" "Tanh")

max_runs=$((\
    ${#dataset_name[@]} *\
    ${#latent_dimensions[@]} *\
    ${#projection_head[@]} *\
    ${#standardize[@]} *\
    ${#freeze_projection_head[@]} *\
    ${#projection_head_embedding_dimension[@]} *\
    ${#projection_head_activation[@]}\
))

AE_WEIGHTS_ROOT="../ae/logs/csv/ae-train"
AE_WEIGHTS_VERSION="version_0/checkpoints"

RUNS_CSV_FILENAME="runs-${_GPU_ID}-${_LATENT_DIMENSIONS}.csv"
echo "Dataset Name,Latent Dimensions,Projection Head,Projection Head Embedding Dim,Standardize,Freeze Projection Head,Projection Head Activation" > "${RUNS_CSV_FILENAME}"

set -a
source .env

run_counter=1
for DATASET_NAME in "${dataset_name[@]}"; do
    for LATENT_DIMENSIONS in "${latent_dimensions[@]}"; do
        for PROJECTION_HEAD in "${projection_head[@]}"; do
            for PROJECTION_HEAD_EMBEDDING_DIM in "${projection_head_embedding_dimension[@]}"; do
                for STANDARDIZE in "${standardize[@]}"; do
                    for FREEZE_PROJECTION_HEAD in "${freeze_projection_head[@]}"; do
                        for PROJECTION_HEAD_ACTIVATION in "${projection_head_activation[@]}"; do
                            if [[ "${DATASET_NAME}" == "CIFAR10" ]]; then
                                if [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                    if [[ "${IMAGE_SIZE}" == 32 ]]; then
                                        AE_WEIGHTS_128_PATH="${AE_WEIGHTS_ROOT}/4e7aa851bbb646a3bdf96de4ea1637f4/${AE_WEIGHTS_VERSION}/epoch=98-step=44550.ckpt"
                                        AE_WEIGHTS_256_PATH="${AE_WEIGHTS_ROOT}/950038af759d407d9e81e85d36a989ff/${AE_WEIGHTS_VERSION}/epoch=99-step=45000.ckpt"
                                        AE_WEIGHTS_512_PATH="${AE_WEIGHTS_ROOT}/4b12920c4f134e61ba506b3e1cda7818/${AE_WEIGHTS_VERSION}/epoch=99-step=45000.ckpt"
                                    else
                                        echo "No AE weights for dataset ${DATASET_NAME} with image size ${IMAGE_SIZE}."
                                        exit 1
                                    fi
                                else
                                    IMAGE_SIZE=32
                                fi
                            elif [[ "${DATASET_NAME}" == "CIFAR100" ]]; then
                                if [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                    if [[ "${IMAGE_SIZE}" == 32 ]]; then
                                        AE_WEIGHTS_128_PATH="${AE_WEIGHTS_ROOT}/5277bcc207c74bfd8b7d390ccde6ea90/${AE_WEIGHTS_VERSION}/epoch=99-step=45000.ckpt"
                                        AE_WEIGHTS_256_PATH="${AE_WEIGHTS_ROOT}/3df7185522e643129ee160079eac7a57/${AE_WEIGHTS_VERSION}/epoch=99-step=45000.ckpt"
                                        AE_WEIGHTS_512_PATH="${AE_WEIGHTS_ROOT}/faf348a6d66c4043869ccf549a7ede7b/${AE_WEIGHTS_VERSION}/epoch=99-step=45000.ckpt"
                                    else
                                        echo "No AE weights for dataset ${DATASET_NAME} with image size ${IMAGE_SIZE}."
                                        exit 2
                                    fi
                                else
                                    IMAGE_SIZE=32
                                fi
                            elif [[ "${DATASET_NAME}" == "STL10" ]]; then
                                if [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                    if [[ "${IMAGE_SIZE}" == 96 ]]; then
                                        AE_WEIGHTS_128_PATH="${AE_WEIGHTS_ROOT}/5466ab1f02574d2fa9d28653fbc71fcf/${AE_WEIGHTS_VERSION}/epoch=99-step=4500.ckpt"
                                        AE_WEIGHTS_256_PATH="${AE_WEIGHTS_ROOT}/d52d5583d60c40c7921b808b6095441a/${AE_WEIGHTS_VERSION}/epoch=99-step=4500.ckpt"
                                        AE_WEIGHTS_512_PATH="${AE_WEIGHTS_ROOT}/43bcca8987eb4cf48aec035a8eeea7b6/${AE_WEIGHTS_VERSION}/epoch=99-step=4500.ckpt"
                                    else
                                        echo "No AE weights for dataset ${DATASET_NAME} with image size ${IMAGE_SIZE}."
                                        exit 3
                                    fi
                                else
                                    IMAGE_SIZE=96
                                fi

                                if [[ "${VAL_PERC}" == "0.1" ]]; then
                                    BATCH_SIZE=500
                                    echo "With dataset '${DATASET_NAME}' the batch size has to be reduced to ${BATCH_SIZE} to get at least a single validation batch with val_perc=${VAL_PERC}."
                                fi
                            elif [[ "${DATASET_NAME}" == "Imagenette" ]]; then
                                if [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                    if [[ "${IMAGE_SIZE}" == 128 ]]; then
                                        AE_WEIGHTS_128_PATH="${AE_WEIGHTS_ROOT}/bfde1a3da1cd4a4fa4355532b2e11e18/${AE_WEIGHTS_VERSION}/epoch=98-step=8514.ckpt"
                                        AE_WEIGHTS_256_PATH="${AE_WEIGHTS_ROOT}/0757bf5c101f4b93b6423c33e63cb1ea/${AE_WEIGHTS_VERSION}/epoch=99-step=8600.ckpt"
                                        AE_WEIGHTS_512_PATH="${AE_WEIGHTS_ROOT}/4c317e7cffa24981bb08c56f215d1e96/${AE_WEIGHTS_VERSION}/epoch=96-step=8342.ckpt"
                                    else
                                        echo "No AE weights for dataset ${DATASET_NAME} with image size ${IMAGE_SIZE}."
                                        exit 4
                                    fi
                                else
                                    IMAGE_SIZE=128
                                fi

                                if [[ "${VAL_PERC}" == "0.1" ]]; then
                                    BATCH_SIZE=392
                                    echo "With dataset '${DATASET_NAME}' the batch size has to be reduced to ${BATCH_SIZE} to get at least a single validation batch with val_perc=${VAL_PERC}."
                                fi
                            elif [[ "${DATASET_NAME}" == "FGVCAircraft" ]]; then
                                if [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                    if [[ "${IMAGE_SIZE}" == 128 ]]; then
                                        AE_WEIGHTS_128_PATH="${AE_WEIGHTS_ROOT}/f4d6c115ba4e41759c3a0b8aa628735f/${AE_WEIGHTS_VERSION}/epoch=99-step=6100.ckpt"
                                        AE_WEIGHTS_256_PATH="${AE_WEIGHTS_ROOT}/edeeefb6d47f4530b4616e554a890ba2/${AE_WEIGHTS_VERSION}/epoch=97-step=5978.ckpt"
                                        AE_WEIGHTS_512_PATH="${AE_WEIGHTS_ROOT}/9fdf277256ca4f9e915ef74c633f10d8/${AE_WEIGHTS_VERSION}/epoch=93-step=5734.ckpt"
                                    else
                                        echo "No AE weights for dataset ${DATASET_NAME} with image size ${IMAGE_SIZE}."
                                        exit 5
                                    fi
                                else
                                    IMAGE_SIZE=128
                                fi

                                if [[ "${VAL_PERC}" == "0.1" ]]; then
                                    BATCH_SIZE=666
                                    echo "With dataset '${DATASET_NAME}' the batch size has to be reduced to ${BATCH_SIZE} to get at least a single validation batch with val_perc=${VAL_PERC}."
                                fi
                            else
                                echo "Unsupported dataset '${DATASET_NAME}'."
                                exit 5
                            fi

                            if [[ "${PROJECTION_HEAD}" == "mlp" ]]; then
                                PROJECTION_HEAD_WEIGHTS=
                                if [[ "${FREEZE_PROJECTION_HEAD}" == "true" ]]; then
                                    echo "Freezing the weights of projection head 'mlp' does not have any effect."
                                    echo "This is only meant to be used with a pre-trained embedding of 'ae' projection head."
                                    echo "Not setting --freeze-projection-head..."
                                    FREEZE_PROJECTION_HEAD="false"
                                fi
                            elif [[ "${PROJECTION_HEAD}" == "ae" ]]; then
                                if [[ "${PROJECTION_HEAD_EMBEDDING_DIM}" == "128" ]]; then
                                    PROJECTION_HEAD_WEIGHTS="${AE_WEIGHTS_128_PATH}"
                                elif [[ "${PROJECTION_HEAD_EMBEDDING_DIM}" == "256" ]]; then
                                    PROJECTION_HEAD_WEIGHTS="${AE_WEIGHTS_256_PATH}"
                                elif [[ "${PROJECTION_HEAD_EMBEDDING_DIM}" == "512" ]]; then
                                    PROJECTION_HEAD_WEIGHTS="${AE_WEIGHTS_512_PATH}"
                                else
                                    echo "Unsupported PROJECTION_HEAD_EMBEDDING_DIM=${PROJECTION_HEAD_EMBEDDING_DIM}" >&2
                                    exit 6
                                fi

                                if [[ ! -f "${PROJECTION_HEAD_WEIGHTS}" ]]; then
                                    echo "Weights '${PROJECTION_HEAD_WEIGHTS}' not found."
                                    exit 7
                                fi
                            else
                                echo "Unsupported PROJECTION_HEAD=${PROJECTION_HEAD}"
                                exit 8
                            fi

                            DEVICES="${_GPU_ID},"

                            echo "Train run ${run_counter}/${max_runs}:"
                            echo "  - DATASET_NAME=${DATASET_NAME}"
                            echo "  - IMAGE_SIZE=${IMAGE_SIZE}"
                            echo "  - BATCH_SIZE=${BATCH_SIZE}"
                            echo "  - LATENT_DIMENSIONS=${LATENT_DIMENSIONS}"
                            echo "  - PROJECTION_HEAD=${PROJECTION_HEAD}"
                            echo "  - PROJECTION_HEAD_WEIGHTS=${PROJECTION_HEAD_WEIGHTS}"
                            echo "  - PROJECTION_HEAD_EMBEDDING_DIM=${PROJECTION_HEAD_EMBEDDING_DIM}"
                            echo "  - STANDARDIZE=${STANDARDIZE}"
                            echo "  - FREEZE_PROJECTION_HEAD=${FREEZE_PROJECTION_HEAD}"
                            echo "  - PROJECTION_HEAD_ACTIVATION=${PROJECTION_HEAD_ACTIVATION}"
                            echo "  - DEVICES=${DEVICES}"

                            echo "${DATASET_NAME},${LATENT_DIMENSIONS},${PROJECTION_HEAD},${PROJECTION_HEAD_EMBEDDING_DIM},${STANDARDIZE},${FREEZE_PROJECTION_HEAD},${PROJECTION_HEAD_ACTIVATION}" >> "${RUNS_CSV_FILENAME}"

                            ./run.sh

                            (( run_counter++ ))
                        done
                    done
                done
            done
        done
    done
done
