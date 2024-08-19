#!/usr/bin/env bash

_GPU_ID="${1}"

# May be comma separated list of multiple dimensions: '128,256,512'
_LATENT_DIMENSIONS="${2}"

if [[ -z "${_GPU_ID}" ]]; then
    echo "No GPU ID provided."
    exit 1
fi

if [[ -z "${_LATENT_DIMENSIONS}" ]]; then
    echo "Missing latent dimensions."
    exit 2
fi

OLD_IFS="${IFS}"
IFS=','
read -ra latent_dimensions <<< "${_LATENT_DIMENSIONS}"
IFS="${OLD_IFS}"

dataset_names=("Imagenette" "STL10" "CIFAR10" "FGVCAircraft" "CIFAR100")

max_runs=$((\
    ${#dataset_names[@]} *\
    ${#latent_dimensions[@]}\
))

set -a
source .env

run_counter=1
for LATENT_DIMENSIONS in "${latent_dimensions[@]}"; do
    for DATASET_NAME in "${dataset_names[@]}"; do
        if [[ "${DATASET_NAME}" == "CIFAR10" ]] || [[ "${DATASET_NAME}" == "CIFAR100" ]]; then
            IMAGE_SIZE=32
        elif [[ "${DATASET_NAME}" == "STL10" ]]; then
            IMAGE_SIZE=96
        elif [[ "${DATASET_NAME}" == "Imagenette" ]]; then
            IMAGE_SIZE=128
        elif [[ "${DATASET_NAME}" == "FGVCAircraft" ]]; then
            IMAGE_SIZE=128
        else
            echo "Unsupported dataset '${DATASET_NAME}'."
            exit 1
        fi

        DEVICES="${_GPU_ID},"

        echo "Train run ${run_counter}/${max_runs}:"
        echo "  - DATASET_NAME=${DATASET_NAME}"
        echo "  - IMAGE_SIZE=${IMAGE_SIZE}"
        echo "  - LATENT_DIMENSIONS=${LATENT_DIMENSIONS}"
        echo "  - DEVICES=${DEVICES}"

        ./run.sh

        (( run_counter++ ))
    done
done
