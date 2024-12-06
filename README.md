# SimCLR-AE

This repository contains the Python code for reproducing our experiments conducted as part of the paper

> [Improving Nonlinear Projection Heads using Pretrained Autoencoder Embeddings](https://arxiv.org/abs/2408.14514)

by Andreas Schliebitz, Heiko Tapken and Martin Atzmueller.

## Installation

This project uses [`poetry`](https://python-poetry.org/) for managing Python dependencies (see [pyproject.toml](./pyproject.toml)). Follow the steps below to install the code from this repository as a standalone `simclr_ae` Python package:

1. Install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Create and activate a virtual environment:

```bash
poetry shell
```

3. Install the requirements:

```bash
poetry lock
poetry install
```

## Usage

This project is mainly subdivided into two experiments. Within the [first experiment](./simclr_ae/experiments/ae), we train the autoencoder embeddings used for replacing the input layer of SimCLR's default projection head. In our [second experiment](./simclr_ae/experiments/simclr_ae), we train and evaluate our modified projectors as part of the SimCLR framework following standard protocol.

In order to reproduce our results, you will have to prepare the following five image classification datasets in a way that Torchvision's [`dataset`](https://pytorch.org/vision/stable/datasets.html) module can load them:

* [Imagenette](https://pytorch.org/vision/stable/_modules/torchvision/datasets/imagenette.html#Imagenette)
* [STL10](https://pytorch.org/vision/stable/_modules/torchvision/datasets/stl10.html#STL10)
* [CIFAR10](https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10)
* [FGVCAircraft](https://pytorch.org/vision/stable/_modules/torchvision/datasets/fgvc_aircraft.html#FGVCAircraft)
* [CIFAR100](https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR100)

**Note**: We advise you to download and extract these datasets into this project's [`datasets`](./datasets) directory. We also recommend the use of [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html) to record all training and evaluation runs. As an alternative, we additionally implement tracking via Lightning's [`CSVLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.CSVLogger.html) by default.

After that, clone this repository to a location of your choice:

```bash
git clone https://github.com/andreas-schliebitz/simclr-ae.git \
    && cd simclr-ae
```

### Training the Autoencoder Embeddings

First, train the 15 autoencoder embeddings using varying latent dimensions (128, 256, 512). We'll use these embeddings in the next section to perform our SimCLR training and evaluation runs.

1. Navigate into the directory of the `ae` experiment:

    ```bash
    cd simclr_ae/experiments/ae
    ```

2. **Optional**: If applicable, provide your MLflow Tracking credentials in the `ae` experiment's [`.env`](./simclr_ae/experiments/ae/.env) file. If you've placed the datasets into a different directory, change `DATASET_DIR` to that path.

3. Execute the experiment's [`run_experiments.sh`](./simclr_ae/experiments/ae/run_experiments.sh) helper script. If you have multiple GPU's at your disposal, specify the GPU's ID as first parameter, otherwise use `0` as the ID of your single GPU. The second parameter can either be a comma separated list of latent dimensions or a single latent dimension. By default, each GPU trains the autoencoder with the specified number of latent dimensions on all datasets:

    ```bash
    # Train autoencoder on GPU 0 with all three latent dimensions
    ./run_experiments.sh 0 128,256,512

    # Train autoencoder on three GPUs, parallelizing over latent dimensions
    ./run_experiments.sh 0 128
    ./run_experiments.sh 1 256
    ./run_experiments.sh 2 512
    ```

4. Verify that all model checkpoints, hyperparameters and metrics are written into the [`logs`](./simclr_ae/experiments/ae/logs) directory.

### End-to-End Training of SimCLR with custom Projectors

1. Navigate into the directory of the `simclr_ae` experiment:

    ```bash
    cd simclr_ae/experiments/simclr_ae
    ```

2. **Optional**: If applicable, provide your MLflow Tracking credentials in the `simclr_ae` experiment's [`.env`](./simclr_ae/experiments/simclr_ae/.env) file. If you've placed the datasets into a different directory, change `DATASET_DIR` to that path.

3. Due to the run IDs of each pretrained autoencoder embedding being randomly generated, you'll have to adapt the IDs in the [`run_experiments.sh`](./simclr_ae/experiments/simclr_ae/run_experiments.sh) helper script of the `simclr_ae` experiment for each dataset (variables `AE_WEIGHTS_128_PATH`,  `AE_WEIGHTS_256_PATH` and `AE_WEIGHTS_512_PATH`). The script will throw an error if no pretrained autoencoder checkpoint with matching latent dimensions is found for a given dataset.

4. Execute the experiment's [`run_experiments.sh`](./simclr_ae/experiments/simclr_ae/run_experiments.sh) helper script. As the first argument, provide your GPU's ID followed by the second argument being the latent dimension of SimCLR's projection space (32, 64, 128).

    ```bash
    # Train on single GPU with single latent dimension
    ./run_experiments.sh 0 32

    # Train on three GPUs with different latent dimensions
    ./run_experiments.sh 0 32
    ./run_experiments.sh 1 64
    ./run_experiments.sh 2 128
    ```

5. Once again, verify that all model checkpoints, hyperparameters and metrics are written into the [`logs`](./simclr_ae/experiments/simclr_ae/logs) directory.

You can now verify our results by either inspecting the CSV files in the `logs` directory of the `ae` and `simclr_ae` experiment or by visiting the web interface of your MLFlow Tracking instance. As a basis for comparison, we provide our MLflow runs as CSV exports in the `results` directory of each experiment.
