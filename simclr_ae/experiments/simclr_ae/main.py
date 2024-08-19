#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simclr_ae.utils import seed_rng

seed_rng()

import torch.nn as nn
import lightning as L

from pprint import pprint
from torch.utils.data import Dataset
from simclr_ae.experiments.args import get_parser
from simclr_ae.models.simclr.model import SimCLRModel
from simclr_ae.models.simclr.module import SimCLRModule
from simclr_ae.models.logreg.dataset import FeatureDataset
from simclr_ae.models.simclr.utils import create_simclr_model
from simclr_ae.models.logreg.module import LogisticRegressionModule
from simclr_ae.experiments.args import add_args as add_experiments_args
from simclr_ae.experiments.simclr_ae.simclr_datamodule import SimCLRDataModule
from simclr_ae.experiments.simclr_ae.args import add_args as add_simclr_ae_args
from simclr_ae.utils import (
    get_backbone_model,
    print_dataset_summary,
    create_loggers,
    create_trainer_callbacks,
    get_parameter_stats,
)
from simclr_ae.experiments.simclr_ae.logreg_datamodule import (
    LogisticRegressionDataModule,
)


if __name__ == "__main__":
    arg_parser = get_parser()
    arg_parser = add_experiments_args(arg_parser)
    arg_parser = add_simclr_ae_args(arg_parser)
    ARGS = arg_parser.parse_args()

    params = vars(ARGS)
    pprint(params)

    # ===== Create loggers for SimCLR =====

    simclr_loggers = create_loggers(
        log_dir=ARGS.log_dir,
        experiment_name=ARGS.experiment_name,
        run_name=ARGS.run_name,
        use_mlflow=ARGS.use_mlflow,
    )

    # ===== SimCLR Training =====

    simclr_datamodule: L.LightningDataModule = SimCLRDataModule(
        dataset_dir=ARGS.dataset_dir,
        dataset_name=ARGS.dataset_name,
        train_perc=ARGS.train_perc,
        val_perc=ARGS.val_perc,
        test_perc=ARGS.test_perc,
        img_size=ARGS.img_size,
        batch_size=ARGS.batch_size,
        standardize=ARGS.standardize,
        num_workers=ARGS.num_workers,
        logger=simclr_loggers.values(),
    )

    backbone_model = get_backbone_model(backbone=ARGS.backbone, weights=ARGS.weights)

    f, g, num_features, projection_head_params = create_simclr_model(
        model=backbone_model,
        batch_size=ARGS.batch_size,
        latent_dim=ARGS.latent_dimensions,
        num_channels=ARGS.num_channels,
        img_size=ARGS.img_size,
        projection_head_type=ARGS.projection_head,
        projection_head_activation=getattr(nn, ARGS.projection_head_activation),
        projection_head_embedding_dim=ARGS.projection_head_embedding_dim,
        projection_head_weights=ARGS.projection_head_weights,
        freeze_projection_head=ARGS.freeze_projection_head,
    )
    params |= projection_head_params

    simclr_model: nn.Module = SimCLRModel(f=f, g=g)
    print("SimCLR model:", simclr_model)

    learning_rate = (
        ARGS.learning_rate if ARGS.learning_rate > 0 else 0.3 * ARGS.batch_size / 256
    )
    params["learning_rate"] = learning_rate

    simclr_module: L.LightningModule = SimCLRModule(
        model=simclr_model,
        epochs=ARGS.epochs,
        learning_rate=learning_rate,
        momentum=ARGS.momentum,
        weight_decay=ARGS.weight_decay,
        temperature=ARGS.temperature,
    )
    params |= get_parameter_stats(module=simclr_module, prefix="simclr_")

    simclr_trainer_callbacks = create_trainer_callbacks(
        patience=ARGS.patience, save_top_k=ARGS.save_top_k
    )
    params["simclr_trainer_callbacks"] = list(simclr_trainer_callbacks.keys())

    simclr_trainer = L.Trainer(
        devices=ARGS.devices,
        max_epochs=ARGS.epochs,
        callbacks=list(simclr_trainer_callbacks.values()),
        deterministic=True,
        logger=simclr_loggers.values(),
        log_every_n_steps=10,
    )
    simclr_trainer.fit(model=simclr_module, datamodule=simclr_datamodule)

    best_simclr_model_ckpt = (
        simclr_trainer_callbacks["ModelCheckpoint"].best_model_path
        if "ModelCheckpoint" in simclr_trainer_callbacks
        else None
    )
    params["simclr_model_ckpt"] = best_simclr_model_ckpt

    # ===== Log all parameters for SimCLR =====

    for logger in simclr_loggers.values():
        logger.log_hyperparams(params)

    # ===== SimCLR generating features (h) after training  =====

    simclr_predictor = L.Trainer(
        devices=ARGS.devices,
        num_nodes=1,
        deterministic=True,
        logger=simclr_loggers.values(),
    )
    simclr_predictions = simclr_predictor.predict(
        model=simclr_module,
        datamodule=simclr_datamodule,
        return_predictions=True,
        ckpt_path=best_simclr_model_ckpt,
    )

    # ===== Create loggers for LogReg =====

    logreg_loggers = create_loggers(
        log_dir=ARGS.log_dir,
        experiment_name=ARGS.experiment_eval_name,
        run_name=ARGS.run_name,
        use_mlflow=ARGS.use_mlflow,
    )

    # ===== Creating Feature Dataset for Evaluation =====

    logreg_dataset: Dataset = FeatureDataset(predictions=simclr_predictions)
    print_dataset_summary(dataset=logreg_dataset, name="logreg_dataset")

    # ===== LogReg Training with Feature Dataset =====

    logreg_datamodule: L.LightningDataModule = LogisticRegressionDataModule(
        dataset=logreg_dataset,
        train_perc=ARGS.eval_train_perc,
        val_perc=ARGS.eval_val_perc,
        test_perc=ARGS.eval_test_perc,
    )

    logreg_module: L.LightningModule = LogisticRegressionModule(
        epochs=ARGS.eval_epochs,
        latent_dim=logreg_dataset.latent_dim,
        num_classes=simclr_datamodule.num_classes,
        learning_rate=ARGS.eval_learning_rate,
        weight_decay=ARGS.eval_weight_decay,
    )
    params |= get_parameter_stats(module=logreg_module, prefix="logreg_")

    logreg_trainer_callbacks = create_trainer_callbacks(
        patience=0, save_top_k=ARGS.save_top_k
    )

    params["logreg_trainer_callbacks"] = list(logreg_trainer_callbacks.keys())

    logreg_trainer = L.Trainer(
        devices=ARGS.devices,
        max_epochs=ARGS.eval_epochs,
        callbacks=list(logreg_trainer_callbacks.values()),
        deterministic=True,
        logger=logreg_loggers.values(),
        log_every_n_steps=10,
    )
    logreg_trainer.fit(model=logreg_module, datamodule=logreg_datamodule)

    best_logreg_model_ckpt = (
        logreg_trainer_callbacks["ModelCheckpoint"].best_model_path
        if "ModelCheckpoint" in logreg_trainer_callbacks
        else None
    )
    params["logreg_model_ckpt"] = best_logreg_model_ckpt

    # ===== Log all parameters for LogReg =====

    for logger in logreg_loggers.values():
        logger.log_hyperparams(params)

    # ===== LogReg Testing =====

    logreg_tester = L.Trainer(
        devices=1,
        num_nodes=1,
        deterministic=True,
        logger=logreg_loggers.values(),
    )
    logreg_tester.test(
        model=logreg_module,
        datamodule=logreg_datamodule,
        ckpt_path=best_logreg_model_ckpt,
    )
