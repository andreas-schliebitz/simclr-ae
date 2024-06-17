#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simclr_ae.utils import seed_rng

seed_rng()

import torch.nn as nn
import lightning as L

from pprint import pprint
from simclr_ae.models.ae.model import AE
from simclr_ae.models.ae.module import AEModule
from simclr_ae.experiments.args import get_parser
from simclr_ae.experiments.ae.datamodule import AEDataModule
from simclr_ae.experiments.ae.args import add_args as add_ae_args
from simclr_ae.utils import create_loggers, create_trainer_callbacks
from simclr_ae.experiments.args import add_args as add_experiments_args

if __name__ == "__main__":
    arg_parser = get_parser()
    arg_parser = add_experiments_args(arg_parser)
    arg_parser = add_ae_args(arg_parser)
    ARGS = arg_parser.parse_args()

    params = vars(ARGS)
    pprint(params)

    loggers = create_loggers(
        log_dir=ARGS.log_dir,
        experiment_name=ARGS.experiment_name,
        run_name=ARGS.run_name,
        use_mlflow=ARGS.use_mlflow,
    )

    ae_datamodule: L.LightningDataModule = AEDataModule(
        dataset_dir=ARGS.dataset_dir,
        dataset_name=ARGS.dataset_name,
        train_perc=ARGS.train_perc,
        val_perc=ARGS.val_perc,
        test_perc=ARGS.test_perc,
        img_size=ARGS.img_size,
        batch_size=ARGS.batch_size,
        standardize=ARGS.standardize,
        num_workers=ARGS.num_workers,
        logger=loggers.values(),
    )

    input_dim = (ARGS.num_channels, ARGS.img_size, ARGS.img_size)
    params["input_dimensions"] = input_dim

    ae_model: nn.Module = AE(input_dim=input_dim, latent_dim=ARGS.latent_dimensions)

    ae_module: L.LightningModule = AEModule(
        model=ae_model,
        epochs=ARGS.epochs,
        learning_rate=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
    )

    trainer_callbacks = create_trainer_callbacks(
        patience=ARGS.patience, save_top_k=ARGS.save_top_k
    )
    params["trainer_callbacks"] = list(trainer_callbacks.keys())

    trainer = L.Trainer(
        devices=ARGS.devices,
        max_epochs=ARGS.epochs,
        callbacks=list(trainer_callbacks.values()),
        deterministic=True,
        logger=loggers.values(),
        log_every_n_steps=10,
    )
    trainer.fit(model=ae_module, datamodule=ae_datamodule)

    best_model_ckpt = (
        trainer_callbacks["ModelCheckpoint"].best_model_path
        if "ModelCheckpoint" in trainer_callbacks
        else None
    )
    params["model_ckpt"] = best_model_ckpt

    for logger in loggers.values():
        logger.log_hyperparams(params)

    trainer = L.Trainer(
        num_nodes=1,
        devices=1,
        deterministic=True,
        logger=loggers.values(),
        limit_predict_batches=1,
    )

    trainer.test(model=ae_module, datamodule=ae_datamodule, ckpt_path=best_model_ckpt)

    predictions = trainer.predict(
        model=ae_module,
        datamodule=ae_datamodule,
        return_predictions=True,
        ckpt_path=best_model_ckpt,
    )
