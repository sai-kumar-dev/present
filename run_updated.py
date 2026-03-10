#!/usr/bin/env python3

import os
import argparse
import time
import json
import csv
from datetime import datetime

import torch
import numpy as np

from sampler import run as sampler_run
from eda_stream import BatchEDA, GlobalEDA, create_run_dir
from model import OceanHeatFluxPINN
from train import PINNTrainer
from core import logger, clear_memory

from viz import (
    generate_all_plots,
    plot_prediction_scatter,
    plot_residuals,
    plot_flux_map
)

MASK_FILE = "ocean_mask.npz"


# =========================================================
# METRICS LOGGER
# =========================================================

class MetricsLogger:

    def __init__(self, run_dir):

        self.metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.batch_file = os.path.join(self.metrics_dir, "batch_metrics.csv")
        self.epoch_file = os.path.join(self.metrics_dir, "epoch_metrics.csv")
        self.json_file = os.path.join(self.metrics_dir, "training_history.json")

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "data_loss": [],
            "physics_loss": []
        }

        logger.info("Initializing metrics logger")

        if not os.path.exists(self.batch_file):

            with open(self.batch_file, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "epoch",
                    "batch",
                    "total_loss",
                    "data_loss",
                    "physics_loss"
                ])

        if not os.path.exists(self.epoch_file):

            with open(self.epoch_file, "w", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "timestamp",
                    "epoch",
                    "train_loss",
                    "data_loss",
                    "physics_loss",
                    "val_loss"
                ])


    def log_batch(self, epoch, batch, total, data, physics):

        ts = datetime.utcnow().isoformat()

        with open(self.batch_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                ts,
                epoch,
                batch,
                total,
                data,
                physics
            ])


    def log_epoch(self, epoch, train, data, physics, val):

        ts = datetime.utcnow().isoformat()

        with open(self.epoch_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                ts,
                epoch,
                train,
                data,
                physics,
                val
            ])

        self.history["train_loss"].append(train)
        self.history["val_loss"].append(val)
        self.history["data_loss"].append(data)
        self.history["physics_loss"].append(physics)


    def save_json(self):

        logger.info("Saving training history JSON")

        with open(self.json_file, "w") as f:
            json.dump(self.history, f, indent=2)


# =========================================================
# MASK CREATION
# =========================================================

def ensure_ocean_mask():

    logger.info("Checking ocean mask")

    if os.path.exists(MASK_FILE):
        logger.info("Ocean mask exists")
        return

    logger.info("Ocean mask missing → building")

    import build
    build.main()

    logger.info("Ocean mask created")


# =========================================================
# VALIDATION DATASET
# =========================================================

def build_validation_dataset(args, samples=40000):

    logger.info("Building validation dataset")

    val_args = argparse.Namespace(
        sampler=args.sampler,
        batch_size=samples,
        batches=1,
        start_year=args.val_start_year,
        end_year=args.val_end_year,
        seed=999
    )

    engine = sampler_run(val_args)

    batch = next(engine)

    X_val = batch["X"].numpy()
    Y_val = batch["Y"].numpy()

    logger.info(f"Validation dataset ready: {len(X_val)} samples")

    return X_val, Y_val


# =========================================================
# TEST DATASET
# =========================================================

def build_test_dataset(args, samples=60000):

    logger.info("Building test dataset")

    test_args = argparse.Namespace(
        sampler=args.sampler,
        batch_size=samples,
        batches=1,
        start_year=args.test_start_year,
        end_year=args.test_end_year,
        seed=1234
    )

    engine = sampler_run(test_args)

    batch = next(engine)

    X_test = batch["X"].numpy()
    Y_test = batch["Y"].numpy()

    lat = batch["lat"].numpy()
    lon = batch["lon"].numpy()

    logger.info(f"Test dataset ready: {len(X_test)} samples")

    return X_test, Y_test, lat, lon


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline(args):

    logger.info("===================================")
    logger.info("ERA5 HEAT FLUX PINN PIPELINE START")
    logger.info("===================================")

    start_time = time.time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_ocean_mask()

    logger.info("Creating run directory")

    run_dir = create_run_dir()

    logger.info(f"Run directory: {run_dir}")

    metrics_logger = MetricsLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Compute device: {device}")


    logger.info("Initializing model")

    model = OceanHeatFluxPINN(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        num_layers=args.num_layers
    )

    trainer = PINNTrainer(
        model,
        device=device,
        lr=args.learning_rate,
        lambda_physics=args.lambda_physics,
        run_dir=run_dir
    )

    logger.info("Model ready")


    global_eda = GlobalEDA(run_dir)


    X_val, Y_val = build_validation_dataset(args)

    X_test, Y_test, lat_test, lon_test = build_test_dataset(args)


    logger.info("Starting training loop")

    for epoch in range(args.epochs):

        logger.info("-----------------------------------")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info("-----------------------------------")

        sampler_args = argparse.Namespace(
            sampler=args.sampler,
            batch_size=args.batch_size,
            batches=args.batches,
            start_year=args.train_start_year,
            end_year=args.train_end_year,
            seed=args.seed + epoch
        )

        engine = sampler_run(sampler_args)

        epoch_total=[]
        epoch_data=[]
        epoch_phys=[]

        for batch in engine:

            batch_id = batch.get("batch",0)+1

            logger.info(f"Processing batch {batch_id}")

            losses = trainer.train_batch(
                batch["X"],
                batch["Y"],
                epoch
            )

            metrics_logger.log_batch(
                epoch,
                batch_id,
                losses["total"],
                losses["data"],
                losses["physics"]
            )

            epoch_total.append(losses["total"])
            epoch_data.append(losses["data"])
            epoch_phys.append(losses["physics"])

            logger.info(
                f"Loss total={losses['total']:.6f} "
                f"data={losses['data']:.6f} "
                f"physics={losses['physics']:.6f}"
            )

            clear_memory()

        epoch_loss=np.mean(epoch_total)
        epoch_data_loss=np.mean(epoch_data)
        epoch_phys_loss=np.mean(epoch_phys)

        logger.info(
            f"Epoch summary → "
            f"train={epoch_loss:.6f} "
            f"data={epoch_data_loss:.6f} "
            f"physics={epoch_phys_loss:.6f}"
        )

        val_loss = trainer.validate(X_val,Y_val)

        logger.info(f"Validation loss: {val_loss:.6f}")

        metrics_logger.log_epoch(
            epoch,
            epoch_loss,
            epoch_data_loss,
            epoch_phys_loss,
            val_loss
        )

        trainer.save_best(val_loss)

        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch,val_loss)


    logger.info("Training complete")

    metrics_logger.save_json()

    logger.info("Running final evaluation")

    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        pred = model(X_t).cpu().numpy()

    logger.info("Generating evaluation plots")

    plot_prediction_scatter(Y_test, pred, run_dir)

    plot_residuals(Y_test, pred, run_dir)

    plot_flux_map(lat_test, lon_test, pred[:,0], run_dir, name="sshf_map")

    plot_flux_map(lat_test, lon_test, pred[:,1], run_dir, name="slhf_map")

    generate_all_plots(run_dir)

    runtime = time.time() - start_time

    logger.info(f"Pipeline finished in {runtime/60:.2f} minutes")

    logger.info("===================================")
    logger.info("PIPELINE COMPLETE")
    logger.info("===================================")


# =========================================================
# ARGUMENTS
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--sampler",default="hybrid")

    parser.add_argument("--batch_size",type=int,default=15000)
    parser.add_argument("--batches",type=int,default=10)

    parser.add_argument("--epochs",type=int,default=40)

    parser.add_argument("--train_start_year",type=int,default=1990)
    parser.add_argument("--train_end_year",type=int,default=2015)

    parser.add_argument("--val_start_year",type=int,default=2016)
    parser.add_argument("--val_end_year",type=int,default=2018)

    parser.add_argument("--test_start_year",type=int,default=2019)
    parser.add_argument("--test_end_year",type=int,default=2020)

    parser.add_argument("--hidden_dim",type=int,default=256)
    parser.add_argument("--num_layers",type=int,default=5)

    parser.add_argument("--learning_rate",type=float,default=1e-3)

    parser.add_argument("--lambda_physics",type=float,default=0.1)

    parser.add_argument("--seed",type=int,default=42)

    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()