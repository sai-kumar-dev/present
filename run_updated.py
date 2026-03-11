#!/usr/bin/env python3

import os
import argparse
import time
import json
from datetime import timezone, datetime
from torch.utils.data import DataLoader
from dataset import SamplerDataset

import torch
import numpy as np

from sampler import run as sampler_run
from eda_stream import BatchEDA, GlobalEDA, create_run_dir
from model import OceanHeatFluxPINN
from train import PINNTrainer
from core import create_logger, clear_memory



from viz import (
    generate_all_plots,
    plot_prediction_scatter,
    plot_residuals,
    plot_flux_map,
    plot_all_predictions
)

MASK_FILE = "ocean_mask.npz"


# =========================================================
# SAFE NUMPY CONVERSION
# =========================================================

def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return x


# =========================================================
# METRICS LOGGER
# =========================================================

class MetricsLogger:

    def __init__(self, run_dir):

        self.run_dir = run_dir

        self.metrics_dir = os.path.join(run_dir, "metrics")
        self.epochs_dir = os.path.join(run_dir, "epochs")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.epochs_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "data_loss": [],
            "physics_loss": []
        }

        logger.info("Metrics logger initialized")


    def start_epoch(self, epoch):

        self.epoch_dir = os.path.join(
            self.epochs_dir,
            f"epoch_{epoch:03d}"
        )

        self.batch_dir = os.path.join(self.epoch_dir, "batches")

        os.makedirs(self.batch_dir, exist_ok=True)

        logger.info(f"Epoch logging directory created: {self.epoch_dir}")

    def log_batch(self, epoch, batch, total, data, physics):

        # ensure epoch directory exists
        epoch_dir = os.path.join(
            self.epochs_dir,
            f"epoch_{epoch:03d}"
        )

        batch_dir = os.path.join(epoch_dir, "batches")

        os.makedirs(batch_dir, exist_ok=True)

        batch_file = os.path.join(
            batch_dir,
            f"batch_{batch:04d}.json"
        )

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "batch": batch,
            "total_loss": float(total),
            "data_loss": float(data),
            "physics_loss": float(physics)
        }

        with open(batch_file, "w") as f:
            json.dump(payload, f, indent=2)

    def log_epoch(self, epoch, train, data, physics, val):

        epoch_dir = os.path.join(
            self.epochs_dir,
            f"epoch_{epoch:03d}"
        )

        os.makedirs(epoch_dir, exist_ok=True)
        self.history["train_loss"].append(float(train))
        self.history["data_loss"].append(float(data))
        self.history["physics_loss"].append(float(physics))
        self.history["val_loss"].append(float(val))
        summary_file = os.path.join(epoch_dir, "summary.json")

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "train_loss": float(train),
            "data_loss": float(data),
            "physics_loss": float(physics),
            "val_loss": float(val)
        }

        with open(summary_file, "w") as f:
            json.dump(payload, f, indent=2)

    def save_history(self):

        history_file = os.path.join(
            self.metrics_dir,
            "training_history.json"
        )

        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info("Training history saved")


# =========================================================
# MASK CHECK
# =========================================================

def ensure_ocean_mask():

    logger.info("Checking ocean mask")

    if os.path.exists(MASK_FILE):

        logger.info("Ocean mask exists")

        return

    logger.info("Ocean mask missing → building")

    import build
    build.main()

    logger.info("Ocean mask built")


# =========================================================
# DATASET BUILDERS
# =========================================================

def build_validation_dataset(args):

    logger.info("Generating validation dataset")

    val_args = argparse.Namespace(
        sampler=args.sampler,
        batch_size=args.val_size,
        batches=1,
        start_year=args.val_start_year,
        end_year=args.val_end_year,
        seed=999
    )

    engine = sampler_run(val_args)

    batch = next(engine)

    X_val = to_numpy(batch["X"])
    Y_val = to_numpy(batch["Y"])

    logger.info(f"Validation samples: {len(X_val)}")

    return X_val, Y_val


def build_test_dataset(args):

    logger.info("Generating test dataset")

    test_args = argparse.Namespace(
        sampler=args.sampler,
        batch_size=args.test_size,
        batches=1,
        start_year=args.test_start_year,
        end_year=args.test_end_year,
        seed=1234
    )

    engine = sampler_run(test_args)

    batch = next(engine)

    X_test = to_numpy(batch["X"])
    Y_test = to_numpy(batch["Y"])
    lat = to_numpy(batch["lat"])
    lon = to_numpy(batch["lon"])

    logger.info(f"Test samples: {len(X_test)}")

    return X_test, Y_test, lat, lon


# =========================================================
# PIPELINE
# =========================================================

def run_pipeline(args):

    logger.info("================================================")
    logger.info("ERA5 HEAT FLUX PINN TRAINING PIPELINE START")
    logger.info("================================================")

    start_time = time.time()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_ocean_mask()

    logger.info("Creating run directory")

    run_dir = create_run_dir()

    # initialize logger AFTER run dir exists
    from core import create_logger
    logger = create_logger(run_dir)

    logger.info(f"Run directory: {run_dir}")

    args_file = os.path.join(run_dir, "args.json")

    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Run arguments saved")

    metrics_logger = MetricsLogger(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")

    logger.info("Initializing model")

    model = OceanHeatFluxPINN(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )

    trainer = PINNTrainer(
        model,
        device=device,
        lr=args.learning_rate,
        lambda_physics=args.lambda_physics,
        run_dir=run_dir
    )

    logger.info("Model initialized")

    global_eda = GlobalEDA(run_dir)

    logger.info("Building validation dataset")

    X_val, Y_val = build_validation_dataset(args)

    logger.info("Building test dataset")

    X_test, Y_test, lat_test, lon_test = build_test_dataset(args)

    logger.info(f"Validation dataset size: {len(X_val)}")
    logger.info(f"Test dataset size: {len(X_test)}")

    logger.info("Starting training loop")

    for epoch in range(args.epochs):

        logger.info("------------------------------------------------")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info("------------------------------------------------")

        metrics_logger.start_epoch(epoch)

        sampler_args = argparse.Namespace(
            sampler=args.sampler,
            batch_size=args.batch_size,
            batches=args.batches,
            start_year=args.train_start_year,
            end_year=args.train_end_year,
            seed=args.seed + epoch
        )

        dataset = SamplerDataset(sampler_args)

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=2
        )

        epoch_total = []
        epoch_data = []
        epoch_phys = []

        batch_id = 0

        for batch in loader:

            batch_id += 1

            X = batch["X"]
            Y = batch["Y"]

            logger.info(f"Training batch {batch_id}")

            losses = trainer.train_batch(
                X,
                Y,
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

            # -------- EDA --------

            if batch_id % args.eda_interval == 0:

                logger.info(
                    f"Running EDA for epoch {epoch} batch {batch_id}"
                )

                try:

                    batch_eda = BatchEDA(batch, run_dir)

                    batch_eda.run_all()

                    global_eda.add_batch(
                        batch_eda.lat,
                        batch_eda.lon,
                        batch_eda.flux,
                        batch_eda.wind,
                        batch_eda.time,
                        batch_eda.X_raw,
                        batch_eda.Y_raw
                    )

                except Exception as e:

                    logger.warning(f"EDA failed: {e}")

            clear_memory()

        # ----- epoch statistics -----

        epoch_loss = np.mean(epoch_total) if epoch_total else 0
        epoch_data_loss = np.mean(epoch_data) if epoch_data else 0
        epoch_phys_loss = np.mean(epoch_phys) if epoch_phys else 0

        val_loss = trainer.validate(X_val, Y_val)

        logger.info(
            f"Epoch summary → "
            f"train={epoch_loss:.6f} "
            f"data={epoch_data_loss:.6f} "
            f"physics={epoch_phys_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        metrics_logger.log_epoch(
            epoch,
            epoch_loss,
            epoch_data_loss,
            epoch_phys_loss,
            val_loss
        )

        trainer.save_best(val_loss)

        if epoch % 5 == 0:
            trainer.save_checkpoint(epoch, val_loss)

    # ---------- TRAINING COMPLETE ----------

    logger.info("Training finished")

    logger.info("Finalizing global EDA")

    try:
        global_eda.finalize()
    except Exception as e:
        logger.warning(f"Global EDA failed: {e}")

    trainer.save_final()

    metrics_logger.save_history()

    runtime = time.time() - start_time

    logger.info(f"Training runtime: {runtime/60:.2f} minutes")

    # ---------- MODEL EVALUATION ----------

    logger.info("Running final model evaluation")

    model = trainer.model
    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        pred = model(X_t).cpu().numpy()

    logger.info("Generating evaluation plots")

    plot_prediction_scatter(Y_test, pred, run_dir)
    plot_residuals(Y_test, pred, run_dir)
    plot_flux_map(lat_test, lon_test, pred[:,0], run_dir, name="sshf_map")
    plot_flux_map(lat_test, lon_test, pred[:,1], run_dir, name="slhf_map")
    plot_all_predictions(Y_test, pred, run_dir)

    generate_all_plots(run_dir)

    logger.info("================================================")
    logger.info("PIPELINE COMPLETE")
    logger.info("================================================")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
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

    parser.add_argument("--val_size",type=int,default=40000)
    parser.add_argument("--test_size",type=int,default=60000)

    parser.add_argument("--hidden_dim",type=int,default=256)
    parser.add_argument("--num_layers",type=int,default=5)

    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--lambda_physics",type=float,default=0.1)

    parser.add_argument("--seed",type=int,default=42)

    parser.add_argument(
        "--eda_interval",
        type=int,
        default=3,
        help="Run batch EDA every N batches"
    )

    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()