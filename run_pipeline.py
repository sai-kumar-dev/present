#!/usr/bin/env python3

import os
import argparse
import time
import torch
import numpy as np

from sampler import run as sampler_run
from eda_stream import BatchEDA, GlobalEDA, create_run_dir
from model import OceanHeatFluxPINN
from train import PINNTrainer
from core import logger, clear_memory
from viz import generate_all_plots


MASK_FILE = "ocean_mask.npz"


def ensure_ocean_mask():

    if os.path.exists(MASK_FILE):
        logger.info("Ocean mask exists")
        return

    logger.info("Ocean mask missing → building")

    import build
    build.main()


def build_validation_dataset(args, samples=40000):

    logger.info("Generating validation dataset")

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

    logger.info(f"Validation samples: {len(X_val)}")

    return X_val, Y_val


def run_pipeline(args):

    start_time = time.time()

    logger.info("=== ERA5 FLUX PINN PIPELINE START ===")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_ocean_mask()

    run_dir = create_run_dir()

    logger.info(f"Run directory: {run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")


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

    logger.info("Model initialized")


    global_eda = GlobalEDA(run_dir)


    X_val, Y_val = build_validation_dataset(args)


    for epoch in range(args.epochs):

        logger.info(f"Epoch {epoch+1}/{args.epochs}")

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

            logger.info(f"Training batch {batch_id}")

            losses = trainer.train_batch(
                batch["X"],
                batch["Y"],
                epoch
            )

            epoch_total.append(losses["total"])
            epoch_data.append(losses["data"])
            epoch_phys.append(losses["physics"])

            logger.info(
                f"loss={losses['total']:.6f} "
                f"data={losses['data']:.6f} "
                f"physics={losses['physics']:.6f}"
            )


            if batch_id % args.eda_interval == 0:

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


        epoch_loss=np.mean(epoch_total)
        epoch_data_loss=np.mean(epoch_data)
        epoch_phys_loss=np.mean(epoch_phys)

        trainer.train_history.append(epoch_loss)

        logger.info(
            f"Epoch summary → "
            f"train={epoch_loss:.6f} "
            f"data={epoch_data_loss:.6f} "
            f"physics={epoch_phys_loss:.6f}"
        )


        val_loss = trainer.validate(X_val,Y_val)

        logger.info(f"Validation loss {val_loss:.6f}")

        trainer.save_best(val_loss)

        if epoch % 5 == 0:

            trainer.save_checkpoint(epoch,val_loss)


    logger.info("Finalizing global EDA")

    global_eda.finalize()


    trainer.save_final()

    trainer.log_metrics()


    runtime = time.time() - start_time

    logger.info(f"Training finished in {runtime/60:.2f} minutes")

    generate_all_plots(run_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampler",
        choices=[
            "random",
            "temporal",
            "seasonal",
            "spatial",
            "flux",
            "hybrid"
        ],
        default="hybrid"
    )

    parser.add_argument("--batch_size",type=int,default=15000)

    parser.add_argument("--batches",type=int,default=10)

    parser.add_argument("--epochs",type=int,default=40)

    parser.add_argument("--train_start_year",type=int,default=1990)

    parser.add_argument("--train_end_year",type=int,default=2015)

    parser.add_argument("--val_start_year",type=int,default=2016)

    parser.add_argument("--val_end_year",type=int,default=2020)

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