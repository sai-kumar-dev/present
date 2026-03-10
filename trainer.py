#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from physics import physics_loss
from core import logger, clear_memory


class PINNTrainer:

    def __init__(
        self,
        model,
        device="cpu",
        lr=1e-3,
        lambda_physics=0.1,
        run_dir=None
    ):

        self.model = model.to(device)
        self.device = device
        self.lambda_physics = lambda_physics

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.data_loss_fn = nn.MSELoss()

        self.best_val = np.inf
        self.run_dir = run_dir

        if run_dir:
            os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "model"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)

        self.history = {
            "epoch": [],
            "train_total": [],
            "train_data": [],
            "train_physics": [],
            "val_loss": [],
            "learning_rate": []
        }


    # ------------------------------------------------
    # PHYSICS CURRICULUM
    # ------------------------------------------------

    def physics_weight(self, epoch):

        if epoch < 10:
            return 0.0

        if epoch < 30:
            return self.lambda_physics * 0.5

        return self.lambda_physics


    # ------------------------------------------------
    # TRAIN SINGLE BATCH
    # ------------------------------------------------

    def train_batch(self, X, Y, epoch=0):

        self.model.train()

        X = X.to(self.device)
        Y = Y.to(self.device)

        self.optimizer.zero_grad()

        Y_pred = self.model(X)

        data_loss = self.data_loss_fn(Y_pred, Y)

        phys_loss = physics_loss(X, Y_pred)

        lam = self.physics_weight(epoch)

        total_loss = data_loss + lam * phys_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            1.0
        )

        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "data": data_loss.item(),
            "physics": phys_loss.item()
        }


    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------

    def validate(self, X_val, Y_val):

        self.model.eval()

        with torch.no_grad():

            X_val = torch.tensor(
                X_val,
                dtype=torch.float32
            ).to(self.device)

            Y_val = torch.tensor(
                Y_val,
                dtype=torch.float32
            ).to(self.device)

            Y_pred = self.model(X_val)

            loss = self.data_loss_fn(Y_pred, Y_val)

        val_loss = loss.item()

        self.scheduler.step(val_loss)

        return val_loss


    # ------------------------------------------------
    # EPOCH TRAINING
    # ------------------------------------------------

    def train_epoch(self, dataloader, epoch):

        total_loss = 0.0
        data_loss = 0.0
        physics_loss_val = 0.0

        batches = 0

        for X, Y in dataloader:

            losses = self.train_batch(X, Y, epoch)

            total_loss += losses["total"]
            data_loss += losses["data"]
            physics_loss_val += losses["physics"]

            batches += 1

        total_loss /= batches
        data_loss /= batches
        physics_loss_val /= batches

        return total_loss, data_loss, physics_loss_val


    # ------------------------------------------------
    # LOG EPOCH METRICS
    # ------------------------------------------------

    def log_epoch(
        self,
        epoch,
        train_total,
        train_data,
        train_physics,
        val_loss
    ):

        lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch} | "
            f"train_total={train_total:.6f} | "
            f"train_data={train_data:.6f} | "
            f"train_physics={train_physics:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={lr:.2e}"
        )

        self.history["epoch"].append(epoch)
        self.history["train_total"].append(train_total)
        self.history["train_data"].append(train_data)
        self.history["train_physics"].append(train_physics)
        self.history["val_loss"].append(val_loss)
        self.history["learning_rate"].append(lr)


    # ------------------------------------------------
    # CHECKPOINT SAVE
    # ------------------------------------------------

    def save_checkpoint(self, epoch, val_loss):

        if not self.run_dir:
            return

        path = os.path.join(
            self.run_dir,
            "checkpoints",
            f"epoch_{epoch:03d}_val{val_loss:.4f}.pt"
        )

        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss
        }, path)


    # ------------------------------------------------
    # BEST MODEL SAVE
    # ------------------------------------------------

    def save_best(self, val_loss):

        if not self.run_dir:
            return

        if val_loss < self.best_val:

            self.best_val = val_loss

            path = os.path.join(
                self.run_dir,
                "model",
                "best_model.pt"
            )

            torch.save(
                self.model.state_dict(),
                path
            )


    # ------------------------------------------------
    # FINAL MODEL SAVE
    # ------------------------------------------------

    def save_final(self):

        if not self.run_dir:
            return

        path = os.path.join(
            self.run_dir,
            "model",
            "final_model.pt"
        )

        torch.save(
            self.model.state_dict(),
            path
        )


    # ------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------

    def save_metrics(self):

        if not self.run_dir:
            return

        path = os.path.join(
            self.run_dir,
            "metrics",
            "training_history.json"
        )

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


    # ------------------------------------------------
    # CLEANUP
    # ------------------------------------------------

    def cleanup(self):

        clear_memory()