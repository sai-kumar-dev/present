#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from physics import physics_loss


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

        self.train_history = []
        self.val_history = []

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

        λ = self.physics_weight(epoch)

        phys_scaled = phys_loss / (phys_loss.detach() + 1e-6)
        total_loss = data_loss + λ * phys_scaled
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
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

        self.val_history.append(val_loss)

        return val_loss

    # ------------------------------------------------
    # CHECKPOINT SAVE
    # ------------------------------------------------

    def save_checkpoint(self, epoch, val_loss=None):

        if not self.run_dir:
            return

        ckpt_path = os.path.join(
            self.run_dir,
            "checkpoints",
            f"epoch_{epoch}.pt"
        )

        torch.save({

            "epoch": epoch,

            "model_state": self.model.state_dict(),

            "optimizer_state": self.optimizer.state_dict(),

            "val_loss": val_loss

        }, ckpt_path)

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
    # METRICS LOGGING
    # ------------------------------------------------

    def log_metrics(self):

        if not self.run_dir:
            return

        path = os.path.join(
            self.run_dir,
            "metrics",
            "training_history.json"
        )

        data = {

            "train_loss": self.train_history,

            "val_loss": self.val_history
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------
    # EPOCH TRAINING LOOP
    # ------------------------------------------------

    def train_epoch(self, dataloader, epoch):

        epoch_loss = 0

        for X, Y in dataloader:

            losses = self.train_batch(X, Y, epoch)

            epoch_total += losses["total"]
            epoch_data += losses["data"]
            epoch_phys += losses["physics"]

        epoch_loss /= len(dataloader)
        epoch_total /= len(dataloader)
        epoch_data /= len(dataloader)
        epoch_phys /= len(dataloader)
        self.train_history.append(epoch_total)
        self.data_history.append(epoch_data)
        self.physics_history.append(epoch_phys)
        self.train_history.append(epoch_loss)

        return epoch_loss
