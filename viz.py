#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from core import logger

sns.set_style("whitegrid")


# ------------------------------------------------
# DIRECTORY HELPERS
# ------------------------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------
# LOAD METRICS
# ------------------------------------------------

def load_metrics(run_dir):

    path = os.path.join(run_dir, "metrics", "training_history.json")

    if not os.path.exists(path):
        logger.warning("Training metrics not found")
        return None

    with open(path) as f:
        data = json.load(f)

    return data


# ------------------------------------------------
# LOSS CURVES
# ------------------------------------------------

def plot_loss_curves(run_dir):

    data = load_metrics(run_dir)

    if data is None:
        return

    train = np.array(data["train_loss"])
    val = np.array(data["val_loss"])

    fig_dir = os.path.join(run_dir, "plots")
    ensure_dir(fig_dir)

    plt.figure(figsize=(8,5))

    plt.plot(train, label="train")
    plt.plot(val, label="validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.savefig(os.path.join(fig_dir,"loss_curve.png"))
    plt.close()

    logger.info("Saved loss curve")


# ------------------------------------------------
# PHYSICS vs DATA LOSS
# ------------------------------------------------

def plot_all_predictions(y_true, y_pred, run_dir):

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = os.path.join(run_dir, "plots")
    os.makedirs(fig_dir, exist_ok=True)

    true_all = y_true.flatten()
    pred_all = y_pred.flatten()

    plt.figure(figsize=(6,6))

    plt.scatter(true_all, pred_all, s=4, alpha=0.3)

    min_val = min(true_all.min(), pred_all.min())
    max_val = max(true_all.max(), pred_all.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Flux")
    plt.ylabel("Predicted Flux")

    plt.title("All Predictions vs Actual")

    plt.savefig(os.path.join(fig_dir, "all_predictions_vs_actual.png"))

    plt.close()

def plot_physics_data_loss(run_dir):

    path = os.path.join(run_dir, "metrics", "training_history.json")

    if not os.path.exists(path):
        return

    with open(path) as f:
        data = json.load(f)

    if "physics_loss" not in data:
        logger.warning("Physics loss history missing")
        return

    physics = np.array(data["physics_loss"])
    data_loss = np.array(data["data_loss"])

    fig_dir = os.path.join(run_dir, "plots")
    ensure_dir(fig_dir)

    plt.figure(figsize=(8,5))

    plt.plot(data_loss,label="data")
    plt.plot(physics,label="physics")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Data vs Physics Loss")

    plt.legend()

    plt.savefig(os.path.join(fig_dir,"physics_vs_data.png"))
    plt.close()

    logger.info("Saved physics vs data loss plot")


# ------------------------------------------------
# PREDICTION SCATTER
# ------------------------------------------------

def plot_prediction_scatter(y_true, y_pred, run_dir):

    fig_dir = os.path.join(run_dir,"plots")
    ensure_dir(fig_dir)

    sshf_true = y_true[:,0]
    slhf_true = y_true[:,1]

    sshf_pred = y_pred[:,0]
    slhf_pred = y_pred[:,1]


    plt.figure(figsize=(6,6))

    plt.scatter(sshf_true, sshf_pred, s=4, alpha=0.3)

    plt.xlabel("True SSHF")
    plt.ylabel("Predicted SSHF")

    plt.title("Sensible Heat Flux Prediction")

    plt.savefig(os.path.join(fig_dir,"sshf_prediction_scatter.png"))
    plt.close()


    plt.figure(figsize=(6,6))

    plt.scatter(slhf_true, slhf_pred, s=4, alpha=0.3)

    plt.xlabel("True SLHF")
    plt.ylabel("Predicted SLHF")

    plt.title("Latent Heat Flux Prediction")

    plt.savefig(os.path.join(fig_dir,"slhf_prediction_scatter.png"))
    plt.close()

    logger.info("Saved prediction scatter plots")


# ------------------------------------------------
# RESIDUAL DISTRIBUTION
# ------------------------------------------------

def plot_residuals(y_true, y_pred, run_dir):

    fig_dir = os.path.join(run_dir,"plots")
    ensure_dir(fig_dir)

    residual = y_pred - y_true

    plt.figure()

    sns.histplot(residual[:,0], bins=100)

    plt.title("SSHF Residual Distribution")

    plt.savefig(os.path.join(fig_dir,"sshf_residuals.png"))
    plt.close()


    plt.figure()

    sns.histplot(residual[:,1], bins=100)

    plt.title("SLHF Residual Distribution")

    plt.savefig(os.path.join(fig_dir,"slhf_residuals.png"))
    plt.close()

    logger.info("Saved residual plots")


# ------------------------------------------------
# SPATIAL PREDICTION MAP
# ------------------------------------------------

def plot_flux_map(lat, lon, flux, run_dir, name="flux_map"):

    fig_dir = os.path.join(run_dir,"plots")
    ensure_dir(fig_dir)

    plt.figure(figsize=(10,5))

    sc = plt.scatter(
        lon,
        lat,
        c=flux,
        s=3,
        cmap="inferno",
        alpha=0.6
    )

    plt.colorbar(sc,label="Flux")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title("Predicted Flux Map")

    plt.savefig(os.path.join(fig_dir,f"{name}.png"))
    plt.close()

    logger.info("Saved spatial flux map")


# ------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------

def evaluate_model(model, X, Y, device, run_dir):

    logger.info("Running model evaluation")

    model.eval()

    with torch.no_grad():

        X_t = torch.tensor(X,dtype=torch.float32).to(device)

        pred = model(X_t).cpu().numpy()

    plot_prediction_scatter(Y,pred,run_dir)

    plot_residuals(Y,pred,run_dir)

    logger.info("Evaluation complete")


# ------------------------------------------------
# MASTER VISUALIZATION
# ------------------------------------------------

def generate_all_plots(run_dir):

    logger.info("Generating experiment plots")

    plot_loss_curves(run_dir)

    plot_physics_data_loss(run_dir)
    

    logger.info("All plots generated")