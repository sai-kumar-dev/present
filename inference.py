#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from core import logger
from model import OceanHeatFluxPINN


sns.set_style("whitegrid")


INPUT_SCALE = np.array([20,20,320,320,105000,1.5,310],dtype=np.float32)
TARGET_SCALE = np.array([500,500],dtype=np.float32)


# ------------------------------------------------
# DIRECTORY
# ------------------------------------------------

def ensure_dir(path):
    os.makedirs(path,exist_ok=True)


# ------------------------------------------------
# MODEL LOADING
# ------------------------------------------------

def load_model(run_dir,device):

    model_path=os.path.join(run_dir,"model","best_model.pt")

    if not os.path.exists(model_path):
        model_path=os.path.join(run_dir,"model","final_model.pt")

    if not os.path.exists(model_path):
        raise RuntimeError("No trained model found")

    model=OceanHeatFluxPINN()

    state=torch.load(model_path,map_location=device)

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    logger.info(f"Loaded model: {model_path}")

    return model


# ------------------------------------------------
# INPUT NORMALIZATION
# ------------------------------------------------

def normalize_inputs(X):

    X=np.asarray(X,dtype=np.float32)

    if X.shape[1]!=7:
        raise ValueError("Input must have 7 features")

    return X/INPUT_SCALE


# ------------------------------------------------
# OUTPUT DENORMALIZATION
# ------------------------------------------------

def denormalize_outputs(Y):

    return Y*TARGET_SCALE


# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

def predict(model,X_norm,device):

    with torch.no_grad():

        X_t=torch.tensor(X_norm,dtype=torch.float32).to(device)

        pred=model(X_t)

    return pred.cpu().numpy()


# ------------------------------------------------
# METRICS
# ------------------------------------------------

def compute_metrics(y_true,y_pred):

    err=y_pred-y_true

    mae=np.mean(np.abs(err),axis=0)
    rmse=np.sqrt(np.mean(err**2,axis=0))

    return {
        "mae_sshf":float(mae[0]),
        "mae_slhf":float(mae[1]),
        "rmse_sshf":float(rmse[0]),
        "rmse_slhf":float(rmse[1])
    }


# ------------------------------------------------
# PLOTS
# ------------------------------------------------

def plot_scatter(y_true,y_pred,plot_dir):

    plt.figure(figsize=(6,6))

    plt.scatter(y_true[:,0],y_pred[:,0],s=4,alpha=0.3)

    lim=[y_true[:,0].min(),y_true[:,0].max()]
    plt.plot(lim,lim,"r--")

    plt.xlabel("True SSHF")
    plt.ylabel("Predicted SSHF")

    plt.savefig(os.path.join(plot_dir,"sshf_scatter.png"))
    plt.close()


    plt.figure(figsize=(6,6))

    plt.scatter(y_true[:,1],y_pred[:,1],s=4,alpha=0.3)

    lim=[y_true[:,1].min(),y_true[:,1].max()]
    plt.plot(lim,lim,"r--")

    plt.xlabel("True SLHF")
    plt.ylabel("Predicted SLHF")

    plt.savefig(os.path.join(plot_dir,"slhf_scatter.png"))
    plt.close()


def plot_error_distribution(y_true,y_pred,plot_dir):

    err=y_pred-y_true

    plt.figure()

    sns.histplot(err[:,0],bins=100)
    plt.title("SSHF Error")

    plt.savefig(os.path.join(plot_dir,"sshf_error_dist.png"))
    plt.close()


    plt.figure()

    sns.histplot(err[:,1],bins=100)
    plt.title("SLHF Error")

    plt.savefig(os.path.join(plot_dir,"slhf_error_dist.png"))
    plt.close()


def plot_spatial_error(lat,lon,y_true,y_pred,plot_dir):

    err=np.sum(np.abs(y_pred-y_true),axis=1)

    plt.figure(figsize=(10,5))

    sc=plt.scatter(
        lon,
        lat,
        c=err,
        s=3,
        cmap="inferno"
    )

    plt.colorbar(sc,label="Flux Error")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(os.path.join(plot_dir,"spatial_error.png"))
    plt.close()


# ------------------------------------------------
# MAIN INFERENCE
# ------------------------------------------------

def run_inference(run_dir, X_raw, Y_raw=None, lat=None, lon=None, device="cpu"):

    logger.info("Starting inference")

    plot_dir=os.path.join(run_dir,"inference")

    ensure_dir(plot_dir)

    model=load_model(run_dir,device)

    X_norm=normalize_inputs(X_raw)

    pred_norm=predict(model,X_norm,device)

    pred=denormalize_outputs(pred_norm)

    if Y_raw is not None:

        metrics=compute_metrics(Y_raw,pred)

        logger.info(f"MAE SSHF {metrics['mae_sshf']:.3f}")
        logger.info(f"MAE SLHF {metrics['mae_slhf']:.3f}")

        logger.info(f"RMSE SSHF {metrics['rmse_sshf']:.3f}")
        logger.info(f"RMSE SLHF {metrics['rmse_slhf']:.3f}")

        with open(os.path.join(plot_dir,"metrics.json"),"w") as f:
            json.dump(metrics,f,indent=2)

        plot_scatter(Y_raw,pred,plot_dir)

        plot_error_distribution(Y_raw,pred,plot_dir)

        if lat is not None and lon is not None:
            plot_spatial_error(lat,lon,Y_raw,pred,plot_dir)

    logger.info("Inference completed")

    return pred