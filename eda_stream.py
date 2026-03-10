#!/usr/bin/env python3

import os
import json
import psutil
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from core import clear_memory
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

INPUT_SCALE = np.array([20,20,320,320,105000,1.5,310])
TARGET_SCALE = np.array([500,500])

VAR_NAMES = ["u10","v10","t2m","d2m","sp","rho","sst"]
TARGET_NAMES = ["sshf","slhf"]

FLUX_BINS = [0,50,150,300,1e9]

MASK_FILE="ocean_mask.npz"

plt.rcParams["figure.dpi"]=140
plt.rcParams["savefig.bbox"]="tight"


# ============================================================
# UTILITIES
# ============================================================

def create_run_dir(base="runs"):

    os.makedirs(base, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid()

    run_dir = os.path.join(base, f"run_{ts}_{pid}")

    os.makedirs(run_dir, exist_ok=True)

    # core directories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "epochs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

    # EDA hierarchy
    os.makedirs(os.path.join(run_dir, "eda"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "eda", "epochs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "eda", "global"), exist_ok=True)

    meta = {
        "created": now(),
        "hostname": socket.gethostname(),
        "pid": pid
    }

    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return run_dir

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p):
    os.makedirs(p,exist_ok=True)


def system_stats():

    mem=psutil.virtual_memory()

    return {
        "cpu":psutil.cpu_percent(),
        "ram":round(mem.used/1e9,2),
        "ram_total":round(mem.total/1e9,2)
    }


def log(run_dir,msg):

    s=system_stats()

    line=f"[{now()}] {msg} | CPU {s['cpu']}% RAM {s['ram']}/{s['ram_total']} GB"

    print(line,flush=True)

    with open(os.path.join(run_dir,"logs","eda.log"),"a") as f:
        f.write(line+"\n")


def save_json(path,data):

    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def load_ocean_mask():

    if not os.path.exists(MASK_FILE):
        return None

    m=np.load(MASK_FILE)

    return m["lat"],m["lon"]


# ============================================================
# BATCH EDA
# ============================================================

class BatchEDA:

    def __init__(self,batch,run_dir):

        self.batch=batch
        self.run_dir=run_dir

        self.batch_id=batch["batch"]+1

        self.X=batch["X"].cpu().numpy()
        self.Y=batch["Y"].cpu().numpy()

        self.lat=np.asarray(batch["lat"])
        self.lon=np.asarray(batch["lon"])

        self.time=pd.to_datetime(batch["time"])

        self.X_raw=self.X*INPUT_SCALE
        self.Y_raw=self.Y*TARGET_SCALE

        self.wind=np.sqrt(self.X_raw[:,0]**2+self.X_raw[:,1]**2)

        self.wind_dir=np.arctan2(self.X_raw[:,1],self.X_raw[:,0])

        self.flux=np.abs(self.Y_raw[:,0])+np.abs(self.Y_raw[:,1])

        self.batch_dir=os.path.join(run_dir,f"batch_{self.batch_id:03d}")
        self.fig_dir=os.path.join(self.batch_dir,"figures")

        ensure_dir(self.batch_dir)
        ensure_dir(self.fig_dir)


    def save(self,name):

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir,name))
        plt.close()


# ============================================================
# SPATIAL
# ============================================================

    def spatial_coverage(self):

        plt.figure(figsize=(10,5))

        sc=plt.scatter(
            self.lon,
            self.lat,
            c=self.flux,
            s=3,
            cmap="inferno",
            alpha=0.6
        )

        plt.colorbar(sc,label="Flux magnitude")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        self.save("spatial_coverage.png")


    def spatial_density(self):

        plt.figure(figsize=(10,5))

        hb=plt.hexbin(
            self.lon,
            self.lat,
            gridsize=80,
            bins="log",
            cmap="plasma"
        )

        plt.colorbar(hb,label="log sample density")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        self.save("spatial_density.png")


    def ocean_mask_overlay(self):

        mask=load_ocean_mask()

        if mask is None:
            return

        lat_mask,lon_mask=mask

        plt.figure(figsize=(10,5))

        plt.scatter(lon_mask,lat_mask,
            s=1,color="lightblue",alpha=0.2)

        plt.scatter(self.lon,self.lat,
            s=3,color="red",alpha=0.7)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.title("Samples on ocean mask")

        self.save("samples_on_ocean_mask.png")


# ============================================================
# TEMPORAL
# ============================================================

    def year_distribution(self):

        years=self.time.year

        bins=np.arange(years.min(),years.max()+2)-0.5

        plt.figure(figsize=(12,4))

        plt.hist(years,bins=bins)

        plt.xticks(np.arange(years.min(),years.max()+1,2))

        plt.xlabel("Year")

        self.save("year_distribution.png")


    def season_distribution(self):

        season_map={
        12:"winter",1:"winter",2:"winter",
        3:"spring",4:"spring",5:"spring",
        6:"summer",7:"summer",8:"summer",
        9:"autumn",10:"autumn",11:"autumn"
        }

        seasons=[season_map[m] for m in self.time.month]

        plt.figure()

        sns.countplot(x=seasons,
        order=["winter","spring","summer","autumn"])

        self.save("season_distribution.png")


# ============================================================
# LAT / LON
# ============================================================

    def latitude_bands(self):

        lat=np.abs(self.lat)

        labels=[]

        for v in lat:

            if v<20: labels.append("tropics")
            elif v<45: labels.append("subtropics")
            elif v<70: labels.append("midlat")
            else: labels.append("polar")

        plt.figure()

        sns.countplot(x=labels,
        order=["tropics","subtropics","midlat","polar"])

        self.save("latitude_bands.png")


    def longitude_distribution(self):

        lon=self.lon.copy()

        lon[lon>180]-=360

        plt.figure()

        sns.histplot(lon,bins=60)

        plt.xlabel("Longitude")

        self.save("longitude_distribution.png")


# ============================================================
# FLUX STRATIFICATION
# ============================================================

    def flux_distribution(self):

        plt.figure()

        sns.histplot(self.flux,bins=80)

        plt.yscale("log")

        plt.xlabel("Flux magnitude")

        self.save("flux_distribution.png")


    def flux_components(self):

        sshf=self.Y_raw[:,0]
        slhf=self.Y_raw[:,1]

        plt.figure()

        sns.histplot(sshf,bins=80,label="sshf")
        sns.histplot(slhf,bins=80,label="slhf")

        plt.legend()

        self.save("flux_components.png")


    def flux_bins(self):

        bins=np.digitize(self.flux,FLUX_BINS)

        plt.figure()

        sns.countplot(x=bins)

        plt.xlabel("Flux bin")

        self.save("flux_bins.png")


# ============================================================
# PHYSICS
# ============================================================

    def wind_flux(self):

        idx=np.random.choice(len(self.wind),min(5000,len(self.wind)),False)

        plt.figure()

        plt.hexbin(
            self.wind[idx],
            self.flux[idx],
            gridsize=60,
            bins="log",
            cmap="plasma"
        )

        plt.xlabel("Wind speed")
        plt.ylabel("Flux")

        self.save("wind_flux.png")


    def sst_flux(self):

        sst=self.X_raw[:,6]
        slhf=self.Y_raw[:,1]

        idx=np.random.choice(len(sst),min(5000,len(sst)),False)

        plt.figure()

        plt.scatter(
            sst[idx],
            slhf[idx],
            s=3,
            alpha=0.3
        )

        plt.xlabel("SST")
        plt.ylabel("Latent flux")

        self.save("sst_flux.png")


    def flux_regime_map(self):

        sshf=np.abs(self.Y_raw[:,0])
        slhf=np.abs(self.Y_raw[:,1])

        ratio=slhf/(sshf+1e-6)

        plt.figure(figsize=(10,5))

        sc=plt.scatter(
            self.lon,
            self.lat,
            c=ratio,
            cmap="coolwarm",
            s=3
        )

        plt.colorbar(sc,label="Latent/Sensible ratio")

        self.save("flux_regime_map.png")


# ============================================================
# VARIABLES
# ============================================================

    def variable_distributions(self):

        for i,name in enumerate(VAR_NAMES):

            plt.figure()

            sns.histplot(self.X_raw[:,i],bins=80)

            plt.xlabel(name)

            self.save(f"{name}_distribution.png")


# ============================================================
# RUN
# ============================================================

    def run_all(self):

        log(self.run_dir,f"Batch {self.batch_id} EDA")

        self.spatial_coverage()
        self.spatial_density()
        self.ocean_mask_overlay()

        self.year_distribution()
        self.season_distribution()

        self.latitude_bands()
        self.longitude_distribution()

        self.flux_distribution()
        self.flux_components()
        self.flux_bins()

        self.wind_flux()
        self.sst_flux()
        self.flux_regime_map()

        self.variable_distributions()
        clear_memory()


# ============================================================
# GLOBAL EDA
# ============================================================

class GlobalEDA:

    def __init__(self,run_dir):

        self.run_dir=run_dir

        self.lat=[]
        self.lon=[]
        self.flux=[]
        self.wind=[]
        self.months=[]
        self.years=[]
        self.X=[]
        self.Y=[]

        self.samples=0


    def add_batch(self, lat, lon, flux, wind, time, X=None, Y=None):

        lat = np.asarray(lat)
        lon = np.asarray(lon)
        flux = np.asarray(flux)
        wind = np.asarray(wind)

        if len(flux) == 0:
            return

        self.lat.extend(lat.tolist())
        self.lon.extend(lon.tolist())
        self.flux.extend(flux.tolist())
        self.wind.extend(wind.tolist())

        t = pd.to_datetime(time)
        self.months.extend(t.month.tolist())
        self.years.extend(t.year.tolist())

        if X is not None and len(X) > 0:
            self.X.append(np.asarray(X))

        if Y is not None and len(Y) > 0:
            self.Y.append(np.asarray(Y))

        self.samples += len(flux)


    def finalize(self):

        global_dir=os.path.join(self.run_dir,"global")
        fig_dir=os.path.join(global_dir,"figures")

        ensure_dir(global_dir)
        ensure_dir(fig_dir)

        lat=np.array(self.lat)
        lon=np.array(self.lon)
        flux=np.array(self.flux)
        wind=np.array(self.wind)
        years=np.array(self.years)

        # sample density map

        plt.figure(figsize=(10,5))

        plt.hexbin(
            lon,
            lat,
            gridsize=120,
            bins="log",
            cmap="plasma"
        )

        plt.colorbar(label="log density")

        plt.savefig(os.path.join(fig_dir,"global_sample_density.png"))
        plt.close()

        # flux map

        plt.figure(figsize=(10,5))

        sc=plt.scatter(
            lon,
            lat,
            c=flux,
            s=2,
            cmap="inferno",
            alpha=0.4
        )

        plt.colorbar(sc,label="Flux")

        plt.savefig(os.path.join(fig_dir,"global_flux_map.png"))
        plt.close()

        # year distribution

        bins=np.arange(years.min(),years.max()+2)-0.5

        plt.figure(figsize=(12,4))

        plt.hist(years,bins=bins)

        plt.xticks(np.arange(years.min(),years.max()+1,2))

        plt.savefig(os.path.join(fig_dir,"year_distribution.png"))
        plt.close()

        # season distribution

        season_map={
        12:"winter",1:"winter",2:"winter",
        3:"spring",4:"spring",5:"spring",
        6:"summer",7:"summer",8:"summer",
        9:"autumn",10:"autumn",11:"autumn"
        }

        seasons=[season_map[m] for m in self.months]

        plt.figure()

        sns.countplot(x=seasons,
        order=["winter","spring","summer","autumn"])

        plt.savefig(os.path.join(fig_dir,"season_distribution.png"))
        plt.close()

        # correlation
# correlation matrix (only if we actually collected data)

        if len(self.X) > 0 and len(self.Y) > 0:

            X = np.vstack(self.X)
            Y = np.vstack(self.Y)

            df = pd.DataFrame(
                np.hstack([X, Y]),
                columns=[
                    "u10","v10","t2m","d2m","sp","rho","sst",
                    "sshf","slhf"
                ]
            )

            plt.figure(figsize=(8,6))

            sns.heatmap(
                df.corr(),
                cmap="coolwarm",
                center=0
            )

            plt.title("Input–Flux Correlation Matrix")

            plt.savefig(os.path.join(fig_dir,"correlation_matrix.png"))
            plt.close()

        stats={
            "samples":self.samples,
            "mean_flux":float(np.mean(flux)),
            "mean_wind":float(np.mean(wind))
        }

        save_json(os.path.join(global_dir,"stats.json"),stats)

        log(self.run_dir,"Global EDA completed")
