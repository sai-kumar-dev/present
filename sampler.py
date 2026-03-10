#!/usr/bin/env python3

import os
import pickle
import argparse
from datetime import datetime
from collections import OrderedDict, defaultdict
from core import logger, clear_memory

import numpy as np
import xarray as xr
import torch
import psutil


BASE_PATH="/data/jayanth_works/era5_inputfluxes/1990-2025"

MASK_FILE="ocean_mask.npz"

INDEX_CACHE="era5_index_cache.pkl"
SEASON_CACHE="season_index_cache.pkl"
SPATIAL_CACHE="spatial_tiles_cache.pkl"


R_D=287.05
SECONDS_PER_HOUR=3600

DATASET_CACHE_LIMIT=24
SPATIAL_TILE=10

INPUT_SCALE=np.array([20,20,320,320,105000,1.5,310],dtype=np.float32)
TARGET_SCALE=np.array([500,500],dtype=np.float32)

SEASONS={
"winter":[12,1,2],
"spring":[3,4,5],
"summer":[6,7,8],
"autumn":[9,10,11]
}

FLUX_BINS=np.array([0,50,150,300,1e9])

MONTH_MAP={
"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
"july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

dataset_cache=OrderedDict()


def load_dataset(path):

    if path in dataset_cache:
        dataset_cache.move_to_end(path)
        return dataset_cache[path]

    ds=xr.open_dataset(path,engine="netcdf4",decode_times=True)

    dataset_cache[path]=ds

    if len(dataset_cache)>DATASET_CACHE_LIMIT:
        _,v=dataset_cache.popitem(last=False)
        v.close()

    return ds


def load_ocean_mask():

    m=np.load(MASK_FILE)

    ys=m["ys"]
    xs=m["xs"]
    lat=m["lat"]
    lon=m["lon"]

    return ys,xs,lat,lon


def build_index(start,end):

    if os.path.exists(INDEX_CACHE):
        return pickle.load(open(INDEX_CACHE,"rb"))

    index={}

    for year in range(start,end+1):

        ypath=os.path.join(BASE_PATH,str(year))

        if not os.path.exists(ypath):
            continue

        for folder in os.listdir(ypath):

            mname=folder.split("_")[0].lower()

            if mname not in MONTH_MAP:
                continue

            month=MONTH_MAP[mname]

            mpath=os.path.join(ypath,folder)

            inst=os.path.join(mpath,"instant_u10_v10_t2m_msl_sst.nc")
            param=os.path.join(mpath,"instant_d2m_sp.nc")
            accum=os.path.join(mpath,"accum_slhf_ssr_sshf_ssrd_tsr.nc")

            if os.path.exists(inst) and os.path.exists(param) and os.path.exists(accum):

                index[(year,month)]={"instant":inst,"param":param,"accum":accum}

    pickle.dump(index,open(INDEX_CACHE,"wb"))

    return index


def build_season_index(index):

    if os.path.exists(SEASON_CACHE):
        return pickle.load(open(SEASON_CACHE,"rb"))

    sidx={k:[] for k in SEASONS}

    for (year,month) in index:

        for s,months in SEASONS.items():

            if month in months:
                sidx[s].append((year,month))

    pickle.dump(sidx,open(SEASON_CACHE,"wb"))

    return sidx


def build_spatial_tiles(lat,lon):

    if os.path.exists(SPATIAL_CACHE):
        return pickle.load(open(SPATIAL_CACHE,"rb"))

    tiles=defaultdict(list)

    for i,(la,lo) in enumerate(zip(lat,lon)):

        lt=int((la+90)//SPATIAL_TILE)
        ln=int((lo+180)//SPATIAL_TILE)

        tiles[(lt,ln)].append(i)

    tiles=dict(tiles)

    pickle.dump(tiles,open(SPATIAL_CACHE,"wb"))

    return tiles


def sample_time_slice(paths,ys,xs,grid_idx,t):

    ds_i=load_dataset(paths["instant"])
    ds_a=load_dataset(paths["accum"])
    ds_p=load_dataset(paths["param"])

    tdim="valid_time" if "valid_time" in ds_i.dims else "time"

    yi=ys[grid_idx]
    xi=xs[grid_idx]

    slice_u10 = ds_i["u10"].isel({tdim:t}).data
    slice_v10 = ds_i["v10"].isel({tdim:t}).data
    slice_t2m = ds_i["t2m"].isel({tdim:t}).data
    slice_sst = ds_i["sst"].isel({tdim:t}).data

    slice_d2m = ds_p["d2m"].isel({tdim:t}).data
    slice_sp  = ds_p["sp"].isel({tdim:t}).data

    slice_sshf = ds_a["sshf"].isel({tdim:t}).data
    slice_slhf = ds_a["slhf"].isel({tdim:t}).data

    u10 = slice_u10[yi, xi]
    v10 = slice_v10[yi, xi]
    t2m = slice_t2m[yi, xi]
    sst = slice_sst[yi, xi]

    d2m = slice_d2m[yi, xi]
    sp  = slice_sp[yi, xi]

    sshf = slice_sshf[yi, xi] / SECONDS_PER_HOUR
    slhf = slice_slhf[yi, xi] / SECONDS_PER_HOUR

    rho = sp / (R_D * np.maximum(t2m, 1e-6))

    X=np.stack([u10,v10,t2m,d2m,sp,rho,sst],axis=1)
    Y=np.stack([sshf,slhf],axis=1)

    flux=np.abs(Y[:,0])+np.abs(Y[:,1])

    times=np.repeat(ds_i[tdim].values[t],len(grid_idx))

    return X,Y,flux,times


def run(args):

    rng=np.random.default_rng(args.seed)

    ys,xs,lat,lon=load_ocean_mask()

    GRID=len(ys)

    index=build_index(args.start_year,args.end_year)
    season_index=build_season_index(index)
    spatial_tiles=build_spatial_tiles(lat,lon)

    seasons=list(SEASONS.keys())
    tiles=list(spatial_tiles.keys())
    keys=list(index.keys())
    years=sorted(set(k[0] for k in keys))

    BLOCK=1024

    for b in range(args.batches):
        batch_start_time = datetime.now()
        logger.info(f"Sampler batch {b+1} START")
        Xs=[];Ys=[];lat_all=[];lon_all=[];time_all=[]

        def draw_random():

            key=keys[rng.integers(len(keys))]
            paths=index[key]

            ds=load_dataset(paths["instant"])
            tdim="valid_time" if "valid_time" in ds.dims else "time"

            t=rng.integers(ds.sizes[tdim])

            start = rng.integers(0, GRID - BLOCK)
            grid_idx = np.arange(start, start + BLOCK)

            X,Y,flux,times=sample_time_slice(paths,ys,xs,grid_idx,t)

            return X,Y,flux,grid_idx,times


        # -------------------------
        # RANDOM
        # -------------------------
        if args.sampler=="random":

            while len(Xs)*BLOCK < args.batch_size:

                X,Y,_,grid_idx,times=draw_random()

                Xs.append(X);Ys.append(Y)
                lat_all.append(lat[grid_idx])
                lon_all.append(lon[grid_idx])
                time_all.append(times)


        # -------------------------
        # SEASONAL
        # -------------------------
        elif args.sampler=="seasonal":

            quota=args.batch_size//4
            rot = b % len(seasons)
            season_order = seasons[rot:] + seasons[:rot]
            for s in season_order:

                count=0

                while count<quota:

                    key=season_index[s][rng.integers(len(season_index[s]))]

                    paths=index[key]

                    ds=load_dataset(paths["instant"])
                    tdim="valid_time" if "valid_time" in ds.dims else "time"

                    t=rng.integers(ds.sizes[tdim])

                    grid_idx=rng.choice(GRID,BLOCK)

                    X,Y,_,times=sample_time_slice(paths,ys,xs,grid_idx,t)

                    Xs.append(X);Ys.append(Y)

                    lat_all.append(lat[grid_idx])
                    lon_all.append(lon[grid_idx])
                    time_all.append(times)

                    count+=len(X)


        # -------------------------
        # TEMPORAL
        # -------------------------
        elif args.sampler=="temporal":
            rot = b % len(years)
            year_order = years[rot:] + years[:rot]
            quota=args.batch_size//len(years)

            for y in year_order:

                months=[m for (yr,m) in keys if yr==y]

                count=0

                while count<quota:

                    m=months[rng.integers(len(months))]
                    key=(y,m)

                    paths=index[key]

                    t=rng.integers(ds.sizes[tdim])

                    grid_idx=rng.integers(0, GRID, BLOCK)

                    X,Y,_,times=sample_time_slice(paths,ys,xs,grid_idx,t)

                    Xs.append(X);Ys.append(Y)

                    lat_all.append(lat[grid_idx])
                    lon_all.append(lon[grid_idx])
                    time_all.append(times)

                    count+=len(X)


        # -------------------------
        # SPATIAL
        # -------------------------
        elif args.sampler=="spatial":
            rot = b % len(tiles)
            tile_order = tiles[rot:] + tiles[:rot]
            quota=args.batch_size//len(tiles)

            for tile in tile_order:

                grid_pool=spatial_tiles[tile]

                count=0

                while count<quota:

                    key=keys[rng.integers(len(keys))]

                    paths=index[key]

                    ds=load_dataset(paths["instant"])
                    tdim="valid_time" if "valid_time" in ds.dims else "time"

                    t=rng.integers(ds.sizes[tdim])

                    grid_idx=rng.choice(grid_pool,BLOCK,replace=True)

                    X,Y,_,times=sample_time_slice(paths,ys,xs,grid_idx,t)

                    Xs.append(X);Ys.append(Y)

                    lat_all.append(lat[grid_idx])
                    lon_all.append(lon[grid_idx])
                    time_all.append(times)

                    count+=len(X)


        # -------------------------
        # FLUX STRATIFIED
        # -------------------------
        elif args.sampler=="flux":

            quota=args.batch_size//(len(FLUX_BINS)-1)
            bins=[0]*(len(FLUX_BINS)-1)

            attempts=0

            while sum(bins)<args.batch_size and attempts<100000:

                X,Y,flux,grid_idx,times=draw_random()

                for i,f in enumerate(flux):

                    b=np.searchsorted(FLUX_BINS,f)-1

                    if b<0 or b>=len(bins):
                        continue

                    if bins[b]>=quota:
                        continue

                    Xs.append(X[i:i+1])
                    Ys.append(Y[i:i+1])

                    lat_all.append([lat[grid_idx[i]]])
                    lon_all.append([lon[grid_idx[i]]])
                    time_all.append([times[i]])

                    bins[b]+=1

                    if sum(bins)>=args.batch_size:
                        break

                attempts+=1


        # -------------------------
        # HYBRID (SEASON + FLUX)
        # -------------------------
        elif args.sampler=="hybrid":

            season_quota=args.batch_size//4
            flux_quota=season_quota//(len(FLUX_BINS)-1)

            for s in seasons:

                bins=[0]*(len(FLUX_BINS)-1)

                while sum(bins)<season_quota:

                    key=season_index[s][rng.integers(len(season_index[s]))]

                    paths=index[key]

                    ds=load_dataset(paths["instant"])
                    tdim="valid_time" if "valid_time" in ds.dims else "time"

                    t=rng.integers(ds.sizes[tdim])

                    grid_idx=rng.choice(GRID,BLOCK)

                    X,Y,flux,times=sample_time_slice(paths,ys,xs,grid_idx,t)

                    for i,f in enumerate(flux):

                        b=np.searchsorted(FLUX_BINS,f)-1

                        if b<0 or b>=len(bins):
                            continue

                        if bins[b]>=flux_quota:
                            continue

                        Xs.append(X[i:i+1])
                        Ys.append(Y[i:i+1])

                        lat_all.append([lat[grid_idx[i]]])
                        lon_all.append([lon[grid_idx[i]]])
                        time_all.append([times[i]])

                        bins[b]+=1

                        if sum(bins)>=season_quota:
                            break


        # -------------------------
        # FINAL CONCAT
        # -------------------------
        X=np.concatenate(Xs)[:args.batch_size]
        Y=np.concatenate(Ys)[:args.batch_size]

        lat_batch=np.concatenate(lat_all)[:args.batch_size]
        lon_batch=np.concatenate(lon_all)[:args.batch_size]
        time_batch=np.concatenate(time_all)[:args.batch_size]
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)
        X=X.astype(np.float32)/INPUT_SCALE
        Y=Y.astype(np.float32)/TARGET_SCALE
        batch_end_time = datetime.now()
        duration = (batch_end_time - batch_start_time).total_seconds()

        logger.info(f"Sampler batch {b+1} END | duration {duration:.2f}s | samples {len(X)}")
        yield {
            "X":torch.from_numpy(X),
            "Y":torch.from_numpy(Y),
            "lat":lat_batch,
            "lon":lon_batch,
            "time":time_batch,
            "batch":b
        }
        clear_memory()


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--sampler",
        choices=["random","temporal","seasonal","spatial","flux","hybrid"],
        default="random"
    )

    parser.add_argument("--batch_size",type=int,default=15000)
    parser.add_argument("--batches",type=int,default=10)
    parser.add_argument("--start_year",type=int,default=1990)
    parser.add_argument("--end_year",type=int,default=2025)
    parser.add_argument("--seed",type=int,default=42)

    args=parser.parse_args()

    engine=run(args)

    for _ in engine:
        pass


if __name__=="__main__":
    main()
