#!/usr/bin/env python3

import torch


# ------------------------------------------------
# PHYSICAL CONSTANTS
# ------------------------------------------------

Cp = 1004.0
Lv = 2.5e6

CH = 1.1e-3
CE = 1.2e-3

MAX_SENSIBLE = 500.0
MAX_LATENT = 1000.0


# ------------------------------------------------
# NORMALIZATION SCALES
# must match sampler.py
# ------------------------------------------------

INPUT_SCALE = torch.tensor(
    [20, 20, 320, 320, 105000, 1.5, 310],
    dtype=torch.float32
)

TARGET_SCALE = torch.tensor(
    [500, 500],
    dtype=torch.float32
)


# ------------------------------------------------
# SATURATION VAPOR PRESSURE
# ------------------------------------------------

def saturation_vapor_pressure(T):

    Tc = T - 273.15

    Tc = torch.clamp(Tc, -80.0, 60.0)

    exponent = (17.67 * Tc) / (Tc + 243.5)

    exponent = torch.clamp(exponent, -50.0, 50.0)

    es = 6.112 * torch.exp(exponent)

    return es * 100.0


# ------------------------------------------------
# SPECIFIC HUMIDITY
# ------------------------------------------------

def specific_humidity(e, p):

    denom = p - 0.378 * e

    denom = torch.clamp(denom, min=1e-6)

    q = 0.622 * e / denom

    return q


# ------------------------------------------------
# BULK FLUX PHYSICS
# ------------------------------------------------

def bulk_flux_physics(X):

    scale = INPUT_SCALE.to(X.device)

    X_phys = X * scale

    u10 = X_phys[:, 0]
    v10 = X_phys[:, 1]
    t2m = X_phys[:, 2]
    d2m = X_phys[:, 3]
    sp  = X_phys[:, 4]
    rho = X_phys[:, 5]
    sst = X_phys[:, 6]

    wind_sq = u10 * u10 + v10 * v10

    wind = torch.sqrt(wind_sq + 1e-6)

    dT = sst - t2m

    e_air = saturation_vapor_pressure(d2m)
    q_air = specific_humidity(e_air, sp)

    e_surf = saturation_vapor_pressure(sst)
    q_surf = specific_humidity(e_surf, sp)

    dq = q_surf - q_air

    sensible_bulk = rho * Cp * CH * wind * dT
    latent_bulk = rho * Lv * CE * wind * dq

    sensible_bulk = torch.nan_to_num(sensible_bulk)
    latent_bulk = torch.nan_to_num(latent_bulk)

    return sensible_bulk, latent_bulk, wind, dT


# ------------------------------------------------
# FLUX DIRECTION RULE
# ------------------------------------------------

def flux_direction_penalty(sensible_pred, dT):

    expected = torch.sign(dT)

    penalty = torch.relu(-expected * sensible_pred)

    return penalty.mean()


# ------------------------------------------------
# WIND MONOTONICITY CONSTRAINT
# ------------------------------------------------

def wind_flux_penalty(pred_flux, wind):

    n = pred_flux.shape[0]

    if n < 4:
        return torch.tensor(0.0, device=pred_flux.device)

    idx = torch.randperm(n, device=pred_flux.device)

    half = n // 2

    i = idx[:half]
    j = idx[half:half*2]

    wind_i = wind[i]
    wind_j = wind[j]

    flux_i = pred_flux[i]
    flux_j = pred_flux[j]

    mask = wind_i > wind_j

    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_flux.device)

    penalty = torch.relu(flux_j[mask] - flux_i[mask])

    return penalty.mean()


# ------------------------------------------------
# EXTREME FLUX BOUNDS
# ------------------------------------------------

def extreme_flux_penalty(sensible_pred, latent_pred):

    p1 = torch.relu(torch.abs(sensible_pred) - MAX_SENSIBLE)

    p2 = torch.relu(torch.abs(latent_pred) - MAX_LATENT)

    return (p1.mean() + p2.mean())


# ------------------------------------------------
# MAIN PHYSICS LOSS
# ------------------------------------------------

def physics_loss(X, Y_pred):

    sensible_pred = Y_pred[:, 0]
    latent_pred = Y_pred[:, 1]

    sensible_bulk, latent_bulk, wind, dT = bulk_flux_physics(X)

    scale = TARGET_SCALE.to(X.device)

    sensible_bulk = sensible_bulk / scale[0]
    latent_bulk = latent_bulk / scale[1]

    sensible_bulk = torch.nan_to_num(sensible_bulk)
    latent_bulk = torch.nan_to_num(latent_bulk)

    bulk_loss = (
        torch.mean((sensible_pred - sensible_bulk) ** 2) +
        torch.mean((latent_pred - latent_bulk) ** 2)
    )

    direction_loss = flux_direction_penalty(sensible_pred, dT)

    wind_loss = wind_flux_penalty(torch.abs(latent_pred), wind)

    extreme_loss = extreme_flux_penalty(sensible_pred, latent_pred)

    physics = (
        bulk_loss
        + 0.1 * direction_loss
        + 0.05 * wind_loss
        + 0.05 * extreme_loss
    )

    physics = torch.nan_to_num(physics)

    return physics