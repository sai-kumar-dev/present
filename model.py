#!/usr/bin/env python3

import torch
import torch.nn as nn


# ------------------------------------------------
# ACTIVATION FACTORY
# ------------------------------------------------

def get_activation(name="silu", negative_slope=0.01):

    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=negative_slope)

    if name == "gelu":
        return nn.GELU()

    if name == "silu":
        return nn.SiLU()

    raise ValueError(f"Unsupported activation: {name}")


# ------------------------------------------------
# WEIGHT INITIALIZATION
# ------------------------------------------------

def init_weights(module):

    if isinstance(module, nn.Linear):

        nn.init.xavier_normal_(module.weight)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0.01)


# ------------------------------------------------
# SHARED FEATURE TRUNK
# ------------------------------------------------

class MLPTrunk(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        activation="silu",
        negative_slope=0.01
    ):

        super().__init__()

        act = get_activation(activation, negative_slope)

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)

        for _ in range(num_layers - 1):

            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        self.network = nn.Sequential(*layers)

        self.network.apply(init_weights)

    def forward(self, x):

        return self.network(x)


# ------------------------------------------------
# OUTPUT HEAD
# ------------------------------------------------

class FluxHead(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.layer = nn.Linear(hidden_dim, 1)

        init_weights(self.layer)

    def forward(self, x):

        return self.layer(x)


# ------------------------------------------------
# MAIN PINN MODEL
# ------------------------------------------------

class OceanHeatFluxPINN(nn.Module):

    """
    Neural emulator for ocean surface heat fluxes.

    Inputs:
        u10, v10, t2m, d2m, sp, rho, sst

    Outputs:
        sshf (sensible heat flux)
        slhf (latent heat flux)

    Architecture:
        shared trunk → two specialized heads
    """

    def __init__(
        self,
        input_dim=7,
        hidden_dim=256,
        num_layers=4,
        activation="silu",
        negative_slope=0.01
    ):

        super().__init__()

        self.trunk = MLPTrunk(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            negative_slope=negative_slope
        )

        self.sensible_head = FluxHead(hidden_dim)

        self.latent_head = FluxHead(hidden_dim)

    def forward(self, x):

        features = self.trunk(x)

        sensible = self.sensible_head(features)

        latent = self.latent_head(features)

        out = torch.cat([sensible, latent], dim=1)

        return out