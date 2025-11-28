import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
from utils import remove_mean, distances_from_vectors, distance_vectors

from .base_set import BaseSet


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    p = 0.9
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


class LennardJonesPotential(BaseSet):
    def __init__(
        self,
        dim,
        n_particles,
        device,
        eps=1.0,
        rm=1.0,
        oscillator=True,
        oscillator_scale=1.0,
        energy_factor=1.0,
        data_path=None,
    ):
        """Energy for a Lennard-Jones cluster.

        Parameters
        ----------
        dim : int
            Number of degrees of freedom ( = space dimension x n_particles)
        n_particles : int
            Number of Lennard-Jones particles
        eps : float
            LJ well depth epsilon
        rm : float
            LJ well radius R_min
        oscillator : bool
            Whether to use a harmonic oscillator as an external force
        oscillator_scale : float
            Force constant of the harmonic oscillator energy
        two_event_dims : bool
            If True, the energy expects inputs with two event dimensions (particle_id, coordinate).
            Else, use only one event dimension.
        """
        super().__init__()
        self._n_particles = n_particles
        self._n_dims = dim // n_particles

        self._eps = eps
        self._rm = rm
        self.oscillator = oscillator
        self._oscillator_scale = oscillator_scale

        # this is to match the eacf energy with the eq-fm energy
        # for lj13, to match the eacf set energy_factor=0.5
        self._energy_factor = energy_factor
        self.stddevs = 0.6807141304016113 # Calculated from all_split_LJ13-120k.npy
        if data_path is not None:
            data = np.load(data_path, allow_pickle=True)
            self.data = remove_mean(torch.tensor(data), self._n_particles, self._n_dims)
            self.n_data = data.shape[0]
            print(f"Ground truth sample shape: {data.shape}") 
        else:
            self.data = None
            self.n_data = 0
            print("No Ground truth sample provided")
        
        self.device = device
        self.data_ndim = dim

    def gt_logz(self):
        return 0. # This is not the true logZ. Just a placeholder.

    def energy(self, x):
        batch_shape = x.shape[0]
        x = x.view(batch_shape, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )

        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        # lj_energies = torch.clip(lj_energies, -1e4, 1e4)
        lj_energies = lj_energies.view(batch_shape, -1).sum(dim=-1) * self._energy_factor

        if self.oscillator:
            osc_energies = 0.5 * remove_mean(x, self._n_particles, self._n_dims).pow(2).sum(dim=(-2, -1)).view(batch_shape)
            lj_energies = lj_energies + osc_energies * self._oscillator_scale
        
        return lj_energies

    def log_reward(self, x):
        if self._n_particles == 13:
            clip = 1e4
        if self._n_particles == 55:
            clip = 1e6
        clipped_energy = torch.clamp(self.energy(x), max=clip)
        return -clipped_energy    

    def sample(self, batch_size):
        assert self.data is not None, "No ground truth sample provided"
        index = np.random.choice(self.n_data, batch_size, replace=False)
        return self.data[index].to(self.device)
    

    def viz_pdf(self, fsave="density.png", lim=3):
        raise NotImplementedError

    def __getitem__(self, idx):
        del idx
        return self.data[0]
