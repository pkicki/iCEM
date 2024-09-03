import os
import pickle
from warnings import warn

import allogger
import colorednoise
import numpy as np
from gymnasium import spaces
from scipy.stats import truncnorm

from controllers.mpc import MpcController
from icem.controllers.icem import MpcICem
from misc.rolloutbuffer import RolloutBuffer

from matplotlib import pyplot as plt
from scipy.signal import periodogram, welch
from scipy.signal import butter, lfilter, freqz


# our improved CEM
class MpcFCem(MpcICem):
    def __init__(self, *, action_sampler_params, **kwargs):
        super().__init__(action_sampler_params=action_sampler_params, **kwargs)
        if self.cutoff_freq > 0:
            self.lp_filter_b, self.lp_filter_a = butter(self.order, self.cutoff_freq, fs=self.fs, btype='low', analog=False)

    def lp_filter(self, data, axis=-1):
        y = lfilter(self.lp_filter_b, self.lp_filter_a, data, axis=axis)
        return y

    def sample_action_sequences(self, obs, num_traj, time_slice=None):
        """
        :param num_traj: number of trajectories
        :param obs: current observation
        :type time_slice: slice
        """
        # colored noise
        if self.noise_beta > 0:
            assert (self.mean.ndim == 2)
            # Important improvement
            # self.mean has shape h,d: we need to swap d and h because temporal correlations are in last axis)
            # noinspection PyUnresolvedReferences
            samples = colorednoise.powerlaw_psd_gaussian(self.noise_beta, size=(num_traj, self.mean.shape[1],
                                                                                self.mean.shape[0])).transpose(
                [0, 2, 1])
            #a = 0
            #nperseg = 1024
            #dt = self.forward_model.env.dt
            #f, psd = welch(samples[0, :, 0], fs=1./dt, nperseg=nperseg)
            #plt.plot(f, psd)
            #plt.show()
        elif self.cutoff_freq > 0:
            samples = np.random.randn(num_traj, *self.mean.shape)
            samples = self.lp_filter(samples, axis=1)
        else:
            samples = np.random.randn(num_traj, *self.mean.shape)
            

        samples = np.clip(samples * self.std + self.mean, self.env.action_space.low, self.env.action_space.high)
        if time_slice is not None:
            samples = samples[:, time_slice]
        return samples

    def _parse_action_sampler_params(
            self, *,
            alpha,
            elites_size,
            opt_iterations,
            init_std,
            use_mean_actions,
            keep_previous_elites,
            shift_elites_over_time,
            fraction_elites_reused,
            noise_beta=1,
            cutoff_freq=0.,
            order=2,
            ):

        self.alpha = alpha
        self.elites_size = elites_size
        self.opt_iter = opt_iterations
        self.init_std = init_std
        self.use_mean_actions = use_mean_actions
        self.keep_previous_elites = keep_previous_elites
        self.shift_elites_over_time = shift_elites_over_time
        self.fraction_elites_reused = fraction_elites_reused
        self.noise_beta = noise_beta
        self.cutoff_freq = cutoff_freq
        self.order = order
        self.fs = 1.0 / self.forward_model.env.dt