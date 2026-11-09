import numpy as np
import torch

from HOCBF_SAC.utils import to_numpy, to_tensor

DYNAMICS_MODE = {"Unicycle": {"n_s": 4, "n_u": 2}}


class DynamicsModel:
    def __init__(self, env, args):
        self.env = env
        if self.env.dynamics_mode != "Unicycle":
            raise Exception("Only Unicycle dynamics are supported in this project.")

        self.get_f, self.get_g = self.get_dynamics()
        self.n_s = DYNAMICS_MODE["Unicycle"]["n_s"]
        self.n_u = DYNAMICS_MODE["Unicycle"]["n_u"]

    def predict_next_state(self, state_batch, u_batch, t_batch=None):
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = np.expand_dims(state_batch, axis=0)

        next_state_batch = state_batch + self.env.dt * (self.get_f(state_batch) + (self.get_g(state_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1)  )

        pred_std = np.zeros(state_batch.shape)

        if expand_dims:
            next_state_batch = next_state_batch.squeeze(0)
            pred_std = pred_std.squeeze(0)

        if t_batch is not None:
            next_t_batch = t_batch + self.env.dt
            return next_state_batch, self.env.dt * pred_std, next_t_batch
        return next_state_batch, self.env.dt * pred_std, t_batch

    def predict_next_obs(self, state, u):
        next_state, _, _ = self.predict_next_state(state, u)
        return self.get_obs(next_state)

    def get_dynamics(self):
        def get_f(state_batch, t_batch=None):
            f_x = np.zeros(state_batch.shape)
            f_x[:, 0] = state_batch[:, 3] * np.cos(state_batch[:, 2])
            f_x[:, 1] = state_batch[:, 3] * np.sin(state_batch[:, 2])
            return f_x

        def get_g(state_batch, t_batch=None):
            # u = [a, omega]
            g_x = np.zeros((state_batch.shape[0], 4, 2))
            g_x[:, 2, 1] = 1.0
            g_x[:, 3, 0] = 1.0
            return g_x

        return get_f, get_g

    def get_state(self, obs):
        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        theta = np.arctan2(obs[:, 3], obs[:, 2])
        state_batch = np.zeros((obs.shape[0], 4))
        state_batch[:, 0] = obs[:, 0]
        state_batch[:, 1] = obs[:, 1]
        state_batch[:, 2] = theta
        state_batch[:, 3] = obs[:, 4]

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch

    def get_obs(self, state_batch):
        obs = np.zeros((state_batch.shape[0], 9))
        obs[:, 0] = state_batch[:, 0]
        obs[:, 1] = state_batch[:, 1]
        obs[:, 2] = np.cos(state_batch[:, 2])
        obs[:, 3] = np.sin(state_batch[:, 2])
        obs[:, 4] = state_batch[:, 3]
        ref = self.env.ref_state.reshape(1, 4)
        e = state_batch - ref
        e[:, 2] = np.arctan2(np.sin(e[:, 2]), np.cos(e[:, 2]))
        obs[:, 5] = e[:, 0]
        obs[:, 6] = e[:, 1]
        obs[:, 7] = e[:, 2]
        obs[:, 8] = e[:, 3]
        return obs

    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
