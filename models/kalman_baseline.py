"""Classical baselines: constant-velocity Kalman filter and polynomial extrapolation.

Both implement ``predict(obs_xy [B, T_OBS, 2]) -> [B, T_PRED, 2]`` and are
vectorised over the batch with plain NumPy.
"""
from __future__ import annotations

import numpy as np

import config


class ConstantVelocityKF:
    """Constant-velocity Kalman filter, state ``[x, y, vx, vy]``, written from scratch.

    * Motion model  x_{k+1} = F x_k + w,  w ~ N(0, Q)  (white-noise acceleration)
    * Measurement   z_k     = H x_k + v,  v ~ N(0, R)

    The state is initialised at the 2nd observation with the finite-difference
    velocity of the first two observations, updated with the remaining ones and
    then propagated ``t_pred`` steps with the predict step only.
    """

    name = "kalman"

    def __init__(self, dt: float = config.DT, sigma_accel: float = config.KF_SIGMA_ACCEL,
                 sigma_meas: float = config.KF_SIGMA_MEAS, t_pred: int = config.T_PRED,
                 init_vel_var: float = config.KF_INIT_VEL_VAR) -> None:
        """Build the F, H, Q, R matrices."""
        self.dt, self.t_pred = dt, t_pred
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        q = sigma_accel ** 2
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        self.Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
                               [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]], float)
        self.R = (sigma_meas ** 2) * np.eye(2)
        self.P0 = np.diag([sigma_meas ** 2, sigma_meas ** 2, init_vel_var, init_vel_var])

    def filter(self, obs_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run predict/update over the observations; returns final (state [B,4], P [B,4,4])."""
        obs = np.asarray(obs_xy, dtype=np.float64)
        b = obs.shape[0]
        x = np.concatenate([obs[:, 1], (obs[:, 1] - obs[:, 0]) / self.dt], axis=1)
        P = np.broadcast_to(self.P0, (b, 4, 4)).copy()
        F, H, Q, R = self.F, self.H, self.Q, self.R
        for k in range(2, obs.shape[1]):
            x = x @ F.T                                    # predict
            P = F @ P @ F.T + Q
            y = obs[:, k] - x @ H.T                        # innovation
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)                 # gain [B, 4, 2]
            x = x + np.einsum("bij,bj->bi", K, y)          # update
            P = (np.eye(4) - K @ H) @ P
        return x, P

    def predict(self, obs_xy: np.ndarray) -> np.ndarray:
        """Forecast [B, T_PRED, 2] future positions."""
        x, _ = self.filter(obs_xy)
        out = np.empty((x.shape[0], self.t_pred, 2))
        for k in range(self.t_pred):
            x = x @ self.F.T
            out[:, k] = x[:, :2]
        return out


class PolynomialExtrapolator:
    """Least-squares polynomial fit of x(t) and y(t) over the history, extrapolated."""

    def __init__(self, degree: int = 1, dt: float = config.DT, t_pred: int = config.T_PRED) -> None:
        """Store the polynomial degree and horizon."""
        self.degree, self.dt, self.t_pred = degree, dt, t_pred
        self.name = f"poly{degree}"

    def predict(self, obs_xy: np.ndarray) -> np.ndarray:
        """Forecast [B, T_PRED, 2] future positions (np.polyfit vectorised over the batch)."""
        obs = np.asarray(obs_xy, dtype=np.float64)
        b, t_obs, _ = obs.shape
        t = np.arange(t_obs) * self.dt
        t_fut = (t_obs - 1 + np.arange(1, self.t_pred + 1)) * self.dt
        # polyfit accepts a 2-D y: fit all 2B coordinate series at once
        ys = obs.transpose(0, 2, 1).reshape(-1, t_obs).T          # [T_OBS, 2B]
        coefs = np.polyfit(t, ys, self.degree)                      # [deg+1, 2B]
        vander = np.vander(t_fut, self.degree + 1)                  # [T_PRED, deg+1]
        fut = (vander @ coefs).T.reshape(b, 2, self.t_pred).transpose(0, 2, 1)
        return fut
