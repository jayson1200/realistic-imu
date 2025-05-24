import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

class Minimizer:
    def __init__(
            self,
            timesteps: int,
            smoothing_weight: float,
            regularization_weight: float,
            start_position_zero_weight: float = 0.0,
            end_position_zero_weight: float   = 0.0,
            start_velocity_zero_weight: float = 0.0,
            end_velocity_zero_weight: float   = 0.0,
    ):
        T = timesteps
        α = smoothing_weight
        λ = regularization_weight
        # scale the “zero‐boundary” weights by T to match the C++ code
        w_sp = start_position_zero_weight * T
        w_ep = end_position_zero_weight   * T
        w_sv = start_velocity_zero_weight * T
        w_ev = end_velocity_zero_weight   * T

        # Build the same sparse B matrix of shape ((T-2 + 4 + T) × T)
        rows, cols, data = [], [], []
        # 1) smoothing rows (second‐difference stencil [-1,2,-1] * α)
        stamp = np.array([-1.0, 2.0, -1.0]) * α
        accT = T - 2
        for i in range(accT):
            for j in range(3):
                rows.append(i)
                cols.append(i + j)
                data.append(stamp[j])

        # 2) start‐/end‐velocity zero constraints
        rows += [accT,      accT,
                 accT + 1,  accT + 1]
        cols += [0,        1,
                 T - 2,    T - 1]
        data += [ w_sv, -w_sv,
                  w_ev, -w_ev]

        # 3) start‐/end‐position zero constraints
        rows += [accT + 2, accT + 3]
        cols += [0,        T - 1]
        data += [ w_sp,     w_ep]

        # 4) diagonal regularization (λ on each x_i)
        for i in range(T):
            rows.append(accT + 4 + i)
            cols.append(i)
            data.append(λ)

        M = accT + 4 + T
        B = coo_matrix((data, (rows, cols)), shape=(M, T))

        # Form C = Bᵀ B once, factorize it, and cache the solver
        C = (B.T @ B).tocsc()
        self._solver = splu(C)
        self._T = T
        self._λ2 = λ**2

    def minimize(self, series_batch: np.ndarray) -> np.ndarray:
        """
        series_batch: shape (N, T)
        returns:      shape (N, T) of smoothed series
        """
        S = np.atleast_2d(series_batch)
        N, T = S.shape
        if T != self._T:
            raise ValueError(f"Expected timesteps={self._T}, got {T}")

        # Build RHS = λ² Sᵀ   (shape T×N)
        rhs = self._λ2 * S.T

        # Solve C·X = rhs  for X (shape T×N)
        X = self._solver.solve(rhs)

        # Return in the original (N×T) layout
        return X.T
