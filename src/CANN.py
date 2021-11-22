import numpy as np


def rotation_matrix(theta, degrees=True):
    """
    2d-rotation matrix implementation
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def grid_cell(f=1, orientation_offset=0, degrees=True, **kwargs):
    """
    Implementation following the logic described in the paper:
    From grid cells to place cells: A mathematical model
    - Moser & Einevoll

    Usage:
        Set static params for a grid cell as: gc = grid_cell(*args)
        Then calculate grid cell response wrt. pos=(x,y) as gc(pos)
    """
    rot_theta = 60  # degrees
    relative_R = rotation_matrix(rot_theta)
    init_R = rotation_matrix(orientation_offset, degrees)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    k2 = relative_R @ k1  # rotate k1 by 60degrees using R
    k3 = relative_R @ k2  # rotate k2 by 60degrees using R
    ks = np.array([k1, k2, k3])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)
    ks *= f  # user-defined spatial frequency

    def wrapper(r):
        # shape of r: (n**2,n**2,2)
        L = np.sqrt(r.shape[0])
        return np.sum(np.cos((r @ ks.T) / L), axis=-1)

    return wrapper


class CANN:
    def __init__(self, Ng=4096, Np=512, nonlinearity="relu", **kwargs):
        self.Ng, self.Np = Ng, Np  # Ng should be a quadratic number
        n = int(np.sqrt(Ng))
        # shape: (n,n,2)
        self.neural_sheet_2d = np.stack(
            np.meshgrid(np.linspace(0, n - 1, n), np.linspace(0, n - 1, n)), axis=-1
        )
        # shape: (n**2,2)
        self.neural_sheet_1d = self.neural_sheet_2d.reshape(
            np.prod(self.neural_sheet_2d.shape[:-1]), self.neural_sheet_2d.shape[-1]
        )
        # shape: (n**2,2)
        self.M = self._init_input_weights(self.neural_sheet_1d)

        self.grid_cell_fn = grid_cell(**kwargs)

    def _init_input_weights(self, neural_sheet_1d):
        """
        Sorscher initialization - NESW, i.e. cardinal diretions

        M can be considered Beta in Sorscher (shift-matrix or smt).
        """
        M = (neural_sheet_1d[:, ::-1] % 2) * (-1) ** neural_sheet_1d
        return M  # shape=(n**2,2)

    def _init_recurrent_weights(self, neural_sheet_1d, beta, grid_cell_fn):
        # (n**2,2,1) - (1,2,n**2) => (n**2,2,n**2)
        old = neural_sheet_1d[..., None] - neural_sheet_1d.T[None]

        # (n**2,1,2) - (1,n**2,2) - (1,n**2,2)=> (n**2,n**2,2)
        W = neural_sheet_1d[:, None] - neural_sheet_1d[None] - beta[None]

        # CAN DELETE OLD AND PRINT STUFF IF CORRECT
        print(
            "_init_recurrent_weights methods are same if zero: ", np.sum(abs(old - W))
        )
        return grid_cell_fn(W)

    def _init_readout_weights(self):
        pass

    def forward(self, N):
        for i in range(N):
            # ...step
            continue
        pass

    def _step(self):
        pass

    def prune_mask():
        """Set a prune mask"""
        pass
