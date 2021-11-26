from typing import Callable
import numpy as np


def rotation_matrix(theta, degrees=True, **kwargs) -> np.ndarray:
    """
    Creates a 2D rotation matrix for theta

    Parameters
    ----------
    theta : float
        angle offset wrt. the cardinal x-axis
    degrees : boolean
        Whether to use degrees or radians

    Returns
    -------
    rotmat : np.ndarray
        the 2x2 rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones(2) / np.sqrt(2)
    >>> rotmat = rotation_matrix(45)
    >>> tmp = rotmat @ x
    >>> eps = 1e-8
    >>> np.sum(np.abs(tmp - np.array([0., 1.]))) < eps 
    True
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def grid_cell(phase_offset=0, orientation_offset=0, f=1, **kwargs) -> Callable:
    """
    Grid cell pattern constructed from three interacting 2D (plane) vectors
    with 60 degrees relative orientational offsets.
    See e.g. the paper: "From grid cells to place cells: A mathematical model"
    - Moser & Einevoll 2005

    Parameters
    ----------
    phase_offset : float
        First plane vector is default along the cardinal x-axis. Phase-offset
        turns this plane vector counter clockwise (default in degrees, but
        can use **kwargs - degrees=False to use radians)
    orientation_offset : np.ndarray
        2D-array. Spatial (vector) phase offset of the grid pattern. Note that
        the grid pattern is np.array([f,f])-periodic, so phase-offsets are also
        f-periodic
    f : float
        Spatial frequency / periodicity. f=1 makes the grid cell unit-periodic

    Returns
    -------
    grid_cell_fn : function
        A grid cell function which can be evaluated at locations r

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(2)
    >>> gc = grid_cell()
    >>> gc(x)
    3.0
    """
    rot_theta = 60  # degrees
    relative_R = rotation_matrix(rot_theta)
    init_R = rotation_matrix(orientation_offset, **kwargs)

    k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
    k1 = init_R @ k1
    k2 = relative_R @ k1  # rotate k1 by 60degrees using R
    k3 = relative_R @ k2  # rotate k2 by 60degrees using R
    ks = np.array([k1, k2, k3])
    ks *= 2 * np.pi  # spatial angular frequency (unit-movement in space is one period)
    ks *= f  # user-defined spatial frequency

    def grid_cell_fn(r):
        # shape of r: (n**2,n**2,2)
        # return shape: (n**2,n**2)
        return np.sum(np.cos(r @ ks.T), axis=-1)

    return grid_cell_fn


class CANN:
    def __init__(self, Ng=4096, Np=512, nonlinearity="relu", Wp_mask=128, **kwargs):
        self.Ng, self.Np = Ng, Np  # Ng should be a quadratic number
        self.L = int(np.sqrt(Ng))
        self.Wp_mask = Wp_mask  # number of grid cell - place cell connections
        if self.L ** 2 != Ng:
            raise UserWarning(f"Non-square Ng not supported: {self.L**2 != Ng=}")

        # Create neural sheets (just imagine: topographically arranged cells)
        # shape: (n,n,2)
        self.neural_sheet_2d = np.stack(
            np.meshgrid(
                np.linspace(0, self.L - 1, self.L), np.linspace(0, self.L - 1, self.L)
            ),
            axis=-1,
        )
        # shape: (n**2,2)
        self.neural_sheet_1d = self.neural_sheet_2d.reshape(
            np.prod(self.neural_sheet_2d.shape[:-1]), self.neural_sheet_2d.shape[-1]
        )

        # set nonlinearity from argument
        if nonlinearity is None:
            self.nonlinearity = lambda x: x  # identity/no activation
        elif nonlinearity == "relu":
            self.nonlinearity = lambda x: x * (x > 0)  # ReLu activation
        else:
            raise NotImplementedError(f"{nonlinearity=} not implemented")

        # initialize weights
        self.grid_cell_fn = grid_cell(**kwargs)
        # shape: (n**2,2)
        self.M = self._init_input_weights(self.neural_sheet_1d)
        # shape: (n**2,n**2)
        self.J = self._init_recurrent_weights(
            self.neural_sheet_1d, self.M, self.grid_cell_fn
        )
        # shape: (n**2)
        self.Wp = self._init_readout_weights()
        self.bias = np.zeros(self.Ng)

    def _init_input_weights(self, neural_sheet_1d) -> np.ndarray:
        """
        Initialise input weights

        Follows Sorschers initialization - NESW, i.e. cardinal directions.
        M can be considered Beta in Sorscher (shift-matrix or smt).

        Parameters
        ----------
        neural_sheet_1d : np.ndarray
            2D-array of shape (Ng,2). 1D-representation of a 2D neural sheet
            of the topographical arrangement of neurons.

        Returns
        -------
        M : np.ndarray
            2D-array of shape (Ng,2)
        """
        M = (neural_sheet_1d[:, ::-1] % 2) * (-1) ** neural_sheet_1d
        return M

    def _init_recurrent_weights(self, neural_sheet_1d, beta, grid_cell_fn):
        # (n**2,1,2) - (1,n**2,2) - (1,n**2,2)=> (n**2,n**2,2)
        W = neural_sheet_1d[:, None] - neural_sheet_1d[None] - beta[None]
        W /= self.L  # relate periodicity to the length of the neural sheet
        orientation_offsets = 2 * np.pi * np.random.random(self.Ng)
        W = np.array(
            [
                grid_cell(orientation_offset=orientation_offsets[i], degrees=False)(
                    W[i]
                )
                for i in range(self.Ng)
            ]
        )
        return W
        # return grid_cell_fn(W)

    def _init_readout_weights(self):
        Wp = np.zeros((self.Np, self.Ng))
        for i in range(self.Np):
            connectivity_idxs = np.random.choice(
                np.arange(self.Ng), size=self.Wp_mask, replace=False
            )
            Wp[i, connectivity_idxs] = 1
        return Wp

    def g(self, h0, vs) -> np.ndarray:
        """
        Iterate dynamics for a set of velocities (vs) and an intial state (h_0)

        Parameters
        ----------
        h0 : np.ndarray
            1D-array of length Ng. initial state of system
        vs : np.ndarray
            2D-array of shape (T,2)

        Returns
        -------
        hn : np.ndarray
            1D-array of shape (Ng). of the state of the system after all
            velocities (vs) have been sequentially applied
        """
        hn = h0
        for v in vs:
            hn = self.nonlinearity(self.J @ hn + self.J @ (self.M @ v) + self.bias)
            hn /= np.sum(hn)
        return hn

    def p(self, g_inputs):
        return self.Wp @ g_inputs

    def forward(self, h0, vs):
        gs = self.g(h0, vs)
        return self.p(gs)

    def prune_mask():
        """Set a prune mask"""
        pass
