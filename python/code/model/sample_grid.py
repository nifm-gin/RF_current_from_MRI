from abc import ABC, abstractmethod
import math
import numpy as np
from utils.utils import scale_linear_by_column
from scipy.stats import qmc


class SampleGrid(ABC):
    """Abstract base class for a sample grid"""

    @abstractmethod
    def make_grid(self):
        pass


class SampleGrid1D(SampleGrid):
    """Class defining a sample grid for a one dimension simulation"""

    def __init__(
        self,
        lambd: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        t2_star: np.ndarray,
        lambda_gradient: np.ndarray,
        signal_sign: np.ndarray,
        size: int = None,
        distribution: str = "grid",
        subgrid_dimensions: tuple = (),
        dtype: type = float,
        samplegrid_mask_keep: bool = False,
        rng_stream: np.random.Generator = None,
    ):
        """Sample grid constructor

        Args:
            lambd (np.ndarray): vector of lambda values. For "sobol" and "rand" distributions: min and max
            t1 (np.ndarray): vector of t1 values. For "sobol" and "rand" distributions: min and max
            t2 (np.ndarray): vector t2 values. For "sobol" and "rand" distributions: min and max
            t2_star (np.ndarray): vector of t2* values. For "sobol" and "rand" distributions: min and max
            lambda_gradient (np.ndarray): vector of lambda_gradient values. For "sobol" and "rand" distributions: min and max
            signal_sign (np.ndarray): Must contain only -1 or 1. For "sobol" and "rand" distributions: [-1.0, 1.0]
            size (int, optional): n_sim for "rand" and "sobol" distribution
            distribution (str, optional): Distribution of the parameter - "sobol", "rand", "grid" or "raw". Defaults to "grid".
            subgrid_dimensions (tuple, optional): Relaxation parameters the MR signal depends on. No acceleration if empty
            dtype (type, optional): dtype of samplegrid. Defaults to float.
            samplegrid_mask_keep: keep or remove voxels in mask
            rng_stream (np.random.Generator): pseudo-random number generator used to generate random grid (can be None)
        """

        lambd = np.array(lambd).astype(dtype)
        t1 = np.array(t1).astype(dtype)
        t2 = np.array(t2).astype(dtype)
        t2_star = np.array(t2_star).astype(dtype)
        lambda_gradient = np.array(lambda_gradient).astype(dtype)
        signal_sign = np.array(signal_sign).astype(dtype)

        self.distribution = distribution
        if self.distribution == "grid":
            t1 = np.unique(t1)
            t2 = np.unique(t2)
            t2_star = np.unique(t2_star)
            lambda_gradient = np.unique(lambda_gradient)
            signal_sign = np.unique(signal_sign)
            lambd = np.unique(lambd)

        if not hasattr(self, "grid"):
            self.grid = dict()

        self.grid["t1"] = t1
        self.grid["t2"] = t2
        self.grid["t2_star"] = t2_star
        self.grid["lambda_gradient"] = lambda_gradient
        self.grid["signal_sign"] = signal_sign
        self.grid["lambda"] = lambd
        self.grid["b1_phase"] = np.array([0.0]).astype(dtype)
        self.grid["b1_phase_c"] = np.cos(self.grid["b1_phase"]) + 1j * np.sin(self.grid["b1_phase"])
        self.n_vox_os = 1

        # define the order of dimensions in the grid
        self.dimension_order = np.array([*self.grid])
        self.n_vox = 1
        self.size = size
        self.dtype = dtype
        self.samplegrid_mask_keep = True

        self.n = dict()
        self.dim = dict()
        self.grid_size = np.ndarray([0])
        self.n_sim = None
        self.grid_vec = dict()
        self.mask = np.full((1, self.n_vox,), False)

        self.has_subgrid = False
        self.subgrid_dimensions = ()
        self.subgrid_vec = None
        self.subgrid_idx = None
        self.subgrid_size = 0
        self.samplegrid_mask_keep = samplegrid_mask_keep

        if rng_stream is None:
            self.rng_stream = np.random.default_rng()
        else:
            self.rng_stream = rng_stream

        self.make_grid()
        self.make_subgrid(subgrid_dimensions)

    def make_grid(self):
        """  Generate parameter grid

        Raises:
            NotImplementedError: If distributions are not ["sobol", "rand", "raw", "grid"]
        """

        sign_choice = [1, -1]
        dtype = self.dtype

        if self.distribution == "grid":
            # create grid from the individual parameter vectors
            grid_vec = np.meshgrid(*[*self.grid.values()], indexing="ij")
            for i, key in enumerate(self.grid):
                self.grid_vec[key] = grid_vec[i].ravel()

            for grid_dim in self.dimension_order:
                self.n[grid_dim] = int(self.grid[grid_dim].size)

            for grid_dim in self.dimension_order:
                self.dim[grid_dim] = np.where(self.dimension_order == grid_dim)[0][0]

        elif self.distribution == "sobol":
            sampler = qmc.Sobol(len(self.grid), scramble=True, seed=self.rng_stream)
            seq = sampler.random(n=self.size).astype(self.dtype)
            for i, key in enumerate(self.grid):
                if key == "signal_sign":
                    if -1 in self.grid[key] and 1 in self.grid[key]:
                        self.grid_vec[key] = ((seq[:, i] < 0.5).astype("int") * 2 - 1).astype(dtype)
                    else:
                        self.grid_vec[key] = np.repeat(self.grid[key][0], self.size)
                else:
                    if self.grid[key].size == 1:
                        self.grid_vec[key] = np.repeat(self.grid[key][0], self.size)
                    else:
                        self.grid_vec[key] = scale_linear_by_column(
                            seq[:, i], self.grid[key][1], self.grid[key][0]
                        ).astype(dtype)

            for grid_dim in self.dimension_order:
                self.n[grid_dim] = self.size

            for grid_dim in self.dimension_order:
                self.dim[grid_dim] = np.where(self.dimension_order == grid_dim)[0][0]

        elif self.distribution == "rand":
            for i, key in enumerate(self.grid):
                if key == "signal_sign":
                    if -1 in self.grid[key] and 1 in self.grid[key]:
                        self.grid_vec[key] = self.rng_stream.choice(sign_choice, self.size).astype(dtype)
                    else:
                        self.grid_vec[key] = np.repeat(self.grid[key][0], self.size)
                else:
                    if self.grid[key].size == 1:
                        self.grid_vec[key] = np.repeat(self.grid[key][0], self.size)
                    else:
                        self.grid_vec[key] = self.rng_stream.uniform(
                            self.grid[key][0], self.grid[key][1], self.size
                        ).astype(dtype)

            for grid_dim in self.dimension_order:
                self.n[grid_dim] = self.size

            for grid_dim in self.dimension_order:
                self.dim[grid_dim] = np.where(self.dimension_order == grid_dim)[0][0]

        elif self.distribution == "raw":
            for i, key in enumerate(self.grid):
                if not np.iscomplexobj(self.grid[key]):
                    this_dtype = dtype
                else:
                    this_dtype = (1j * np.zeros(1).astype(dtype)).dtype  # derive equivalent complex type from dtype
                if self.grid[key].size == 1:
                    self.grid_vec[key] = np.repeat(np.reshape(self.grid[key], 1)[0], self.size).astype(this_dtype)
                else:
                    self.grid_vec[key] = self.grid[key].astype(this_dtype)

            for grid_dim in self.dimension_order:
                self.n[grid_dim] = self.size

            for grid_dim in self.dimension_order:
                self.dim[grid_dim] = np.where(self.dimension_order == grid_dim)[0][0]

        else:
            raise NotImplementedError(f"Distribution '{self.distribution}' is unknown")

        self.grid_size = np.array(self.grid_vec["t1"].shape)
        self.n_sim = np.prod(self.grid_size)

    def modify_grid(self, dict):
        """ store new values for the selected parameter
        """
        for parameter, values in dict.items():
            self.grid_vec[parameter] = self.rng_stream.choice(values, size=self.grid_vec[parameter].size, replace=True)
            # update lambda if necessary
            if parameter in ["I", "phj", "lambda_b", "lambda_b_dx", "lambda_b_dy", "xij", "thj", "r0", "ph0", "phb"]:
                self.grid_vec["lambda"], self.grid_vec["b1_phase"] = self.simulate_lambda()
                self.grid_vec["b1_phase"] = - self.grid_vec["b1_phase"]
                self.grid_vec["b1_phase_c"] = np.cos(self.grid_vec["b1_phase"]) + 1j * np.sin(self.grid_vec["b1_phase"])
        self.make_subgrid(self.subgrid_dimensions)

    def make_subgrid(self, subgrid_dimensions):
        """ make subgrid in order to test only discrete values for chosen parameters
        """
        if len(subgrid_dimensions) > 0:
            self.subgrid_dimensions = subgrid_dimensions
            subgrid_values = np.column_stack(tuple(self.grid_vec[key] for key in subgrid_dimensions))
            subgrid_unique, recon_indices = np.unique(subgrid_values, axis=0, return_inverse=True)
            self.subgrid_vec = dict()
            for i, key in enumerate(self.subgrid_dimensions):
                self.subgrid_vec[key] = subgrid_unique[:, i]
            self.subgrid_idx = recon_indices
            self.subgrid_size = subgrid_unique.shape[0]
            self.has_subgrid = True
        return


class SampleGrid2D(SampleGrid1D):
    """Class defining a sample grid for a two dimensions simulation"""

    def __init__(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        t2_star: np.ndarray,
        signal_sign: np.ndarray,
        I: np.ndarray,
        phj: np.ndarray,
        lambda_b: np.ndarray,
        lambda_b_dx: np.ndarray,
        lambda_b_dy: np.ndarray,
        xij: np.ndarray,
        thj: np.ndarray,
        r0: np.ndarray = np.array([0.0]),
        ph0: np.ndarray = np.array([0.0]),
        phb: np.ndarray = np.array([0.0]),
        mask_radius: float = np.nan,
        cfac: float = 0.5,
        mu_r: float = 1.0,
        fov: np.ndarray = np.array([0.024, 0.024]),
        image_size: np.ndarray = np.array([12, 12]),
        oversampling: int = 1,
        size: int = None,
        distribution: str = "grid",
        subgrid_dimensions: tuple = ("t1", "t2", "t2_star",),
        dtype: type = float,
        samplegrid_mask_keep: bool = False,
        rng_stream: np.random.Generator = None,
    ):
        """Constructor of SampleGrid2D

        Args:
            t1 (np.ndarray): vector of t1 values. For "sobol" and "rand" distributions: min and max
            t2 (np.ndarray): vector t2 values. For "sobol" and "rand" distributions: min and max
            t2_star (np.ndarray): vector of t2* values. For "sobol" and "rand" distributions: min and max
            signal_sign (np.ndarray): Must contain only -1 or 1. For "sobol" and "rand" distributions: [-1.0, 1.0]
            I (np.ndarray): vector of current amplitudes. For "sobol" and "rand" distributions: min and max
            phj (np.ndarray): vector of current phases. For "sobol" and "rand" distributions: min and max
            lambda_b (np.ndarray): vector of background lambdas constant coefficients. For "sobol" and "rand" distributions: min and max
            lambda_b_dx (np.ndarray): vector of background lambdas x coefficients. For "sobol" and "rand" distributions: min and max
            lambda_b_dy (np.ndarray): vector of background lambdas y coefficients. For "sobol" and "rand" distributions: min and max
            xij (np.ndarray): vector of electrode angulations. For "sobol" and "rand" distributions: min and max
            thj (np.ndarray): vector of electrode azimuths. For "sobol" and "rand" distributions: min and max
            r0 (np.ndarray): vector of electrode offset distances (mm). For "sobol" and "rand" distributions: min and max
            ph0 (np.ndarray): vector of electrode offset angles (rad). For "sobol" and "rand" distributions: min and max
            mask_radius (float): radius in mm for masking the center of the patch. Determined from x0,y0 if not given
            cfac (float, optional): Fudge factor on current, defaults to 0.5
            mu_r (float, optional): relative sample permittivity, defaults to 1
            fov (np.ndarray): xy field of view in m, defaults to [0.024, 0.024]
            image_size (np.ndarray): xy number of voxels, defaults to [12, 12]
            oversampling (int): spatial oversampling in model, default 1
            size (int, optional): n_sim for "rand" and "sobol" distribution, default None
            distribution (str, optional): Distribution of the parameter - "sobol", "rand", "grid" or "raw". Defaults to "rand".
            subgrid_dimensions (tuple, optional): Relaxation parameters the MR signal depends on. No acceleration if empty
            samplegrid_mask_keep: keep or remove voxels in mask
            rng_stream (np.random.Generator): pseudo-random number generator used to generate random grid (can be None)
        """

        I = np.array(I).astype(dtype)
        phj = np.array(phj).astype(dtype)
        lambda_b = np.array(lambda_b).astype(dtype)
        lambda_b_dx = np.array(lambda_b_dx).astype(dtype)
        lambda_b_dy = np.array(lambda_b_dy).astype(dtype)
        xij = np.array(xij).astype(dtype)
        thj = np.array(thj).astype(dtype)
        r0 = np.array(r0).astype(dtype)
        ph0 = np.array(ph0).astype(dtype)
        phb = np.array(phb).astype(dtype)
        mu_r = np.array(mu_r).astype(dtype)

        self.distribution = distribution
        if self.distribution == "grid":
            I = np.unique(I)
            phj = np.unique(phj)
            lambda_b = np.unique(lambda_b)
            lambda_b_dx = np.unique(lambda_b_dx)
            lambda_b_dy = np.unique(lambda_b_dy)
            xij = np.unique(xij)
            thj = np.unique(thj)
            r0 = np.unique(r0)
            ph0 = np.unique(ph0)
            phb = np.unique(phb)

        if not hasattr(self, "grid"):
            self.grid = dict()
        self.grid["I"] = I
        self.grid["phj"] = phj
        self.grid["lambda_b"] = lambda_b
        self.grid["lambda_b_dx"] = lambda_b_dx
        self.grid["lambda_b_dy"] = lambda_b_dy
        self.grid["xij"] = xij
        self.grid["thj"] = thj
        self.grid["r0"] = r0
        self.grid["ph0"] = ph0
        self.grid["phb"] = phb

        # Now call superclass init, which also calls make_grid
        super(SampleGrid2D, self).__init__(
            lambd=np.array([0.0]),
            t1=t1,
            t2=t2,
            t2_star=t2_star,
            lambda_gradient=np.array([0.0]),
            signal_sign=signal_sign,
            size=size,
            distribution=distribution,
            subgrid_dimensions=subgrid_dimensions,
            dtype=dtype,
            rng_stream=rng_stream,
        )

        self.mu_r = mu_r
        self.mu_0 = 4 * math.pi * 1e-7
        if np.isnan(mask_radius):  # calculate the required masking radius from the position variability
            self.mask_radius = 2e-3 + np.max(self.grid_vec["r0"])
        else:
            self.mask_radius = mask_radius
        self.samplegrid_mask_keep = samplegrid_mask_keep

        # calculate positions of voxel centers for the high-resolution (oversampled) grid
        os = round(oversampling)
        n_vox_os = image_size * os
        dxy_os = fov / n_vox_os
        xylim_os = (fov - dxy_os) / 2
        [x_os, y_os] = np.meshgrid(
            np.linspace(-xylim_os[0], xylim_os[0], n_vox_os[0], endpoint=True, dtype=dtype),
            np.linspace(-xylim_os[1], xylim_os[1], n_vox_os[1], endpoint=True, dtype=dtype), indexing="ij"
        )

        # generate mask for the normal ("low resolution") grid
        dxy = fov / image_size
        xylim = (fov - dxy) / 2
        [x, y] = np.meshgrid(
            np.linspace(-xylim[0], xylim[0], image_size[0], endpoint=True, dtype=dtype),
            np.linspace(-xylim[1], xylim[1], image_size[1], endpoint=True, dtype=dtype),
        )
        mask = (np.sqrt(x ** 2 + y ** 2) < self.mask_radius)
        keep = np.where(np.logical_not(mask.flatten()))[0]

        # reorder and reduce this to speed up subsequent processing
        x_os = np.reshape(x_os, (image_size[0], os, image_size[1], os))
        y_os = np.reshape(y_os, (image_size[0], os, image_size[1], os))
        x_os = np.swapaxes(x_os, 1, 2).reshape((-1, os * os))  # This is now size (n_vox, os**2)
        y_os = np.swapaxes(y_os, 1, 2).reshape((-1, os * os))  # This is now size (n_vox, os**2)
        if self.samplegrid_mask_keep:
            x_os = x_os.reshape((1, -1))
            y_os = y_os.reshape((1, -1))
        else:
            x_os = x_os[keep, :].reshape((1, -1))  # This is now size (1, sum(mask) * os**2)
            y_os = y_os[keep, :].reshape((1, -1))  # This is now size (1, sum(mask) * os**2)

        self.oversampling = os
        self.image_size = image_size
        self.fov = fov
        self.n_vox_os = x_os.size  # This is the number of voxels with oversampling, after masking
        if self.samplegrid_mask_keep:
            self.n_vox = int(x_os.size / os ** 2)
        else:
            self.n_vox = keep.size  # This is the number of voxels without oversampling, after masking
        self.x = x_os
        self.y = y_os
        self.mask = mask.reshape((1, -1))

        self.grid_vec["lambda"], self.grid_vec["b1_phase"] = self.simulate_lambda()
        self.grid_vec["b1_phase"] = - self.grid_vec["b1_phase"]
        self.grid_vec["b1_phase_c"] = np.cos(self.grid_vec["b1_phase"]) + 1j * np.sin(self.grid_vec["b1_phase"])
        # self.grid_vec["b1_phase"] = self.grid_vec["b1_phase"]

    def simulate_lambda(self):
        """ Simulate lambda in the acquisition windows. 
        """
        # b.I, b.lambda_b, b.thj and b.xij are row vectors (N_sim x 1) and b.r, b.thr are column vectors (1 x N_vox)
        # b1tot has a size of N_sim x N_vox
        # Values in the s.grid_vec dictionary should all have size N_sim x 1.
        # Returned signal has size N_sim x N_vox
        b1ref = 3.1e-6

        n_sim = self.grid_vec["I"].size
        I = self.grid_vec["I"].reshape((n_sim, 1))
        phj = self.grid_vec["phj"].reshape((n_sim, 1))
        lambda_b = self.grid_vec["lambda_b"].reshape((n_sim, 1)).real
        lambda_b_dx = self.grid_vec["lambda_b_dx"].reshape((n_sim, 1)).real
        lambda_b_dy = self.grid_vec["lambda_b_dy"].reshape((n_sim, 1)).real
        xij = self.grid_vec["xij"].reshape((n_sim, 1))
        thj = self.grid_vec["thj"].reshape((n_sim, 1))
        r0 = self.grid_vec["r0"].reshape((n_sim, 1))
        ph0 = self.grid_vec["ph0"].reshape((n_sim, 1))
        phb = self.grid_vec["phb"].reshape((n_sim, 1))

        # Calculate wire positions in x and y
        x0 = r0 * np.cos(ph0)
        y0 = r0 * np.sin(ph0)
        # Calculate distances in x and y between voxel centers and the center of the wire
        dx = self.x - x0
        dy = self.y - y0
        dr = np.sqrt(dx ** 2 + dy ** 2)  # True distance of a voxel from the wire
        dr[dr < self.mask_radius] = np.nan  # This masks the output as it gets propagated into lambda_vec and lambda_pha

        s_thr = dy / dr
        c_thr = dx / dr

        lambda_b_map = lambda_b + lambda_b_dx * dx + lambda_b_dy * dy
        del dx
        del dy

        s_phj = np.sin(phj)
        c_phj = np.cos(phj)
        s_thj = np.sin(thj)
        c_thj = np.cos(thj)
        s_xij = np.sin(xij)
        c_xij = np.cos(xij)

        # For the other calculations, order operands by shape to first combine operands of identical shapes to avoid
        # broadcasting at each operation
        angulation_normalization = dr * (1 - ((s_xij * c_thj) * c_thr + (s_xij * s_thj) * s_thr) ** 2.0)
        del dr, c_thj, s_thj, s_xij
        with np.errstate(divide='ignore', invalid='ignore'):  # We have sometimes warnings here we want to ignore
            b1j_norm = (self.mu_0 * self.mu_r / (2 * math.pi)) * (I * c_xij / (lambda_b * b1ref)) / angulation_normalization
        del angulation_normalization, c_xij

        b1j_sin = b1j_norm * (s_phj * c_thr - c_phj * s_thr)
        b1j_cos = b1j_norm * (c_phj * c_thr + s_phj * s_thr)
        del s_thr
        del c_thr
        del s_phj, c_phj

        sqrt_argument = 1 + b1j_sin + b1j_norm ** 2 / 4
        del b1j_norm
        if np.any(sqrt_argument < 0):
            sqrt_argument[sqrt_argument < 0] = 0  # This is happening due to numerical imprecision

        lambda_vec = lambda_b_map * np.sqrt(sqrt_argument)
        del sqrt_argument
        lambda_pha = - phb + np.arctan2(b1j_cos, 2 + b1j_sin)
        del b1j_cos
        del b1j_sin

        # Handle sign in lambda_vec due to extreme values of lambda_b_dx/dy correctly
        lambda_pha[lambda_vec < 0.0] = lambda_pha[lambda_vec < 0.0] + np.pi
        lambda_vec = np.abs(lambda_vec)
        lambda_vec = lambda_vec.astype(self.dtype)
        lambda_pha = lambda_pha.astype(self.dtype)

        return lambda_vec, lambda_pha

