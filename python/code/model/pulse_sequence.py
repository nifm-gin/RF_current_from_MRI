import math
import numpy as np
import h5py
import time

from model import (
    AcquisitionParameters,
    SampleGrid1D,
    afi_precalc_1alpha,
    afi_precalc_2alpha,
)
from utils.utils import get_transpose_vector


# slice profile AFI (with flip angles and phases dico)
def afi_precalc(a, s, i_seq, two_dim, slice_profile_dict, dtype: type = float, slice_profile_dict_type=None, slice_profile_dict_signal=None):
    """
        Calculate (da-hdr)AFI MRI signals

        Parameters
        ----------
        a: AcquisitionParameters
            Acquisition parameters of the sequence to be simulated.
        s: SampleGrid1D or SampleGrid1D
            Grid of sample-dependent parameters on which MRI signal will be simulated.
        i_seq: int
            Index of the sequence whose signal will be simulated.
        two_dim: bool
            True for using SampleGrid2D, False for using SampleGrid1D
        slice_profile_dict: string
            containing the file name of the dictionary of slice profiles
            (dictionary of mx-mz for sequence parameters optimization.
        slice_profile_dict_type: string
            Indicates which type of dictionary to use to simulate the signal.
            "signal" dictionary that contain directly signal for current fitting from a known sequence ;
            "mx_mz" dictionary that contain mx-mz for sequence parameters optimization
        slice_profile_dict_signal: list of string
            list of loaded sliceprofile signal dictionary for current fitting

        Returns
        -------
        dict
            contain "s1" and "s2" the two signals of (da-hdr)AFI sequence
        """

    print_execution_times = False

    start = time.time()
    if s.has_subgrid:
        t1 = s.subgrid_vec["t1"]
        idx_subgrid = s.subgrid_idx
    else:
        t1 = s.grid_vec["t1"]
        idx_subgrid = np.arange(s.grid_size)

    e1 = np.exp(-a.tr1 / t1)
    e2 = np.exp(-a.tr2 / t1)
    te = np.exp(-a.te / s.grid_vec["t2_star"])
    ss = s.grid_vec["signal_sign"]
    if two_dim:
        e1 = e1[..., np.newaxis]
        e2 = e2[..., np.newaxis]
        te = te[..., np.newaxis]
        ss = ss[..., np.newaxis]
    if slice_profile_dict_type == "mx_mz":
        if slice_profile_dict is None:
            if print_execution_times:
                print("afi prep: ", time.time() - start, flush=True)
                start = time.time()

            if a.alpha1 == a.alpha2:
                a1 = a.alpha1 * s.grid_vec["lambda"]
                ca = np.cos(a1)  # Using a single call to np.exp(1j * a1) is MUCH slower
                sa = np.sin(a1)
                del a1

                e1e2 = e1 * e2
                sf = sa / (1 - e1e2[idx_subgrid, ...] * ca**2)
                del sa
                signal_1 = sf * (
                    (1 - e2)[idx_subgrid, ...] + (e2 - e1e2)[idx_subgrid, ...] * ca
                )
                signal_2 = sf * (
                    (1 - e1)[idx_subgrid, ...] + (e1 - e1e2)[idx_subgrid, ...] * ca
                )
                del ca
            else:
                a1 = a.alpha1 * s.grid_vec["lambda"]
                ca1 = np.cos(a1)  # Using a single call to np.exp(1j * a1) is MUCH slower
                sa1 = np.sin(a1)
                del a1
                a2 = a.alpha2 * s.grid_vec["lambda"]
                ca2 = np.cos(a2)
                sa2 = np.sin(a2)
                del a2

                e1e2 = e1 * e2
                sf = 1 / (1 - e1e2[idx_subgrid, ...] * (ca1 * ca2))
                signal_1 = (
                    sa1
                    * sf
                    * ((1 - e2)[idx_subgrid, ...] + (e2 - e1e2)[idx_subgrid, ...] * ca2)
                )
                del sa1, ca2
                signal_2 = (
                    sa2
                    * sf
                    * ((1 - e1)[idx_subgrid, ...] + (e1 - e1e2)[idx_subgrid, ...] * ca1)
                )
                del sa2, ca1
            del sf, e1, e2, e1e2

            if print_execution_times:
                print("afi precalc: ", time.time() - start, flush=True)
                start = time.time()
        else:
            # Add third dimension for slice profile
            e1 = e1[..., np.newaxis]
            e2 = e2[..., np.newaxis]
            idx_subgrid = idx_subgrid[
                ..., np.newaxis
            ]  # Add second dimension needed in afi_precalc

            idx_alpha_nom_1 = slice_profile_dict["get_idx_alpha_nom"](a.alpha1)
            idx_real_1 = slice_profile_dict["get_idx_alpha_real"](
                a.alpha1 * s.grid_vec["lambda"]
            )
            sa1 = slice_profile_dict["dict_mx"][[idx_alpha_nom_1]]
            ca1 = slice_profile_dict["dict_mz"][[idx_alpha_nom_1]]
            extent = slice_profile_dict["extent"]
            if print_execution_times:
                print("afi prep: ", time.time() - start, flush=True)
                start = time.time()

            if a.alpha1 == a.alpha2:
                signal_1, signal_2 = afi_precalc_1alpha(e1, e2, sa1, ca1, extent)
                idx_real_2 = idx_real_1
            else:
                idx_alpha_nom_2 = slice_profile_dict["get_idx_alpha_nom"](a.alpha2)
                idx_real_2 = slice_profile_dict["get_idx_alpha_real"](
                    a.alpha2 * s.grid_vec["lambda"]
                )
                sa2 = slice_profile_dict["dict_mx"][[idx_alpha_nom_2]]
                ca2 = slice_profile_dict["dict_mz"][[idx_alpha_nom_2]]

                signal_1, signal_2 = afi_precalc_2alpha(e1, e2, sa1, sa2, ca1, ca2, extent)

            if print_execution_times:
                print("afi precalc: ", time.time() - start, flush=True)
                start = time.time()

            # Now perform the indexing to put into each voxel the signal of the corresponding flip angle
            signal_1 = signal_1[idx_subgrid, idx_real_1]
            signal_2 = signal_2[idx_subgrid, idx_real_2]
            if print_execution_times:
                print("afi lookup: ", time.time() - start, flush=True)
                start = time.time()
    elif slice_profile_dict_type == "signal":
        slice_profile_dict_signal = slice_profile_dict_signal[i_seq]
        alphas_1 = a.alpha1 * s.grid_vec["lambda"]
        all_alphas_dict = slice_profile_dict_signal["flip_angles1"] / 180 * np.pi
        max_all_alphas = np.max(np.array(all_alphas_dict))
        alphas_1[alphas_1 > max_all_alphas] = max_all_alphas
        alphas_1[alphas_1 < 0] = 0
        if np.any(np.isnan(alphas_1)):
            mask_nan = np.isnan(alphas_1)
            alphas_1[mask_nan] = 0
        else:
            mask_nan = None
        idx_real_1 = slice_profile_dict_signal["get_idx_alpha_real"](alphas_1)
        signal_1 = slice_profile_dict_signal["real_afi_sim"][idx_real_1, 0]
        signal_2 = slice_profile_dict_signal["real_afi_sim"][idx_real_1, 1]
        if mask_nan is not None:
            signal_1[mask_nan] = np.nan
            signal_2[mask_nan] = np.nan
    # Perform global scaling with exp(-te/t2star) and signal_sign
    if not (slice_profile_dict_type == "signal"):
        signal_1 = ((te * ss) * signal_1).astype(dtype)
        signal_2 = ((te * ss) * signal_2).astype(dtype)
    if print_execution_times:
        print("afi scaling: ", time.time() - start, flush=True)
    return {"s1": signal_1, "s2": signal_2}


# specify AFI signal equations for signals from first and second TR
def sig1_afi(pc):  # FID signal w/ T1, T2
    """
        Extract first signal of the (da-hdr)AFI sequence

        Parameters
        ----------
        pc: dict
            dictionary calculated by afi_precalc

        Returns
        -------
        np.ndarray
            array of the first signal of (da-hdr)AFI sequence
        """
    return pc["s1"]


def sig2_afi(pc):
    """
        Extract first signal of the (da-hdr)AFI sequence

        Parameters
        ----------
        pc: dict
            dictionary calculated by afi_precalc

        Returns
        -------
        np.ndarray
            array of the first signal of (da-hdr)AFI sequence
    """
    return pc["s2"]


class PulseSequence:
    """Class defining a pulse sequence"""

    def __init__(
        self,
        sequence: str = None,
        min_n=None,
        max_n=None,
        min_alpha2=None,
        max_alpha2=None,
        min_tr=None,
        n_phase_encode_lines_afi=None,
        abort_if_out_of_bounds=None,
        rf_energy_limit=None,
        acquisition_time_limit=None,
        min_beta=None,
        max_beta=None,
        alpha_rf_energy_factor=None,
        n_phase_encode_lines_dream=None,
        acquisition_params=None,
        rf_energy=None,
        acquisition_time=None,
        is_checked=False,
        n_phase_encode_lines_spgr=None,
        dual_alpha: bool = False,
        repeat_acquisition: int = 1,
        params_out_of_bounds: bool = False,
        two_dimension: bool = False,
        slice_profile_dict=None,
        slice_profile_dict_type="mx_mz",
        slice_profile_dict_signal=None,
        n_workers=1,
    ):
        """Constructor of a pulse sequence

        Args:
            sequence (str, optional): Type of sequence, "afi" ("dream" is not supported yet). Defaults to None.
            min_n ([type], optional):  Minimum value of TR ratio in AFI. Defaults to None.
            max_n ([type], optional): Maximum value of TR ratio in AFI. Defaults to None.
            min_alpha2 ([type], optional): Minimal acceptable flip angle for alpha2 in degrees. Defaults to None.
            max_alpha2 ([type], optional):  Maximal acceptable flip angle for alpha2 in degrees. Defaults to None.
            min_tr ([type], optional):Minimum value of TR in AFI (in milliseconds). Ignored for DREAM. Defaults to None.
            n_phase_encode_lines_afi ([type], optional): [description]. Defaults to None.
            abort_if_out_of_bounds ([type], optional): If present and true, check sequence will not correct the sequence
                parameters, but abort if specified RF energy and acquisition time limits are not met. Defaults to None.
            rf_energy_limit ([type], optional): If present, the flip angles of the last acquisition will be adjusted to
                meet the specified RF energy limitation (arbitrary units, see code for scaling). Defaults to None.
            acquisition_time_limit ([type], optional): If present, the TR(s) of the last acquisition will be adjusted to
                meet the specified total acqisition time limitation (milliseconds). Defaults to None.
            min_beta ([type], optional): Minimal acceptable flip angle for alpha2 in degrees. Defaults to None.
            max_beta ([type], optional): Maximal acceptable flip angle for alpha2 in degrees. Defaults to None.
            alpha_rf_energy_factor ([type], optional): Relative energy of alpha to beta pulses in DREAM. Default: None.
            n_phase_encode_lines_dream ([type], optional):  # of phase-encoding lines for DREAM sequence. Default: None.
            acquisition_params ([type], optional): See Acquisition_parameters class. Defaults to None.
            rf_energy ([type], optional):  RF energy of the sequence (arbitrary units). Defaults to None.
            acquisition_time ([type], optional): Acquisition time (ms). Defaults to None.
            n_phase_encode_lines_spgr ([type], optional):  # of phase-encoding lines for SPGR sequence. Default: None.
            is_checked (bool, optional): Flag indicating if the pulse sequence has been checked. The check verifies that
                RFenergy and acquisition time limits are met and calculates any missing infos. Defaults to False.
            dual_alpha (bool, optional): [description]. Defaults to False.
            repeat_acquisition (int, optional): [description]. Defaults to 1.
            params_out_of_bounds (bool, optional): [description]. Defaults to False.
            two_dimension (bool, optional): Flag indicating if the simulation is in one dimension or in two dimensions.
                Defaults to False.
            slice_profile_dict (string): containing the file name of the dictionary of slice profiles
            slice_profile_dict_type: "signal" ; "mx_mz"
            slice_profile_dict_signal: list of loaded sliceprofile signal dictionary for current_from_acquired_data
            n_workers (int, optional): number of workers for signal simulation
        """

        self.sequence = sequence
        self.min_n = min_n
        self.max_n = max_n
        self.min_alpha2 = min_alpha2
        self.max_alpha2 = max_alpha2
        self.min_tr = min_tr
        self.n_phase_encode_lines_afi = n_phase_encode_lines_afi
        self.abort_if_out_of_bounds = abort_if_out_of_bounds
        self.rf_energy_limit = rf_energy_limit
        self.acquisition_time_limit = acquisition_time_limit
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.alpha_rf_energy_factor = alpha_rf_energy_factor
        self.n_phase_encode_lines_dream = n_phase_encode_lines_dream
        self.n_phase_encode_lines_spgr = n_phase_encode_lines_spgr
        self.acquisition_params = acquisition_params
        if self.acquisition_params is None:
            self.acquisition_params = []
        self.rf_energy = rf_energy
        self.acquisition_time = acquisition_time
        self.is_checked = is_checked
        self.dual_alpha = dual_alpha
        self.repeat_acquisition = repeat_acquisition
        self.cost = 0
        self.n_signals = None
        self.n_sequences = None
        self.params_out_of_bounds = params_out_of_bounds
        self.two_dimension = two_dimension
        self.n_monte_carlo = 1
        self.slice_profile_dict = slice_profile_dict
        self.slice_profile_dict_type = slice_profile_dict_type
        self.slice_profile_dict_signal = slice_profile_dict_signal
        self.n_workers = n_workers
        self.print_execution_times = False

    def set_sequence_parameters(self, parameter_vector: list, te: np.array):
        if type(te) is not np.ndarray:
            te = np.array(te).flatten()
        if str.lower(self.sequence) == "afi":
            end = len(parameter_vector)
            if self.dual_alpha:
                quarter = int(end / 4)
                flip1 = parameter_vector[:quarter]
                flip2 = parameter_vector[quarter : 2 * quarter]
                tr1 = parameter_vector[2 * quarter : 3 * quarter]
                n = parameter_vector[3 * quarter :]
            else:
                third = int(end / 3)
                flip1 = parameter_vector[:third]
                flip2 = parameter_vector[:third]
                tr1 = parameter_vector[third : 2 * third]
                n = parameter_vector[2 * third :]

            n_sequences = len(flip2)
            for iSeq in range(n_sequences):
                acq = AcquisitionParameters()
                acq.flip1 = flip1[iSeq]
                acq.flip2 = flip2[iSeq]
                acq.tr1 = tr1[iSeq]
                acq.n = n[iSeq]
                if te.size == n_sequences:
                    acq.te = te[iSeq]
                else:
                    acq.te = te[0]
                self.acquisition_params.append(acq)
        else:
            print("Unknown sequence type. Aborting", flush=True)

        self.initialize_sequence()

    def fix_acquisition_time(
        self, i_seq, acquisition_time, repeat_acquisition, this_tr1
    ):
        """ Calculate and set tr2 to obtain a fixed acquisition time.
        """
        this_tr2 = (self.acquisition_time_limit - acquisition_time) / (
            repeat_acquisition * self.n_phase_encode_lines_afi
        ) - this_tr1
        # in case the acquisition time requested can not be
        # realized, choose the next best feasible value and
        # penalize the solution with additional math.cost later
        if (
            math.isnan(this_tr2)
            or this_tr2 < this_tr1 * self.min_n
            or this_tr2 < self.min_tr
        ):
            this_tr2 = max(this_tr1 * self.min_n, self.min_tr)
            self.params_out_of_bounds = True

        if this_tr2 > this_tr1 * self.max_n:
            this_tr2 = this_tr1 * self.max_n
            self.params_out_of_bounds = True

        self.acquisition_params[i_seq].n = this_tr2 / this_tr1

    def initialize_sequence(self):
        """ Check the sequence to be in the chosen bounds.
        """
        if self.is_checked:
            print("Sequence already checked", flush=True)
        else:
            n_sequences = len(self.acquisition_params)
            self.n_sequences = n_sequences

            # Process TRs and check acquisition times if necessary
            acquisition_time = 0

            if self.repeat_acquisition:
                repeat_acquisition = self.repeat_acquisition
            else:
                repeat_acquisition = 1
                self.repeat_acquisition = 1

            for i_seq in range(0, n_sequences):
                fix_acquisition_time = (
                    i_seq == n_sequences - 1
                ) and self.acquisition_time_limit is not None
                if str.lower(self.sequence) == "afi":
                    this_tr1 = self.acquisition_params[i_seq].tr1
                    if not np.isfinite(this_tr1) or this_tr1 < self.min_tr:
                        this_tr1 = self.min_tr
                        self.params_out_of_bounds = True

                    if fix_acquisition_time:
                        self.fix_acquisition_time(
                            i_seq,
                            acquisition_time,
                            repeat_acquisition,
                            this_tr1,
                        )

                    this_tr2 = this_tr1 * self.acquisition_params[i_seq].n

                    # store the (updated) TR2 for later use
                    self.acquisition_params[i_seq].tr1 = this_tr1
                    self.acquisition_params[i_seq].tr2 = this_tr2
                    this_acquisition_time = repeat_acquisition * (
                        self.n_phase_encode_lines_afi * (this_tr1 + this_tr2)
                    )

                    self.cost += sum(
                        10
                        * np.log(
                            1
                            + np.maximum(
                                self.min_tr - np.array([this_tr1, this_tr2]), 0
                            )
                            / self.min_tr
                        )
                    )

                    for i in range(2 * repeat_acquisition):
                        if i % 2 == 0:
                            self.acquisition_params[i_seq].signal_functions.append(
                                sig1_afi
                            )
                        else:
                            self.acquisition_params[i_seq].signal_functions.append(
                                sig2_afi
                            )
                    self.acquisition_params[i_seq].n_signals = 2 * repeat_acquisition
                else:
                    raise (
                        RuntimeError(
                            f"Sequence of type {self.sequence} is not supported"
                        )
                    )
                acquisition_time = acquisition_time + this_acquisition_time
                # Make sure we don't discard a sequence due to numerical
                # imprecision: round to the nearest millisecond
                acquisition_time = round(acquisition_time)
                if fix_acquisition_time:
                    # Process flip angles and check RF energy if necessary
                    self.cost += 10 * math.log(
                        1
                        + max(acquisition_time - self.acquisition_time_limit, 0)
                        / self.acquisition_time_limit
                    )
                    if acquisition_time > self.acquisition_time_limit:
                        self.params_out_of_bounds = True

            self.acquisition_time = acquisition_time

            # Calculate total number of signals in all pulse sequences
            self.n_signals = np.sum(
                [ac.n_signals for ac in self.acquisition_params]
            ).astype(int)

            # Process flip angles and check RF energy if necessary
            rf_energy = 0
            for i_seq in range(0, n_sequences):
                fix_energy = (i_seq == n_sequences) and self.rf_energy_limit
                this_rf_energy = 0
                requested_rf_energy = 0
                if str.lower(self.sequence) == "afi":
                    this_alpha1 = self.acquisition_params[i_seq].flip1 * math.pi / 180
                    this_alpha2 = self.acquisition_params[i_seq].flip2 * math.pi / 180
                    if fix_energy:
                        requested_rf_energy = self.rf_energy_limit
                        if (
                            self.acquisition_params[i_seq].dualAlpha
                            and self.acquisition_params[i_seq].dualAlpha
                        ):
                            this_alpha2 = math.sqrt(
                                (requested_rf_energy - rf_energy)
                                / (repeat_acquisition * self.n_phase_encode_lines_afi)
                                - this_alpha1
                                ^ 2
                            )
                        else:
                            this_alpha2 = math.sqrt(
                                (requested_rf_energy - rf_energy)
                                / (
                                    2
                                    * repeat_acquisition
                                    * self.n_phase_encode_lines_afi
                                )
                            )
                            this_alpha1 = this_alpha2

                    # Check that values are legal.
                    # In case the RF energy requested can not be realized,
                    # choose the next best feasible value and penalize the
                    # solution with additional cost later
                    if (
                        not np.isreal(this_alpha1)
                        or not math.isfinite(this_alpha1)
                        or this_alpha1 < self.min_alpha2 * math.pi / 180
                    ):
                        this_alpha1 = self.min_alpha2 * math.pi / 180
                        self.params_out_of_bounds = True

                    if this_alpha1 > self.max_alpha2 * math.pi / 180:
                        this_alpha1 = self.max_alpha2 * math.pi / 180
                        self.params_out_of_bounds = True

                    if (
                        not np.isreal(this_alpha2)
                        or not math.isfinite(this_alpha2)
                        or this_alpha2 < self.min_alpha2 * math.pi / 180
                    ):
                        this_alpha2 = self.min_alpha2 * math.pi / 180
                        self.params_out_of_bounds = True

                    if this_alpha2 > self.max_alpha2 * math.pi / 180:
                        this_alpha2 = self.max_alpha2 * math.pi / 180
                        self.params_out_of_bounds = True

                    # store the (updated) flip angles in radians for later use:
                    self.acquisition_params[i_seq].alpha1 = this_alpha1
                    self.acquisition_params[i_seq].alpha2 = this_alpha2
                    this_rf_energy = (
                        repeat_acquisition
                        * self.n_phase_encode_lines_afi
                        * (this_alpha1**2 + this_alpha2**2)
                    )
                    self.acquisition_params[i_seq].alpha1 = this_alpha1
                rf_energy = rf_energy + this_rf_energy
                if fix_energy:
                    self.cost += 10 * math.log(
                        1
                        + max(rf_energy - requested_rf_energy, 0) / requested_rf_energy
                    )
                    if rf_energy > requested_rf_energy:
                        self.params_out_of_bounds = True

            self.rf_energy = rf_energy

            if self.abort_if_out_of_bounds and self.params_out_of_bounds:
                self.is_checked = False
            else:
                self.is_checked = True

    def simulate_signal(self, sample_grid: SampleGrid1D):
        """ Simulate MRI signal from self and a SampleGrid.
        """
        if not self.is_checked:
            raise RuntimeError(
                "Sequence needs to be checked before using it to simulate signal."
            )

        start = time.time()
        # get number of entries in the final dictionary
        sim_sizes = sample_grid.grid_size
        n_sim = sample_grid.n_sim
        sim_sizes_orig = sim_sizes
        n_sim_orig = n_sim
        dtype = sample_grid.dtype

        if np.count_nonzero(sample_grid.grid_vec["lambda_gradient"]) > 0:
            # check that we are working on a regular grid
            if sample_grid.distribution != "grid":
                print(
                    "Lambda gradients can currently only be calculated from a regular grid.",
                    flush=True,
                )
                raise NotImplementedError()

            do_simulate_gradient = True

            # to simulate gradients, first, simulate raw signals for gradient zero
            sim_sizes = sim_sizes_orig.copy()
            sim_sizes[0] = sim_sizes[0] / len(sample_grid.grid["lambda_gradient"])
            n_sim = np.prod(sim_sizes)
            grid_vec_orig = sample_grid.grid_vec.copy()

            keep_idx = np.where(sample_grid.grid_vec["lambda_gradient"] == 0.0)
            for grid_dim in sample_grid.dimension_order:
                sample_grid.grid_vec[grid_dim] = sample_grid.grid_vec[grid_dim][
                    keep_idx
                ]
        else:
            do_simulate_gradient = False

        acquisition_params = self.acquisition_params
        n_sequences = len(acquisition_params)
        n_signals = self.n_signals
        if not self.two_dimension:
            mr_signal = np.zeros((n_sim, n_signals), dtype=dtype)
        else:
            mr_signal = np.zeros((n_sim, sample_grid.n_vox_os, n_signals), dtype=dtype)
        # signal at each MC iteration
        # in the 2D case, mr_signal = np.zeros((n_sim*n_vox, n_signals))
        # Find out if we need to tile the signal. Calculated signals will have size of sample_grid.grid_vec["lambda"]
        if not self.two_dimension:
            signal_repeats = sim_sizes // sample_grid.grid_vec["lambda"].shape
        else:
            signal_repeats = sim_sizes // sample_grid.grid_vec["lambda"].shape[0]
        i_sig = 0
        if self.print_execution_times:
            print("sim sig prep: ", time.time() - start, flush=True)

        start = time.time()
        for i_seq in range(n_sequences):
            this_acquisition_params = acquisition_params[i_seq]
            this_n_signals = this_acquisition_params.n_signals

            if str.lower(self.sequence) == "afi":
                # FID signal w/ TR1, TR2
                this_start = time.time()
                pc = afi_precalc(
                    this_acquisition_params,
                    sample_grid,
                    i_seq,
                    self.two_dimension,
                    self.slice_profile_dict,
                    dtype=dtype,
                    slice_profile_dict_type=self.slice_profile_dict_type,
                    slice_profile_dict_signal=self.slice_profile_dict_signal
                )
                if self.print_execution_times:
                    print("time whole pc: ", time.time() - this_start, flush=True)
                for iThisSig in range(this_n_signals):
                    # expand signal to the expected size
                    if not self.two_dimension:
                        if signal_repeats != (1,):
                            mr_signal[:, i_sig] = np.tile(
                                this_acquisition_params.signal_functions[iThisSig](pc),
                                signal_repeats,
                            )
                        else:
                            mr_signal[
                                :, i_sig
                            ] = this_acquisition_params.signal_functions[iThisSig](pc)
                    else:
                        if signal_repeats != (1,):
                            mr_signal[:, :, i_sig] = np.tile(
                                this_acquisition_params.signal_functions[iThisSig](pc),
                                signal_repeats,
                            )
                        else:
                            mr_signal[
                                :, :, i_sig
                            ] = this_acquisition_params.signal_functions[iThisSig](pc)
                    i_sig += 1
                del pc
            else:
                for iThisSig in range(this_n_signals):
                    # expand signal to the expected size
                    if not self.two_dimension:
                        if signal_repeats != (1,):
                            mr_signal[:, i_sig] = np.tile(
                                this_acquisition_params.signal_functions[iThisSig](
                                    this_acquisition_params,
                                    sample_grid,
                                    self.two_dimension,
                                    dtype=dtype,
                                ),
                                signal_repeats,
                            )
                        else:
                            mr_signal[
                                :, i_sig
                            ] = this_acquisition_params.signal_functions[iThisSig](
                                this_acquisition_params,
                                sample_grid,
                                self.two_dimension,
                                dtype=dtype,
                            )
                    else:
                        if signal_repeats != (1,):
                            mr_signal[:, :, i_sig] = np.tile(
                                this_acquisition_params.signal_functions[iThisSig](
                                    this_acquisition_params,
                                    sample_grid,
                                    self.two_dimension,
                                    dtype=dtype,
                                ),
                                signal_repeats,
                            )
                        else:
                            mr_signal[
                                :, :, i_sig
                            ] = this_acquisition_params.signal_functions[iThisSig](
                                this_acquisition_params, sample_grid
                            )
                    i_sig += 1

        if self.print_execution_times:
            print("sim sig sim: ", time.time() - start, flush=True)

        start = time.time()
        # In the 2D case, mr_signal will be reshaped to size [n_sim, n_vox * n_sig]
        rf_phase = sample_grid.grid_vec["b1_phase_c"][..., np.newaxis]
        if self.two_dimension:
            if sample_grid.oversampling != 1:  # Need to down-sample the signals here
                os = sample_grid.oversampling
                mr_signal = (
                    mr_signal * rf_phase
                )  # add B1 transmit phase from RF current and background B1 to signal
                mr_signal = mr_signal.reshape(
                    (-1, os**2, n_signals)
                )  # separate oversampling dimension from the others
                mr_signal = np.einsum("ijk->ik", mr_signal) / (
                    os**2
                )  # average signal in each voxel across subvoxels
            mr_signal = mr_signal.reshape((n_sim, sample_grid.n_vox * n_signals))
        else:
            rf_phase = np.tile(
                rf_phase.reshape((-1, 1)), (1, sample_grid.n_vox * n_signals)
            )
            mr_signal = mr_signal * rf_phase
        del rf_phase
        if self.print_execution_times:
            print("sim sig phase: ", time.time() - start, flush=True)

        if do_simulate_gradient:
            sample_grid.grid_vec = grid_vec_orig

        # # Apply maks from grid
        # if sample_grid.samplegrid_mask_keep:
        #     mask = np.tile(sample_grid.mask[:, :, np.newaxis], (1, 1, n_signals,)).reshape((sample_grid.n_vox * n_signals,))
        #     if self.two_dimension:
        #         mr_signal[:, mask] = 0
        return mr_signal

    def get_ref_signal(self):
        n_signals = self.n_signals

        # sequence parameters to determine reference signal used to
        # calculate noise amplitude as a function of SNR

        if str.lower(self.sequence) == "afi":
            reference_sequence = PulseSequence(
                sequence=self.sequence,
                min_n=1,
                max_n=39,
                min_alpha2=10,
                max_alpha2=360,
                min_tr=1,
                n_phase_encode_lines_afi=self.n_phase_encode_lines_afi,
                abort_if_out_of_bounds=True,
                dual_alpha=False,
                repeat_acquisition=1,
            )
            reference_sequence.set_sequence_parameters([45, 8.3, 5], te=3.0)
        else:
            raise RuntimeError(f"Sequence Type {self.sequence} is unknown")

        reference_grid = SampleGrid1D(
            lambd=np.array([1.0]),
            t1=np.array([1000.0]),
            t2=np.array([70.0]),
            t2_star=np.array([40]),
            lambda_gradient=np.array([0]),
            signal_sign=np.array([1]),
            distribution="grid",
        )

        # simulate reference signal
        reference_signal = reference_sequence.simulate_signal(reference_grid)
        reference_signal = np.abs(reference_signal[0][0])

        reference_signal = np.tile(reference_signal, n_signals)
        return np.abs(reference_signal)

    def add_noise(
        self,
        signal=None,
        sample_grid: SampleGrid1D = None,
        n_monte_carlo=800,
        snr: float = 50,
        add_phase_noise: bool = True,
        random_seed=False,
        imported_noise_file=None,
    ):
        self.n_monte_carlo = n_monte_carlo
        # Simulate data
        n_lambda_sim = sample_grid.n["lambda"]
        n_t1_sim = sample_grid.n["t1"]
        n_grad_sim = sample_grid.n["lambda_gradient"]
        n_sequences = self.n_sequences
        n_sim = sample_grid.n_sim
        dtype = sample_grid.dtype

        if self.two_dimension:
            n_signals = self.n_signals * sample_grid.n_vox
        else:
            n_signals = self.n_signals

        size_signal = np.hstack(([1], [n_sim], [n_signals]))
        size_noise = np.hstack(([n_monte_carlo], [n_sim], [n_signals]))
        size_phase = np.hstack(([n_monte_carlo], [n_sim], [1]))
        signal_dim_vec = (1, 1, n_signals)
        phase_range_b1 = 2 * math.pi
        phase_range_b0 = 0.5 * math.pi

        # Transform signal to complex type, preserving float representation
        signal = np.reshape(signal, size_signal) + 1j * 0.0

        noise_amplitude = self.get_ref_signal() / snr
        if self.two_dimension:
            noise_amplitude = np.tile(noise_amplitude, sample_grid.n_vox)
        noise_amplitude = np.reshape(noise_amplitude, signal_dim_vec).astype(dtype)

        if random_seed is None or isinstance(random_seed, np.integer):
            rng = np.random.default_rng(random_seed)
        elif isinstance(random_seed, (np.random.RandomState, np.random.Generator)):
            rng = random_seed
        else:
            raise ValueError(f'{random_seed!r} cannot be used to random_seed a'
                             ' numpy.random.Generator instance')

        if imported_noise_file is None:
            # prepare noise and random phases
            if add_phase_noise:
                phase_noise = rng.random(size_phase, dtype=dtype) * phase_range_b1
                if str.lower(self.sequence) == "dream":
                    noise_b0 = (
                        2 * rng.random(size_phase, dtype=dtype) - 1
                    ) * phase_range_b0
                    phase_noise = np.tile(phase_noise, signal_dim_vec)
                    i_sig = 1
                    for iSeq in range(1, n_sequences):
                        this_n_signals = self.acquisition_params[iSeq].n_signals
                        this_repeat = self.acquisition_params[iSeq].repeat_acquisition
                        this_signals = i_sig + 2 * np.arange(0, this_repeat - 1)
                        phase_noise[:, :, this_signals] += noise_b0
                        phase_noise[:, :, this_signals + 1] -= noise_b0
                        i_sig = i_sig + this_n_signals

                # Apply the phase noise here
                signal = signal * np.exp(1j * phase_noise)

            # Add real part of the noise
            signal = (
                signal + rng.standard_normal(size_noise, dtype=dtype) * noise_amplitude
            )
            # Add imaginary part of the noise
            signal = signal + rng.standard_normal(size_noise, dtype=dtype) * (
                1j * noise_amplitude
            )
        else:
            # Pre-calculate the transpose and reshape vectors for Matlab data
            size_matlab = [
                n_monte_carlo,
                n_lambda_sim,
                n_t1_sim,
                n_grad_sim,
                n_signals,
            ]
            # Add any additional dimensions we have in our grid at the end
            dim_in = [
                "monte_carlo",
                "lambda",
                "t1",
                "lambda_gradient",
                "signals",
            ]
            dim_out = list(
                np.hstack((["monte_carlo"], sample_grid.dimension_order, ["signals"]))
            )
            [t_vec, ex] = get_transpose_vector(dim_in, dim_out)

            with h5py.File(imported_noise_file, "r+") as f:
                # read noise from file. Dimensions are all backwards from
                # Matlab code.
                if add_phase_noise:
                    phase_noise = np.transpose(np.array(f["NOISE_B1"]))
                    # Now bring these matrices into the right shape for our grid
                    phase_noise = np.expand_dims(phase_noise, axis=ex).transpose(t_vec)
                    phase_noise = phase_noise.reshape(size_phase)
                    phase_noise *= phase_range_b1

                    if str.lower(self.sequence) == "dream":
                        noise_b0 = np.transpose(np.array(f["NOISE_B0"]))
                        noise_b0 = np.expand_dims(noise_b0, axis=ex).transpose(t_vec)
                        noise_b0 = noise_b0.reshape(size_phase)
                        noise_b0 = (2 * noise_b0 - 1) * phase_range_b0
                        i_sig = 1
                        for iSeq in range(1, n_sequences):
                            this_n_signals = self.acquisition_params[iSeq].n_signals
                            this_repeat = self.acquisition_params[
                                iSeq
                            ].repeat_acquisition
                            this_signals = i_sig + 2 * np.arange(0, this_repeat - 1)
                            phase_noise[:, :, this_signals] += noise_b0
                            phase_noise[:, :, this_signals + 1] -= noise_b0
                            i_sig = i_sig + this_n_signals

                    # Apply the phase noise here
                    signal = signal * np.exp(1j * phase_noise)

                noise = np.transpose(np.array(f["NOISE_REAL"])).astype(dtype=dtype)
                # Verify dimensions
                if np.any(np.array(noise.shape) != size_matlab):
                    raise RuntimeError(
                        f"File {imported_noise_file} contains data of incompatible size."
                    )
                noise = np.expand_dims(noise, axis=ex).transpose(t_vec)
                noise = noise.reshape(size_noise)
                # Add real part of the noise
                signal = signal + noise * noise_amplitude

                noise = np.transpose(np.array(f["NOISE_IMAG"])).astype(dtype=dtype)
                noise = np.expand_dims(noise, axis=ex).transpose(t_vec)
                noise = noise.reshape(size_noise)
                # Add imaginary part of the noise
                signal = signal + noise * (1j * noise_amplitude)

        return signal


def remove_phase_afi(this_s, dimension: int):
    """Determine and remove common phase of signals"""

    # fit phase using total least squares (orthogonal distance regression)
    this_real = this_s.real
    this_imag = this_s.imag
    this_realimag = np.sum(this_real * this_imag, axis=dimension, keepdims=True)
    this_realimag[this_realimag == 0] = 1
    this_b = (
        (1 / 2)
        * np.sum(this_real**2 - this_imag**2, axis=dimension, keepdims=True)
        / this_realimag
    )

    # There are two slopes that are extrema of the orthogonal distance, the
    # minimum and the maximum. Right now we don't know which is which:
    this_sqrtb2p1 = np.sqrt(this_b**2 + 1)
    this_slope_a = -this_b + this_sqrtb2p1
    this_slope_b = -this_b - this_sqrtb2p1
    # Compare orthogonal distances for both solutions and find for each lambda
    # value the one that is lower

    this_orth_da = np.sum(
        (this_imag - this_slope_a * this_real) ** 2,
        axis=dimension,
        keepdims=True,
    ) / (1 + this_slope_a**2)
    this_orth_db = np.sum(
        (this_imag - this_slope_b * this_real) ** 2,
        axis=dimension,
        keepdims=True,
    ) / (1 + this_slope_b**2)

    try:
        a_is_better = np.array(this_orth_da < this_orth_db).astype(int)
    except:
        print(np.all(np.isfinite(this_orth_da)), flush=True)
        print(np.all(np.isfinite(this_orth_db)), flush=True)
        print(np.all(np.isfinite(this_slope_a)), flush=True)
        print(np.all(np.isfinite(this_slope_b)), flush=True)
        print(np.all(np.isfinite(this_real)), flush=True)
        print(np.all(np.isfinite(this_imag)), flush=True)
        a_is_better = np.zeros_like(this_orth_da).astype(int)

    this_slope = np.choose(a_is_better, [this_slope_b, this_slope_a])

    # this_phase = np.arctan(this_slope)
    # exp_i_phi = np.exp(1j * this_phase)
    # this_s = this_s / exp_i_phi
    # return [this_s, this_phase]

    exp_i_phi = (1 + 1j * this_slope) / np.sqrt(1 + this_slope**2)
    this_s = this_s / exp_i_phi

    return this_s, exp_i_phi
