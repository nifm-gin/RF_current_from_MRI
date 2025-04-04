import numpy as np
from os.path import exists
from scipy.io import loadmat
import os


def get_data_directory():
    data_dir = os.getenv('RF_CURRENT_FROM_MRI_DATA')
    if data_dir is None:
        home_dir = os.getenv('HOME')
        data_dir = os.path.join(home_dir, "rf_current_from_mri_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_transpose_vector(dimensions_in: list, dimensions_out: list):
    """
    Provide a transposition vector usable by numpy.transpose() to
    convert a numpy ndarray between dimension orders specified by the
    inputs. Usage:
    [transpose_vec, expand_vec] = get_transpose_vector(dim_in, dim_out)
    data = np.expand_dims(data_in, axis=expand_vec).transpose(transpose_vec)

    Parameters
    ----------
    dimensions_in : list of str
        List of dimension names in the order of the input data. Must contain
        only dimension names also present in dimensions_out.
    dimensions_out : list of str
        List of dimension names in the order of the output data. Dimensions
        that only exist in the output will be expanded as singleton dimensions
        from the input.

    Returns
    -------
    transpose_vec : tuple of int
        Axis reordering that can be passed to numpy.transpose
    expand_vec : tuple of int
        Vector to be passed to numpy.expand_dims to expand dimensions

    Raises
    ------
    KeyError
        when dimensions_in contains a string not present in dimensions_out
    """

    if np.any([d not in dimensions_out for d in dimensions_in]):
        raise KeyError(f"Input list contains entries not present in output")

    # list of dimensions to add to the input list
    new_dims = list(set(dimensions_out) - set(dimensions_in))
    # create dictionary translating dimension names to input position
    dict_in = dict(
        zip(
            np.hstack((dimensions_in, new_dims)),
            np.arange(len(dimensions_out)),
        )
    )
    transpose_vec = [dict_in[d] for d in dimensions_out]
    expand_vec = np.arange(len(dimensions_in), len(dimensions_out))

    return [tuple(transpose_vec), tuple(expand_vec)]


def scale_linear_by_column(raw_points, high=1, low=0, minima=None, maxima=None):
    if not minima:
        minima = np.min(raw_points, axis=0)
    if not maxima:
        maxima = np.max(raw_points, axis=0)
    rng = maxima - minima
    return high - (((high - low) * (maxima - raw_points)) / rng)


def pass_train():
    pass


def condition_signal(signal, n_samples, n_vox, n_signals, options=None):
    from model import (
        remove_phase_afi,
    )
    """Normalize both acquired and simulated MRI signals"""

    if options is None:
        options = ProcessingOptions()

    # Choose direction along which to remove phase
    # conditioning_option must be either 1, 2 or 3:
    #  - 1 means we normalize and remove 1 phase across all voxels and signals,
    #  - 2 means we normalize and remove n_seq phases across all voxels and signals of each sequence
    #  - 3 means we normalize and remove n_vox phases across signals, individually for each voxel
    if options.model_conditioning_option == 1:
        conditioning_dimension = 1
        signal = signal.reshape((n_samples, n_vox * n_signals))
    elif options.model_conditioning_option == 2:
        conditioning_dimension = 1
        n_sig_per_seq = 2
        signal = signal.reshape(
            (n_samples, n_vox, n_signals // n_sig_per_seq, n_sig_per_seq)
        )
        signal = signal.swapaxes(2, 3)
        signal = signal.reshape(
            (n_samples, n_vox * n_sig_per_seq, n_signals // n_sig_per_seq)
        )
    elif options.model_conditioning_option == 3:
        conditioning_dimension = 2
        signal = signal.reshape((n_samples, n_vox, n_signals))
    else:
        raise NotImplementedError

    # remove phase along the specified conditioning dimension
    # Since fitting the background phase we may not need this anymore
    if not options.model_fit_background_b1_phase:
        signal, _ = remove_phase_afi(signal, conditioning_dimension)
        _ = np.angle(_).reshape(
            (n_samples, options.seq_image_size[0], options.seq_image_size[1])
        )
        __unwrapped = np.zeros(np.shape(_))
        for i_sample in range(n_samples):
            __unwrapped[i_sample, :, :] = unwrap(2 * _[i_sample, :, :])
        diff = np.exp(1j * (_ - __unwrapped / 2))
        sign = diff.reshape((n_samples, n_vox, 1)).real
        signal = sign * signal

    if options.model_fit == "R":
        signal = signal.real
    elif options.model_fit == "M":
        signal = np.abs(signal)
    elif options.model_fit == "I":
        signal = signal.imag
    elif options.model_fit == "P":
        signal = np.angle(signal)

    if options.model_conditioning_option != 1:
        # Normalize signal in the specified conditioning dimension
        signal_norm = np.sqrt(
            np.sum(np.abs(signal) ** 2, axis=conditioning_dimension, keepdims=True)
        )
        signal_norm[signal_norm == 0] = 1
        signal = signal / signal_norm

        # Reshape to original dimension
        if options.model_conditioning_option == 2:
            signal = signal.reshape((n_samples, n_vox, 2, n_signals // 2))
            signal = signal.swapaxes(2, 3)
        signal = signal.reshape((n_samples, n_vox * n_signals))

    # Normalize the signal of the entire slice to have comparable amplitudes for all normalization options
    if options.global_signal_scaling and n_vox > 1:
        # Make this insensitive to NaN values. Use np.nanmean to be insensitive to the number of voxels masked.
        # Multiply by n_samples to have the same scale as the previous np.sum
        signal_norm = np.sqrt(n_samples * np.nanmean(np.abs(signal) ** 2, axis=1, keepdims=True))
        signal_norm[signal_norm == 0] = 1.0
        signal = signal / signal_norm

    if options.model_fit == "C":
        signal = np.concatenate(
            (signal.real[:, :, np.newaxis], signal.imag[:, :, np.newaxis]), axis=2
        )
        # signal = np.concatenate((np.abs(signal[:, :, np.newaxis]), np.angle(signal[:, :, np.newaxis])), axis=2)
        signal = signal.reshape((n_samples, n_vox * n_signals * 2))
    return signal.real


def read_reco_params_file(file_path):
    """ read fitted parameters from .txt file obtained with current_from_acquired data"""
    file = open(file_path, "r")
    v = file.read()
    file.close()
    I = v[v.find('"I":') + 6 : v.find('"phj":') - 3].split(", ")
    phj = v[v.find('"phj":') + 8 : v.find('"lambda_b":') - 3].split(", ")
    lambda_b = v[v.find('"lambda_b":') + 13 : v.find('"lambda_b_dx":') - 3].split(", ")
    lambda_b_dx = v[v.find('"lambda_b_dx":') + 16 : v.find('"lambda_b_dy":') - 3].split(
        ", "
    )
    lambda_b_dy = v[v.find('"lambda_b_dy":') + 16 : v.find('"xij":') - 3].split(", ")
    xij = v[v.find('"xij":') + 8 : v.find('"thj":') - 3].split(", ")
    thj = v[v.find('"thj":') + 8 : v.find('"r0":') - 3].split(", ")
    r0 = v[v.find('"r0":') + 7 : v.find('"ph0":') - 3].split(", ")
    ph0 = v[v.find('"ph0":') + 8 : v.find('"phb":') - 3].split(", ")
    phb = v[v.find('"phb":') + 8 : v.find('"t1":') - 3].split(", ")
    for k in range(len(I)):
        I[k] = float(I[k])
        phj[k] = float(phj[k])
        lambda_b[k] = float(lambda_b[k])
        lambda_b_dx[k] = float(lambda_b_dx[k])
        lambda_b_dy[k] = float(lambda_b_dy[k])
        xij[k] = float(xij[k])
        thj[k] = float(thj[k])
        r0[k] = float(r0[k])
        ph0[k] = float(ph0[k])
        phb[k] = float(phb[k])
    return (
        np.array(I),
        np.array(phj),
        np.array(lambda_b),
        np.array(lambda_b_dx),
        np.array(lambda_b_dy),
        np.array(phj),
        np.array(thj),
        np.array(r0),
        np.array(ph0),
        np.array(phb),
    )


def load_slice_profile_dict(slice_profile_dict_name, dtype, dict_type="mx_mz"):
    """ load slice profile dictionary ("mx-mz" for sequence optimization and "signal" for current fitting)
    """
    if exists(slice_profile_dict_name):
        if dict_type == "mx_mz":
            slice_profile_dict = loadmat(slice_profile_dict_name)
            slice_profile_dict["dict_mx"] = slice_profile_dict["dict_mx"].astype(dtype)
            slice_profile_dict["dict_mz"] = slice_profile_dict["dict_mz"].astype(dtype)
            slice_profile_dict["extent"] = slice_profile_dict["extent"].astype(dtype)
            alpha_nom = slice_profile_dict["nominal_flip"].squeeze() * (
                np.pi / 180.0
            )  # use rad everywhere
            max_idx_nom = alpha_nom.size - 1
            alpha_nom_min, alpha_nom_max = np.min(alpha_nom), np.max(alpha_nom)
            alpha_nom_step = (alpha_nom_max - alpha_nom_min) / max_idx_nom

            slice_profile_dict["get_idx_alpha_nom"] = lambda alpha: np.minimum(
                np.maximum(np.round((alpha - alpha_nom_min) / alpha_nom_step), 0),
                max_idx_nom,
            ).astype(int)
            alpha_real = slice_profile_dict["flip_angles"].squeeze() * (
                np.pi / 180.0
            )  # use rad everywhere
            max_idx_real = alpha_real.size - 1
            alpha_real_min, alpha_real_max = np.min(alpha_real), np.max(alpha_real)
            alpha_real_step = (alpha_real_max - alpha_real_min) / max_idx_real
            slice_profile_dict["get_idx_alpha_real"] = lambda alpha: np.minimum(
                np.maximum(np.round((alpha - alpha_real_min) / alpha_real_step), 0),
                max_idx_real,
            ).astype(int)
        elif dict_type == "signal":
            slice_profile_dict = loadmat(slice_profile_dict_name)
            slice_profile_dict["signal"] = slice_profile_dict["real_afi_sim"].astype(dtype)
            alpha_real = slice_profile_dict["flip_angles1"].squeeze() * (
                    np.pi / 180.0
            )  # use rad everywhere
            max_idx_real = alpha_real.size - 1
            alpha_real_min, alpha_real_max = np.min(alpha_real), np.max(alpha_real)
            alpha_real_step = (alpha_real_max - alpha_real_min) / max_idx_real
            slice_profile_dict["get_idx_alpha_real"] = lambda alpha: np.minimum(
                np.maximum(np.round((alpha - alpha_real_min) / alpha_real_step), 0),
                max_idx_real,
            ).astype(int)
    else:
        slice_profile_dict = None
    return slice_profile_dict


class ProcessingOptions:
    """Class aimed to store processing options all along the data processing"""
    def __init__(
        self,
        model_fit="M",
        model_conditioning_option=3,
        global_signal_scaling=True,
        model_fit_background_b1_phase=True,
        model_do_noisy_learning=False,
        model_fit_learning_rate=False,
        nn_layers=(300, 300, 50, 50, 50, 20, 20, 20, 5, 5, 5, 5, 5),
        nn_n_separ=8,
        nn_loss="mse",
        nn_n_branches=10,
        nn_loss_weights=3,
        nn_loss_normalization_weights=None,
        nn_activation="relu",
        nn_output_activation="linear",
        model_snr=50,
        I_max=2.2,
        parameter=["I"],
        learning_set_size=65536,
        learning_set_distribution="sobol",
        testing_set_size=8192,
        subgrid_dimensions=(),
        modified_grid_parameters=dict(),
        optimization_cost_repetitions=3,
        optimization_pop_size=40,
        optimization_max_iter=50,
        seq_name="hdr_afi",
        seq_te=3.0,
        seq_oversampling=3,
        seq_fov=np.array([0.024, 0.024]),
        seq_image_size=np.array([12, 12]),
        n_sig_per_seq=1,
        plot_mode="M",
        plot_color_map="viridis",
        plot_color_bar=False,
        dictionary_size=57344,
        dtype=np.float32,
        slice_profile_dict_name="",
        n_workers=1,
        samplegrid_mask_keep=False,
        mask_radius=np.nan,
        rng_seed_all=None,
        rng_seed_algo=None,
        rng_seed_learning=None,
        rng_seed_problem=None,
        bounds=None,
        constrain_large_first_flip=True,
    ):
        """
        Args:
            model_fit:  Choose to fit data with real, imaginary, module, phase or complex signals.
            model_conditioning_option: Choose direction along which to remove phase.
                       conditioning_option must be either 1, 2 or 3:
                        - 1 means we normalize and remove 1 phase across all voxels and signals,
                        - 2 means we normalize and remove n_seq phases across all voxels and signals of each sequence
                        - 3 means we normalize and remove n_vox phases across signals, individually for each voxel
            global_signal_scaling: True global scaling of signals in condition_signal
            model_fit_background_b1_phase: Fit background B1 phase or not. If background B1 phase it not fitted,
                       remove_phase_afi is called in conditioning the signal.
            model_do_noisy_learning: True add noise for training
            model_fit_learning_rate: True Fit learning rate in addition to other parameters by the DNN
            nn_layers: list of hidden layers sizes of the DNN
            nn_n_separ:
            nn_loss: string name of DNN loss function
            nn_n_branches:
            nn_loss_weights: weights in the loss for different outputs of the DNN
            nn_loss_normalization_weights: normalization of loss weights
            nn_activation="relu",
            nn_output_activation: activation function at the output of the DNN
            model_snr: considered SNR
            I_max: maximal current
            parameter: parameter(s) to fit
            learning_set_size: learning set size for the DNN training
            learning_set_distribution: distribution of sample to create a SampleGrid
            testing_set_size: testing set size for the DNN training
            subgrid_dimensions: parameters selected to be discretized to obtain subgrids
            modified_grid_parameters:
            optimization_cost_repetitions: number of cost evaluation repetitions
            optimization_pop_size: population size for Differential Evolution
            optimization_max_iter: maximum number of iterations for Differential Evolution
            seq_name: name of the sequence ("hdr_afi", "da_hdr_afi", "dream")
            seq_te: echo time of the sequence
            seq_oversampling: oversampling of the lambda map
            seq_fov: field of view of the sequence
            seq_image_size: size in voxel of the selected patch
            n_sig_per_seq: number of signal per MRI sequence (=2 for AFI)
            plot_mode: "M" magnitude, "R" real, "I" imaginary, "P" phase, "C" complex
            plot_color_map: colormap for plots
            plot_color_bar: True display color bars
            dictionary_size: size of dictionary for dictionary lookup (not used)
            dtype:
            slice_profile_dict_name: path to slice profile dict
            n_workers: numbers of workers used in the sequence optimization
            samplegrid_mask_keep: keep or remove voxels in mask
            mask_radius: mask radius to apply around the center of the wire
            rng_seed_all: dealing with randomization in parallel jobs
            rng_seed_algo: dealing with randomization in parallel jobs
            rng_seed_learning: dealing with randomization in parallel jobs
            rng_seed_problem: dealing with randomization in parallel jobs
            bounds: sample-dependent parameters bounds in the fitting
            constrain_large_first_flip:
        """
        self.model_fit = model_fit
        self.model_conditioning_option = model_conditioning_option
        self.global_signal_scaling = global_signal_scaling
        self.model_fit_background_b1_phase = model_fit_background_b1_phase
        self.model_do_noisy_learning = model_do_noisy_learning
        self.model_fit_learning_rate = model_fit_learning_rate
        self.nn_layers = nn_layers
        self.nn_n_separ = nn_n_separ
        self.nn_loss = nn_loss
        self.nn_n_branches = nn_n_branches
        self.nn_loss_weights = nn_loss_weights
        self.nn_loss_normalization_weights = nn_loss_normalization_weights
        self.nn_activation = nn_activation
        self.nn_output_activation = nn_output_activation
        self.model_snr = model_snr
        self.I_max = I_max
        self.parameter = parameter
        self.learning_set_size = learning_set_size
        self.learning_set_distribution = learning_set_distribution
        self.testing_set_size = testing_set_size
        self.subgrid_dimensions = subgrid_dimensions
        self.modified_grid_parameters = modified_grid_parameters
        self.optimization_cost_repetitions = optimization_cost_repetitions
        self.optimization_pop_size = optimization_pop_size
        self.optimization_max_iter = optimization_max_iter
        self.seq_name = seq_name
        self.seq_te = np.array(seq_te).flatten()
        self.seq_oversampling = seq_oversampling
        self.seq_fov = seq_fov
        self.seq_image_size = seq_image_size
        self.n_sig_per_seq = n_sig_per_seq
        self.plot_mode = plot_mode
        self.plot_color_map = plot_color_map
        self.plot_color_bar = plot_color_bar
        self.dictionary_size = dictionary_size
        self.dtype = dtype
        self.slice_profile_dict_name = slice_profile_dict_name
        self.n_workers = n_workers
        self.samplegrid_mask_keep = samplegrid_mask_keep
        self.mask_radius = mask_radius
        self.rng_seed_all = rng_seed_all
        self.rng_seed_algo = rng_seed_algo
        self.rng_seed_learning = rng_seed_learning
        self.rng_seed_problem = rng_seed_problem
        self.bounds = bounds
        self.constrain_large_first_flip = constrain_large_first_flip

    def print(self, file=None):
        """ printing ProcessingOptions"""
        attrs = vars(self)
        print("\n".join("# %s: %s" % item for item in attrs.items()), file=file)
