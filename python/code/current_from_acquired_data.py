import math
from model.sample_grid import SampleGrid2D
from model.pulse_sequence import (
    PulseSequence,
)
from fit_parameters_methods import fit_parameters
from utils.utils import condition_signal, ProcessingOptions, load_slice_profile_dict, get_data_directory
from utils.plotting import plot_patch, plot_compare, plot_compare_interface
import nibabel as nib
import matplotlib.pyplot as plt
import json
import numpy as np
import re
import os
from scipy import interpolate
import gc
import subprocess
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_acq_parameters(json_files):
    """Function to read the acquisition parameters from two .json files.

     Args:
         json_files (list of str): List of 2 JSON file names for acquisition parameters

    Function returns a list of six elements: [alpha1, alpha2, TR1, TR2, n1, n2]
    """
    n_files = 0
    sequence_name = None
    alphas = []
    tr1 = []
    n = []
    te = []
    dual_alpha = False

    for this_json_file in json_files:
        with open(this_json_file) as json_file:
            data = json.load(json_file)
        n_slices = data.get('Dataset dimensions (Count, X,Y,Z,T...)').get('value')[3][0]
        this_alpha = data.get('FlipAngle').get('value')[0][0]
        this_rep_times = data.get('RepetitionTime').get('value')
        this_te = data.get('EchoTime').get('value')[0][0]
        this_tr1 = this_rep_times[0][0]       # * n_slices when multislice and not multi2D
        if len(this_rep_times) > 1:
            this_n = this_rep_times[1][0] / this_rep_times[0][0]
            n.append(this_n)
            if n_files == 0:
                sequence_name = 'AFI'
                n_sig_per_seq = 2
            if len(data.get('FlipAngle').get('value')) > 1:
                this_alpha2 = data.get('FlipAngle').get('value')[1][0]
                dual_alpha = True
        elif n_files == 0:
            sequence_name = 'SPGR'
            n_sig_per_seq = 1
        alphas.append(this_alpha)
        if dual_alpha:
            alphas.append(this_alpha2)
        tr1.append(this_tr1)
        te.append(this_te)
        n_files = n_files + 1

    if str.lower(sequence_name) == "afi":
        if dual_alpha:
            acq_parameters = [alphas[1], alphas[3], alphas[0], alphas[2], tr1[0], tr1[1], n[0], n[1]]
        else:
            acq_parameters = [alphas[0], alphas[1], tr1[0], tr1[1], n[0], n[1]]
    elif str.lower(sequence_name) == "spgr":
        acq_parameters = [alphas, tr1]
    resolution = np.array([x[0] for x in data.get('Grid spacings (X,Y,Z,T,...)').get('value')[:2]])
    # Convert resolution to m
    resolution = resolution / 1000.
    return acq_parameters, n_slices, resolution, te, sequence_name, n_sig_per_seq, dual_alpha


def current_from_acquired_data(input_files, center_coordinates):
    """Function to fit RF current from B1-mapping data. Return value is a vector of RF current intensities in units of
    ampere, one per slice.

     Args:
         input_files (list of str): List of 2xN nifti file names for input data - acq1,real; acq1,imag; acq2,real;
            acq2,imag;..., where N is the number of acquisitions
         center_coordinates (list of float): XY coordinates [voxels] of the center of the patch to be extracted
            (location of the implant wire). Either a 2-vector, or 4-vector if the wire is not parallel to z. A 4-vector
            should contain the XY coordinates of the wire in the first and last slices in the order
            [Xfirst, Yfirst, Xlast, Ylast]. These coordinates should be in Matlab convention (indices starting at 1),
            as determined by SPM for example.
     """
    # loading the nifti files and figuring out data directory
    file_address_1 = input_files[0]
    experiment_dir, file_name_1 = os.path.split(file_address_1)
    orig_dir, experiment_name = os.path.split(experiment_dir)
    data_dir = os.path.dirname(orig_dir)
    base_dir = os.path.dirname(orig_dir)
    out_dir = ''
    results_dir = data_dir + "/results/" + experiment_name + "/"
    for file_address in input_files:
        this_file_name = os.path.basename(file_address)
        out_dir += this_file_name[0:5] + "_"
    results_dir = os.path.join(data_dir,"results", experiment_name, out_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("# Created directory: %s" % (results_dir,))

    # Read acquisition parameters from .json files accompanying the .nii files
    json_files = [re.sub('.nii$', '.json', my_str) for my_str in input_files]
    acq_parameters, n_slices, resolution, te, sequence_name, n_sig_per_seq, dual_alpha = get_acq_parameters(json_files)

    # Define the position of the wire
    if len(center_coordinates) > 2:  # in case of angulation between wire and z-direction
        if len(center_coordinates) > 4:
            center_coordinates = np.array(center_coordinates)
            f = interpolate.interp1d(center_coordinates[[2, 5]], center_coordinates[[0, 3]],
                                     fill_value="extrapolate")
            x_values = f(np.arange(n_slices) + 1)
            f = interpolate.interp1d(center_coordinates[[2, 5]], center_coordinates[[1, 4]],
                                     fill_value="extrapolate")
            y_values = f(np.arange(n_slices) + 1)
            center_coordinates = np.array(list(zip(x_values, y_values)))
        else:
            x_values = np.linspace(center_coordinates[0], center_coordinates[2], n_slices)
            y_values = np.linspace(center_coordinates[1], center_coordinates[3], n_slices)
            center_coordinates = np.array(list(zip(x_values, y_values)))
    else:
        center_coordinates = np.tile(np.array(center_coordinates).reshape((1, 2,)), (n_slices, 1))
    # Convert Matlab center coordinates to Python.
    center_coordinates = center_coordinates - 1.0
    # We will use a range of integer indices into the array, such that the center will be located at a half-integer
    # coordinate. Round specified center to the nearest half-integer value and choose the next integer above as center
    # index for the calculation of the range bounds. The center will be half a voxel below the calculated center index.
    center_coordinates = np.round(center_coordinates + 0.5).astype(int)

    # script options
    parameter = ["I", "phj", "lambda_b", "lambda_b_dx", "lambda_b_dy", "xij", "thj", "r0", "ph0", "phb", "signal_sign"]
    fit_parameters_method = "DE"
    model_fit = ["M"]    # C, R, I, M, P
    n_param_combinations = len(model_fit)
    model_conditioning_option = [3]
    # We need to avoid global signal scaling in signal conditioning, in case the number of masked voxels is variable
    global_signal_scaling = [False]
    model_fit_background_b1_phase = [True]
    model_do_noisy_learning = [False]
    model_snr = 50
    learning_set_distribution = "sobol"
    dictionary_size = 57344
    optimization_pop_size = 61
    optimization_max_iter = 122
    I_max = 2.2
    seq_image_size = np.array([12, 12])                    # np.array([12, 12])
    seq_fov = seq_image_size * resolution
    seq_te = te
    seq_oversampling = 3
    plot_mode = ("M")
    plot_color_map = "turbo"
    plot_color_bar = True
    plot_savefig = True
    plot_interface = False
    plot_show_experimental_data = True
    plot_show_reconstructed_data = True
    plot_show_comparison = True
    plot_show_parameters = True
    dtype = np.float32
    git_version = subprocess.check_output(["git", "describe", "--always"]).decode("utf-8").strip()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    git_version = git_version + "(" + git_branch + ")"

    if not isinstance(model_fit, list):
        model_fit_list = [model_fit, ]
        model_conditioning_option_list = [model_conditioning_option, ]
        global_signal_scaling_list = [global_signal_scaling, ]
        model_fit_background_b1_phase_list = [model_fit_background_b1_phase, ]
        model_do_noisy_learning_list = [model_do_noisy_learning, ]
    else:
        model_fit_list = model_fit
        model_conditioning_option_list = model_conditioning_option
        global_signal_scaling_list = global_signal_scaling
        model_fit_background_b1_phase_list = model_fit_background_b1_phase
        model_do_noisy_learning_list = model_do_noisy_learning

    # Prepare pulse sequence (forward model) and DNN (inverse model)
    t1 = 500.0
    if dual_alpha:
        slice_profile_dict = [os.path.join(base_dir, "dico_bloch/signal_dict/dahdrafi_sim_") + format(acq_parameters[0], ".2f") + "_" + format(acq_parameters[2], ".2f") + "_" + format(acq_parameters[4], ".2f") + "_" + format(acq_parameters[6], ".2f") + "_" + str(round(t1)) + "_real",
                              os.path.join(base_dir, "dico_bloch/signal_dict/dahdrafi_sim_") + format(acq_parameters[1], ".2f") + "_" + format(acq_parameters[3], ".2f") + "_" + format(acq_parameters[5], ".2f") + "_" + format(acq_parameters[7], ".2f") + "_" + str(round(t1)) + "_real"]
    else:
        slice_profile_dict = [os.path.join(base_dir, "dico_bloch/signal_dict/dahdrafi_sim_") + format(acq_parameters[0], ".2f") + "_" + format(acq_parameters[2], ".2f") + "_" + format(acq_parameters[4], ".2f") + "_" + str(round(t1)) + "_real",
                              os.path.join(base_dir, "dico_bloch/signal_dict/dahdrafi_sim_") + format(acq_parameters[1], ".2f") + "_" + format(acq_parameters[3], ".2f") + "_" + format(acq_parameters[5], ".2f") + "_" + str(round(t1)) + "_real"]
    forward_model = PulseSequence(
        sequence=sequence_name,
        min_n=1.0,
        max_n=39.0,
        min_alpha2=1.0,
        max_alpha2=360.0,
        min_tr=10.0,
        n_phase_encode_lines_afi=40,
        n_phase_encode_lines_spgr=40,
        abort_if_out_of_bounds=True,
        # change the sequence to dual angle
        dual_alpha=dual_alpha,
        repeat_acquisition=1,
        two_dimension=True,
        slice_profile_dict_type="signal",
        slice_profile_dict_signal=[load_slice_profile_dict(slice_profile_dict[0], dtype, dict_type="signal"), load_slice_profile_dict(slice_profile_dict[1], dtype, dict_type="signal")],
    )
    forward_model.set_sequence_parameters(
        acq_parameters, te=te
    )
    n_sig_per_vox = forward_model.n_signals
    n_vox = int(np.prod(seq_image_size))

    # Let's read experimental data
    # input the address of image file then load to access the header and array information
    n_files_read = 0
    for file_address in input_files:
        this_image = nib.load(file_address)
        this_image_array = this_image.get_fdata()
        del this_image
        if this_image_array.shape[3] == 2:
            temp_image_array = np.copy(this_image_array)
            if len(this_image_array.shape) < 5:
                this_image_array = this_image_array.reshape(this_image_array.shape + (1,))
                temp_image_array = temp_image_array.reshape(temp_image_array.shape + (1,))
            # The two AFI signals are inverted between converted Philips data and our convention.
            # Bring the stored data back to our convention.
            this_image_array[:, :, :, 0, :] = temp_image_array[:, :, :, 1, :]
            this_image_array[:, :, :, 1, :] = temp_image_array[:, :, :, 0, :]
            del temp_image_array
        if n_files_read == 0:
            if str.lower(sequence_name) == "afi":
                if this_image_array.shape[-1] == 1:
                    all_images = this_image_array
                    complex_image_flag = 0
                else:
                    all_images = this_image_array[:, :, :, :, 1:3]
                    complex_image_flag = 1
            elif str.lower(sequence_name) == "spgr":
                if len(this_image_array.shape)==3:
                    all_images = this_image_array.reshape(this_image_array.shape + (1,))
                    complex_image_flag = 0
                else:
                    all_images = np.expand_dims(this_image_array[:, :, :, 1:3], axis=3)
                    complex_image_flag = 1
        else:
            if str.lower(sequence_name) == "afi":
                if complex_image_flag:
                    all_images = np.concatenate((all_images, this_image_array[:, :, :, :, 1:3]), axis=3)
                else:
                    all_images = np.concatenate((all_images, this_image_array[:, :, :, :, :]), axis=3)
            elif str.lower(sequence_name) == "spgr":
                if complex_image_flag:
                    all_images = np.concatenate((all_images, np.expand_dims(this_image_array[:, :, :, 1:3], axis=3)), axis=3)
                else:
                    all_images = np.concatenate((all_images, this_image_array.reshape(this_image_array.shape + (1,))),
                                            axis=3)
        n_files_read = n_files_read + 1
    all_images = all_images.swapaxes(2, 3)

    raw_size = all_images.shape[:2]
    sx = seq_image_size[0]
    sy = seq_image_size[1]
    raw_signal = np.zeros((n_vox, n_sig_per_vox, n_slices), dtype=complex)
    slice_complete = np.full((n_slices,), True)
    for i in range(n_slices):
        cx = center_coordinates[i][0]
        cy = center_coordinates[i][1]
        for i_seq in range(all_images.shape[2]):
            try:
                if str.lower(sequence_name) == "afi":
                    if complex_image_flag:
                        this_sr = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i, 0].reshape(
                            (n_vox,))
                        this_si = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i, 1].reshape(
                            (n_vox,))
                    else:
                        this_sm = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i, 0].reshape(
                            (n_vox,))
                elif str.lower(sequence_name) == "spgr":  # Only because we do not have imaginary phase and real data
                    if complex_image_flag:
                        this_sr = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i, 0].reshape(
                            (n_vox,))
                        this_si = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i, 1].reshape(
                            (n_vox,))
                    else:
                        this_sm = all_images[cx - sx // 2:cx + sx // 2, cy - sy // 2:cy + sy // 2, i_seq, i].reshape(
                            (n_vox,))
            except ValueError:  # The patch area is partly outside of the acquired image
                slice_complete[i] = False
                continue

            # create the signal equation
            if complex_image_flag:
                s = this_sr + 1j * this_si
                del this_sr, this_si
            else:
                s = this_sm
                del this_sm
            raw_signal[:, i_seq, i] = s
        print("# Voxel x-coordinates for slice %d: " % i,
              np.array(range(raw_size[0]))[cx - sx // 2:cx + sx // 2])
        print("# Voxel y-coordinates for slice %d: " % i,
              np.array(range(raw_size[1]))[cy - sy // 2:cy + sy // 2])

    del all_images, cx, cy
    del s

    # Eliminate parts of the signal array that are not used because of partial data
    slice_min = np.argmax(slice_complete)  # index of first good slice
    slice_max = n_slices - np.argmax(np.flip(slice_complete))  # index of last good slice + 1
    del slice_complete
    raw_signal = raw_signal[:, :, slice_min:slice_max]
    n_slices = slice_max - slice_min

    xylim = (seq_fov - resolution) / 2
    [x, y] = np.meshgrid(
        np.linspace(-xylim[0], xylim[0], seq_image_size[0], endpoint=True, dtype=dtype),
        np.linspace(-xylim[1], xylim[1], seq_image_size[1], endpoint=True, dtype=dtype),
    )
    x = np.reshape(x, (n_vox,))
    y = np.reshape(y, (n_vox,))
    r = np.sqrt(x ** 2 + y ** 2)

    # start loop over processing parameter lists
    for i_param_set in range(n_param_combinations):
        # Limit parameters for fit
        t1_min, t1_max = t1, t1  # 500. 500.
        t2_min, t2_max = 80., 80.
        t2_star_min, t2_star_max = 40., 40.
        signal_sign_min, signal_sign_max = -1, 1
        I_min, I_max = 0., I_max
        phj_min, phj_max = 0., 360. * math.pi / 180.
        lambda_b_dx_min, lambda_b_dx_max = -20.0, 20.0       # -100.0, 100.0     # -25.0, 25.0
        lambda_b_dy_min, lambda_b_dy_max = -20.0, 20.0       # -100.0, 100.0     # -25.0, 25.0
        lambda_b_min, lambda_b_max = 0.5, 6.
        xij_min, xij_max = 0., 55. * math.pi / 180.
        thj_min, thj_max = 0., 180. * math.pi / 180.
        r0_min, r0_max = resolution[0] * 0., resolution[0] * 2.
        ph0_min, ph0_max = 0., 360. * math.pi / 180.
        if model_fit_background_b1_phase_list[i_param_set]:
            phb_min, phb_max = 0., 360. * math.pi / 180.
            if model_fit_list[i_param_set] == "M":
                phb_min, phb_max = 0., 0.
            signal_sign_min, signal_sign_max = 1, 1
        else:
            phb_min, phb_max = 0., 0.
        mask_radius = 4e-3

        model_fit = model_fit_list[i_param_set]
        model_conditioning_option = model_conditioning_option_list[i_param_set]
        global_signal_scaling = global_signal_scaling_list[i_param_set]
        model_fit_background_b1_phase = model_fit_background_b1_phase_list[i_param_set]
        model_do_noisy_learning = model_do_noisy_learning_list[i_param_set]
        phase_noise = not model_fit_background_b1_phase
        bounds = [[I_min, I_max], [phj_min, phj_max], [lambda_b_min, lambda_b_max], [lambda_b_dx_min, lambda_b_dx_max],
                  [lambda_b_dy_min, lambda_b_dy_max], [xij_min, xij_max], [thj_min, thj_max], [r0_min, r0_max],
                  [ph0_min, ph0_max], [phb_min, phb_max], [signal_sign_min, signal_sign_max], [t1_min, t1_max], [t2_min, t2_max], [t2_star_min, t2_star_max]]

        plot_save_fname = "%s-%d-%s-%s-%s-v%s" % \
                          (str(model_fit),
                           model_conditioning_option,
                           str(model_fit_background_b1_phase),
                           str(model_do_noisy_learning),
                           str(fit_parameters_method),
                           git_version)
        plot_save_fname = os.path.join(results_dir, plot_save_fname)

        processing_options = ProcessingOptions(
            model_fit=model_fit,
            model_conditioning_option=model_conditioning_option,
            global_signal_scaling=global_signal_scaling,
            model_fit_background_b1_phase=model_fit_background_b1_phase,
            model_do_noisy_learning=model_do_noisy_learning,
            model_snr=model_snr,
            learning_set_distribution=learning_set_distribution,
            I_max=I_max,
            seq_te=seq_te,
            seq_oversampling=seq_oversampling,
            seq_fov=seq_fov,
            seq_image_size=seq_image_size,
            n_sig_per_seq=n_sig_per_seq,
            plot_mode=plot_mode,
            plot_color_map=plot_color_map,
            plot_color_bar=plot_color_bar,
            optimization_pop_size=optimization_pop_size,
            optimization_max_iter=optimization_max_iter,
            dictionary_size=dictionary_size,
            bounds=bounds,
            samplegrid_mask_keep=True,
            mask_radius=mask_radius,
        )

        print("# Git version number: %s" % git_version)
        processing_options.print()
        n_files_read = 0
        for file_address in input_files:
            print("# Read file " + str(n_files_read + 1) + ": " + file_address)
            n_files_read = n_files_read + 1

        # save trace of processing options to log if we save data
        if plot_savefig:
            with open(plot_save_fname + '_params.txt', 'w') as f:
                print("# Git version number: %s" % git_version, file=f)
                processing_options.print(file=f)
                n_files_read = 0
                for file_address in input_files:
                    print("# Read file " + str(n_files_read + 1) + ": " + file_address, file=f)
                    n_files_read = n_files_read + 1

        experimental_data = raw_signal
        # experimental_data[r < mask_radius, :, :] = 0

        # Reshape and transpose signal to n_slices x n_sig_per_vox
        experimental_data = experimental_data.reshape((n_vox * n_sig_per_vox, n_slices))
        experimental_data = experimental_data.T

        conditioned_data = condition_signal(experimental_data, n_slices, n_vox, n_sig_per_vox, processing_options)
        # processing_options.model_fit_background_b1_phase = True

        error_weights = np.abs(experimental_data) / conditioned_data  # retrieve normalization weights
        error_weights[np.isnan(error_weights)] = 1.0
        error_weights = error_weights.reshape((n_slices, n_vox, n_sig_per_vox))
        experimental_data = conditioned_data
        del conditioned_data

        # fit sample-dependent parameters from experimental_data
        fitted_parameters = fit_parameters(fit_parameters_method, parameter, forward_model, processing_options, bounds,
                                           experimental_data, n_slices, error_weights)
        predicted_parameters = fitted_parameters

        # Now reconstruct the signals from fitted sample-dependent parameters, slice by slice
        if processing_options.model_fit == "C":
            reconstruction_signal = np.zeros((n_slices, n_vox * n_sig_per_vox * 2))
            reconstruction_lambda = np.zeros((n_slices, n_vox * processing_options.seq_oversampling ** 2 * 2))
        else:
            reconstruction_signal = np.zeros((n_slices, n_vox * n_sig_per_vox))
            reconstruction_lambda = np.zeros((n_slices, n_vox * processing_options.seq_oversampling ** 2))

        for i_slice in range(n_slices):
            reconstruction_mesh = SampleGrid2D(
                t1=predicted_parameters['t1'][i_slice],
                t2=predicted_parameters['t2'][i_slice],
                t2_star=predicted_parameters['t2_star'][i_slice],
                signal_sign=predicted_parameters['signal_sign'][i_slice],
                I=predicted_parameters['I'][i_slice],
                phj=predicted_parameters['phj'][i_slice],
                lambda_b=predicted_parameters['lambda_b'][i_slice],
                lambda_b_dx=predicted_parameters['lambda_b_dx'][i_slice],
                lambda_b_dy=predicted_parameters['lambda_b_dy'][i_slice],
                xij=predicted_parameters['xij'][i_slice],
                thj=predicted_parameters['thj'][i_slice],
                r0=predicted_parameters['r0'][i_slice],
                ph0=predicted_parameters['ph0'][i_slice],
                phb=predicted_parameters['phb'][i_slice],
                mask_radius=mask_radius,
                fov=seq_fov,
                image_size=seq_image_size,
                distribution='grid',
                oversampling=seq_oversampling,
                dtype=dtype,
                samplegrid_mask_keep=processing_options.samplegrid_mask_keep,
            )
            this_slice_signal = forward_model.simulate_signal(reconstruction_mesh)
            this_slice_signal = condition_signal(this_slice_signal, 1, n_vox, n_sig_per_vox, processing_options)
            reconstruction_signal[i_slice, :] = this_slice_signal
            this_lambda = np.sqrt(reconstruction_mesh.grid_vec["lambda"])  # compress amplitude for better display
            this_b1_phase = reconstruction_mesh.grid_vec["b1_phase"]
            max_lambda = this_lambda.max() / 2
            this_lambda[this_lambda > max_lambda] = max_lambda  # clip data to avoid outliers
            this_lambda = this_lambda * np.exp(1j * this_b1_phase)
            if processing_options.model_fit == "C":
                this_lambda = np.concatenate((this_lambda.real[:, :, np.newaxis], this_lambda.imag[:, :, np.newaxis]),
                                             axis=2)
                this_lambda = this_lambda.reshape((1, n_vox * seq_oversampling ** 2 * 2))
            reconstruction_lambda[i_slice, :] = this_lambda

        #apply same sign
        reconstruction_signal = np.sign(predicted_parameters['signal_sign']).reshape(-1, 1) * reconstruction_signal
        experimental_data = np.sign(predicted_parameters['signal_sign']).reshape(-1, 1) * experimental_data

        # experimental_data plotting
        if plot_show_experimental_data:
            if plot_savefig:
                plot_patch(experimental_data, seq_image_size, n_slices, processing_options,
                           plot_save_fname + "_exp-data")
            else:
                plot_patch(experimental_data, seq_image_size, n_slices, processing_options)

        # fitted signal plotting
        if plot_show_reconstructed_data:
            reconstruction_lambda = reconstruction_lambda.reshape(
                (n_slices, reconstruction_mesh.image_size[0], reconstruction_mesh.image_size[1], reconstruction_mesh.oversampling, reconstruction_mesh.oversampling))
            reconstruction_lambda = np.swapaxes(reconstruction_lambda, 2, 3)
            reconstruction_lambda = reconstruction_lambda.reshape((n_slices, reconstruction_mesh.n_vox_os))
            if plot_savefig:
                plot_patch(reconstruction_signal, seq_image_size, n_slices, processing_options,
                           plot_save_fname + "_reco-data")
                plot_patch(reconstruction_lambda, seq_image_size * seq_oversampling, n_slices, processing_options,
                           plot_save_fname + "_lambda")
            else:
                plot_patch(reconstruction_signal, seq_image_size, n_slices, processing_options)
                plot_patch(reconstruction_lambda, seq_image_size * seq_oversampling, n_slices, processing_options)
        # plotting comparison between target and fitted signals
        if plot_show_comparison:
            if plot_savefig:
                plot_compare(experimental_data, reconstruction_signal, seq_image_size, n_slices, processing_options,
                             plot_save_fname + "_comparison", all_slices=True)
            else:
                plot_compare(experimental_data, reconstruction_signal, seq_image_size, n_slices, processing_options)

        # interface to see influence of sample-dependent parameters
        if plot_interface:
            n_slice_to_display = 4

            window = Tk()
            window.title = "Interface de modélisation"
            window.geometry("1500x825")

            frame = Frame(window)
            width = 12.8
            height = 9.6
            fig = plt.figure(figsize=(width, height))
            canvas = FigureCanvasTkAgg(fig,
                                       master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            frame.pack(side=LEFT, fill=X)

            def function():
                modif_vars = my_var
                interface_mesh = SampleGrid2D(
                    t1=np.array([modif_vars[10].get()]),
                    t2=np.array([modif_vars[11].get()]),
                    t2_star=np.array([modif_vars[12].get()]),
                    signal_sign=np.array([modif_vars[14].get()]),
                    I=np.array([modif_vars[0].get()]),
                    phj=np.array([modif_vars[1].get()]),
                    lambda_b=np.array([modif_vars[2].get()]),
                    lambda_b_dx=np.array([modif_vars[3].get()]),
                    lambda_b_dy=np.array([modif_vars[4].get()]),
                    xij=np.array([modif_vars[5].get()]),
                    thj=np.array([modif_vars[6].get()]),
                    r0=np.array([modif_vars[7].get()]),
                    ph0=np.array([modif_vars[8].get()]),
                    phb=np.array([modif_vars[9].get()]),
                    mask_radius=mask_radius,
                    fov=seq_fov,
                    image_size=seq_image_size,
                    distribution='grid',
                    oversampling=seq_oversampling,
                    dtype=dtype,
                    samplegrid_mask_keep=True,
                )

                this_slice_signal = forward_model.simulate_signal(interface_mesh)
                interface_signal = condition_signal(this_slice_signal, 1, n_vox, n_sig_per_vox, processing_options)
                interface_signal = interface_signal.reshape((1, -1)) * np.sign(interface_mesh.grid['signal_sign'].reshape((-1,1)))
                fig = plt.figure()
                print(modif_vars[15].get())
                processing_options.plot_mode = (modif_vars[15].get(),)
                fig = plot_compare_interface(experimental_data[n_slice_to_display, :].reshape((1, -1)), interface_signal,
                                       seq_image_size, processing_options, fig)

                canvas.figure = fig
                canvas.draw()
                # canvas.get_tk_widget().pack()

            def reset_button():
                for i, key in enumerate(predicted_parameters):
                    my_var[i].set(predicted_parameters[key][n_slice_to_display])

            right_frame = Frame(window)
            label_title = Label(right_frame, text='Variables')
            label_title.pack(expand=YES)

            my_var = []
            entree = []
            for i, key in enumerate(predicted_parameters):
                this_var = DoubleVar(master=right_frame)
                this_var.set(predicted_parameters[key][n_slice_to_display])
                my_var.append(this_var)
                label_title = Label(right_frame, text=key)
                label_title.pack(expand=YES)
                this_entree = Entry(right_frame, textvariable=my_var[i])
                this_entree.pack()
                entree.append(this_entree)
            this_var = StringVar(master=right_frame)
            this_var.set('R')
            my_var.append(this_var)
            label_title = Label(right_frame, text='plot mode')
            label_title.pack(expand=YES)
            this_entree = Entry(right_frame, textvariable=my_var[-1])
            this_entree.pack()
            entree.append(this_entree)

            bouton = Button(right_frame, text='Show simulation', command=function)
            bouton.pack()
            Button(right_frame, text="Reset", command=reset_button).pack()
            Button(right_frame, text="Quit", command=window.quit).pack()

            right_frame.pack(side=LEFT, fill=X)
            window.mainloop()

        # plot reconstructed parameters
        predicted_parameters['x0'] = predicted_parameters['r0'] * np.cos(predicted_parameters['ph0'])
        predicted_parameters['y0'] = predicted_parameters['r0'] * np.sin(predicted_parameters['ph0'])
        n_params = len(predicted_parameters)
        if plot_show_parameters:
            n_axes1 = math.floor(math.sqrt(n_params))
            n_axes2 = math.ceil(n_params / n_axes1)
            fig, axes = plt.subplots(n_axes1, n_axes2)
            fig.set_figwidth(20)
            fig.set_figheight(15)
            for i, key in enumerate(predicted_parameters):
                axes[i % n_axes1, i // n_axes1].plot(range(n_slices), predicted_parameters[key], )
                axes[i % n_axes1, i // n_axes1].set(xlabel='', ylabel=key, title='Fitted ' + key)
            if plot_savefig:
                plt.savefig(plot_save_fname + "_parameters.png")
                plt.clf()
                plt.close(fig)
        # plot reconstructed currents
        fig1, axes1 = plt.subplots(1, 1)
        axes1.set(xlabel='slice', ylabel='current', title='Fitted current')
        plt.plot(range(n_slices), predicted_parameters['I'])
        if (plot_show_experimental_data or plot_show_reconstructed_data or plot_show_comparison or plot_show_parameters) \
                and not plot_savefig:
            plt.show()

        # Calculate difference between acquired signal and reconstructed signal
        rmse_diff = np.sqrt(np.nansum(np.abs(experimental_data - reconstruction_signal) ** 2.0, axis=1))
        # Calculate RMSE for R I M P maps
        if processing_options.model_fit == "C":
            exp_signal = experimental_data.reshape((n_slices, seq_image_size[0], seq_image_size[1], -1, 2))
            exp_signal = exp_signal[:, :, :, :, 0] + 1j * exp_signal[:, :, :, :, 1]
            reco_signal = reconstruction_signal.reshape((n_slices, seq_image_size[0], seq_image_size[1], -1, 2))
            reco_signal = reco_signal[:, :, :, :, 0] + 1j * reco_signal[:, :, :, :, 1]
            rmse_I = np.sqrt(np.sum(np.abs(exp_signal.imag - reco_signal.imag) ** 2.0, axis=(1, 2, 3)))
            rmse_P = np.sqrt(np.sum(np.abs(np.angle(exp_signal) - np.angle(reco_signal)) ** 2.0, axis=(1, 2, 3)))
        if not processing_options.model_fit == "C":
            exp_signal = experimental_data.reshape((n_slices, seq_image_size[0], seq_image_size[1], -1))
            reco_signal = reconstruction_signal.reshape((n_slices, seq_image_size[0], seq_image_size[1], -1))
            rmse_I = np.zeros(n_slices)
            rmse_P = np.zeros(n_slices)
        rmse_R = np.sqrt(np.nansum(np.abs(exp_signal.real - reco_signal.real) ** 2.0, axis=(1, 2, 3)))
        rmse_M = np.sqrt(np.nansum(np.abs(np.abs(exp_signal) - np.abs(reco_signal)) ** 2.0, axis=(1, 2, 3)))

        print("Median relative error between reconstructed signal and experimental data" + str(np.nanmedian(np.abs(experimental_data - reconstruction_signal) / experimental_data)))
        print("# Best fit parameters: ")
        print(predicted_parameters)
        print("# RMSE difference between data and reco: ")
        print(rmse_diff, '\n', rmse_R, '\n', rmse_M, '\n', )
        print(rmse_I, '\n', rmse_P, '\n', )

        if plot_savefig:
            with open(plot_save_fname + '_params.txt', 'a') as f:
                print("Median relative error between reconstructed signal and experimental data" + str(np.nanmedian(np.abs(experimental_data - reconstruction_signal) / experimental_data)), file=f)
                print("# Best fit parameters: ", file=f)
                print(json.dumps(predicted_parameters, cls=NumpyEncoder), file=f)
                print("# RMSE difference between data and reco: ", file=f)
                print(json.dumps(rmse_diff, cls=NumpyEncoder), file=f)
                print("# RMSE R difference between data and reco: ", file=f)
                print(json.dumps(rmse_R, cls=NumpyEncoder), file=f)
                print("# RMSE M difference between data and reco: ", file=f)
                print(json.dumps(rmse_M, cls=NumpyEncoder), file=f)
                print("# RMSE I difference between data and reco: ", file=f)
                print(json.dumps(rmse_I, cls=NumpyEncoder), file=f)
                print("# RMSE P difference between data and reco: ", file=f)
                print(json.dumps(rmse_P, cls=NumpyEncoder), file=f)

        # free unused memory
        gc.collect()

        del exp_signal
        del experimental_data
        del predicted_parameters
        del reconstruction_lambda, reconstruction_signal, reconstruction_mesh
        del rmse_I, rmse_R, rmse_M, rmse_P, rmse_diff
        del this_lambda
        del this_b1_phase
        del this_slice_signal
        del axes, fig, fig1
        del f
        del reco_signal

    return


data_dir = os.path.join(get_data_directory())
print(data_dir)

# Copper wire position  1
current_from_acquired_data(
    [data_dir + '/copper_wire/03-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
     data_dir + '/copper_wire/04-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
    [114.4, 67.1, 0.6, 112.6, 67.8, 19.4])
# # Copper wire position 2
# current_from_acquired_data(
#     [data_dir + '/copper_wire/08-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/09-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [114.2, 67.1, 0.6, 85.5, 68.0, 19.4])
# # Copper wire position 3
# current_from_acquired_data(
#     [data_dir + '/copper_wire/14-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/15-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [115.2, 67.1, 0.4, 72.8, 68.2, 19.6])
# # Copper wire position 4
# current_from_acquired_data(
#     [data_dir + '/copper_wire/19-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/20-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [96.9, 67.3, 0.6, 96.2, 68.0, 19.3])
# # Copper wire position 5
# current_from_acquired_data(
#     [data_dir + '/copper_wire/24-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/25-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [86.2, 67.2, 0.7, 87.0, 67.9, 19.5])
# # Copper wire position 6
# current_from_acquired_data(
#     [data_dir + '/copper_wire/29-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/30-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [76.4, 67.2, 0.7, 76.1, 67.8, 19.4])
# # Copper wire position 7
# current_from_acquired_data(
#     [data_dir + '/copper_wire/34-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/35-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [57.8, 67.5, 0.7, 60.9, 68.0, 19.3])
# # Copper wire position 8
# current_from_acquired_data(
#     [data_dir + '/copper_wire/39-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/40-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [62.4, 67.3, 0.6, 106.4, 67.4, 19.4])
# # Copper wire position 9
# current_from_acquired_data(
#     [data_dir + '/copper_wire/44-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/copper_wire/45-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [72.6, 67.3, 0.6, 45.9, 68.0, 19.5])
# # DBS lead position 1
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/03-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/04-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [111.4, 68.2, 0.6, 112.9, 67.7, 12.9])
# # DBS lead position  2
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/08-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/09-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [111.9, 67.9, 0.7, 113.9, 68.0, 12.5])
# # DBS lead position 3
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/13-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/14-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [112.5, 68.0, 0.7, 113.5, 68.0, 13.7])
# # DBS lead position  4
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/18-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/19-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [112.8, 67.7, 0.7, 114.2, 67.2, 11.9])
# # DBS lead position  5
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/23-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/24-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [113.1, 67.8, 0.5, 114.2, 66.9, 11.9])
# # DBS lead position  6
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/28-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/29-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [112.6, 67.6, 0.4, 113.7, 67.5, 10.5])
# # DBS lead position 7
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/33-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/34-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [85.7, 67.6, 0.6, 87.0, 67.0, 10.3])
# # DBS lead position 8
# current_from_acquired_data(
#     [data_dir + '/DBS_lead/38-01-WIPdahdrAFI1_45_2_69_3_23_7_2_60_quadrature_new_opti.nii',  # noqa
#      data_dir + '/DBS_lead/39-01-WIPdahdrAFI2_21_9_39_0_26_5_7_12_quadrature_new_opti.nii'],  # noqa
#     [53.5, 68.5, 0.5, 53.7, 67.5, 10.5])
# # simulated copper wire parallel to B0
# current_from_acquired_data(
#     [data_dir + '/simulated_copper_wire/01_sequence_1_da_hdr_AFI_new_opti_straight_wire.nii',  # noqa
#      data_dir + '/simulated_copper_wire/02_sequence_2_da_hdr_AFI_new_opti_straight_wire.nii'],  # noqa
#     [76.5, 148.3, 1, 76.5, 148.3, 41])
# # simulated copper wire tilted at 45°
# current_from_acquired_data(
#     [data_dir + '/simulated_copper_wire/03_sequence_1_da_hdr_AFI_new_opti_tilted_wire.nii',  # noqa
#      data_dir + '/simulated_copper_wire/04_sequence_2_da_hdr_AFI_new_opti_tilted_wire.nii'],  # noqa
#     [84.9, 149.9, 1.0, 155.6, 149.9, 41.0])



