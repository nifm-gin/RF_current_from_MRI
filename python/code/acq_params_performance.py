import numpy as np
from model import SampleGrid2D
from utils.utils import condition_signal, ProcessingOptions

# Implement multiprocessing to speed up the calculation of multiple repetitions
# This code is taken from https://stackoverflow.com/a/53839801
from multiprocessing import Pipe, Process


def child_process(func):
    """Makes the function run as a separate process."""
    def wrapper(*args, **kwargs):
        def worker(conn, func, args, kwargs):
            conn.send(func(*args, **kwargs))
            conn.close()
        parent_conn, child_conn = Pipe()
        p = Process(target=worker, args=(child_conn, func, args, kwargs))
        p.start()
        ret = parent_conn.recv()
        p.join()
        return ret
    return wrapper


# @child_process

dtype = np.float32


def acq_params_performance(
        parameters: np.ndarray,
        experimental_data,
        forward_model,
        processing_options=None,
        this_error_weights=None,
):
    """ evaluate the accuracy of sample-dependent parameters for their fit"""
    if processing_options is None:
        processing_options = ProcessingOptions()

    n_param_sets = parameters.shape[0]
    learning_mesh = SampleGrid2D(
        t1=500.,
        t2=80.,
        t2_star=40.,
        signal_sign=parameters[:, 10],
        I=parameters[:,0],
        phj=parameters[:,1],
        lambda_b=parameters[:,2],
        lambda_b_dx=parameters[:,3],
        lambda_b_dy=parameters[:,4],
        xij=parameters[:,5],
        thj=parameters[:,6],
        r0=parameters[:,7],
        ph0=parameters[:,8],
        phb=parameters[:,9],
        mask_radius=processing_options.mask_radius,
        fov=processing_options.seq_fov,
        image_size=processing_options.seq_image_size,
        oversampling=processing_options.seq_oversampling,
        dtype=dtype,
        size=n_param_sets,
        distribution="raw",
        samplegrid_mask_keep=processing_options.samplegrid_mask_keep,
    )
    # simulate the raw signal for the dictionary
    learning_signal = forward_model.simulate_signal(learning_mesh)
    learning_signal = condition_signal(learning_signal,
                                       learning_mesh.size,
                                       learning_mesh.n_vox,
                                       forward_model.n_signals,
                                       processing_options)

    # weighting the error by signal contribution
    error = experimental_data.reshape((1, -1)) - learning_signal
    error = error.reshape((n_param_sets, learning_mesh.n_vox, forward_model.n_signals))
    error = error * this_error_weights
    error = error.reshape(learning_signal.shape)
    # Make this insensitive to NaN values. Use np.nanmean to be insensitive to the number of voxels masked.
    # Multiply by learning_mesh.n_vox to have the same scale as the previous np.sum
    rmse = np.sqrt(learning_mesh.n_vox * np.nanmean(np.abs(error) ** 2.0, axis=1))

    return rmse
