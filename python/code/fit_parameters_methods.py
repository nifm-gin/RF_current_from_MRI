from DE_fit_experimental_data import optimization, de_fit_params
import numpy as np

def fit_parameters(method, parameters, forward_model, processing_options, bounds, experimental_data, n_slices, error_weights):
    """ launch Differential evolution to fit sample-dependent parameters and store the resulting fitted parameters"""
    prediction = de_fit_params(n_slices, optimization, bounds[:11], experimental_data, forward_model, processing_options, error_weights)
    predicted_parameters = dict()
    predicted_parameters['I'] = prediction[:, 0]
    predicted_parameters['phj'] = prediction[:, 1]
    predicted_parameters['lambda_b'] = prediction[:, 2]
    predicted_parameters['lambda_b_dx'] = prediction[:, 3]
    predicted_parameters['lambda_b_dy'] = prediction[:, 4]
    predicted_parameters['xij'] = prediction[:, 5]
    predicted_parameters['thj'] = prediction[:, 6]
    predicted_parameters['r0'] = prediction[:, 7]
    predicted_parameters['ph0'] = prediction[:, 8]
    predicted_parameters['phb'] = prediction[:, 9]
    predicted_parameters['t1'] = 500.0*np.ones((n_slices, ))
    predicted_parameters['t2'] = 80.0 * np.ones((n_slices,))
    predicted_parameters['t2_star'] = 40.0 * np.ones((n_slices,))
    predicted_parameters['lambda_gradient'] = np.ones((n_slices,))
    predicted_parameters['signal_sign'] = prediction[:, 10]
    return predicted_parameters




