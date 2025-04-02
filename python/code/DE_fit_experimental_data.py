from scipy.optimize import differential_evolution
import numpy as np
import time
from acq_params_performance import acq_params_performance

acq_time_limit = 12000

# objective function
def optimization(v, experimental_data, forward_model, options, this_error_weights):
    """ function used by DE at each iteration to evaluate the cost for each population member"""
    start_opt = time.time()

    rmse = acq_params_performance(v, experimental_data, forward_model, options, this_error_weights)

    # make sure to return a finite value
    if (rmse.any == float("Nan")) or (rmse.any == np.inf):
        message = "Invalid error"
        rmse = 100

    end_opt = time.time()
    opt_duration = end_opt - start_opt
    # print('In optimization : f(%s) = %.5f. Duration : %.2f. %s' % (v, rmse, opt_duration, message))
    return rmse


def my_map_function(f, params):
    """ parallel estimation of all population members costs"""
    return f(params)


def de_fit_params(n_slices, optimization, bounds, experimental_data, forward_model, processing_options, error_weights):
    """ set Differential Evolution up for sample-dependent parameters fitting"""
    n_params_to_optimize = len(bounds)
    fitted_params = np.zeros((n_slices, n_params_to_optimize))
    for i_slice in range(n_slices):
        this_error_weights = error_weights[(i_slice,), :, :]
        result = differential_evolution(optimization,
                                        bounds,
                                        args=(experimental_data[i_slice, :], forward_model, processing_options, this_error_weights),
                                        maxiter=processing_options.optimization_max_iter,
                                        popsize=processing_options.optimization_pop_size,
                                        updating='deferred',
                                        workers=my_map_function,
                                        polish=False,
                                        disp=False,
                                        tol=0.001)
        # summarize the result
        print('status: %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        # evaluate solution
        solution = result.x
        evaluation = result.fun
        print('Solution: f(%s) = %.5f' % (solution, evaluation))
        fitted_params[i_slice, :] = solution
        print('Fitted slice %d / %d' % (i_slice+1, n_slices))
    return fitted_params

