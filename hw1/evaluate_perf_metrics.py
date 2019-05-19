import numpy as np

def calc_perf_metric__squared_error(y_N, yhat_N):
    ''' Compute the mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mse : scalar float
        mse = \mean_{n=1}^N (y_n - \hat{y}_n)^2
    '''
    #print(y_N-yhat_N)
    l2Sum=1.0*np.linalg.norm(y_N-yhat_N)**2;
    vlen=1.0*y_N.shape[0];
    #print(l2Sum, vlen)
    return  l2Sum/vlen; # TODO FIXME

def calc_perf_metric__absolute_error(y_N, yhat_N):
    ''' Compute the mean absolute error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mae : scalar float
        mae = \mean_{n=1}^N | y_n - \hat{y}_n |
    '''
    l1Sum=1.0*np.linalg.norm(y_N-yhat_N, ord=1);
    vlen=1.0*y_N.shape[0];
    return l1Sum/vlen

def calc_perf_metrics_for_regressor_on_dataset(
        regressor, x_NF, y_N,
        label_str=None,
        show_perf_metrics=[]):
    ''' Compute predictions for a regressor on a specific dataset of (x,y) pairs

    Will compute all of:
    * 'mse' (mean squared error)
    * 'mae' (mean absolute error)

    If provided list 'show_perf_metrics' is not empty, will print those values

    Returns
    -------
    perf_dict : dict
        Contains a key,value pair for each perf_metric
    '''
    yhat_N = regressor.predict(x_NF)
    mse = calc_perf_metric__squared_error(y_N, yhat_N)
    mae = calc_perf_metric__absolute_error(y_N, yhat_N)
    #print(mse, " ", mae);
    if len(show_perf_metrics) > 0:
        if label_str is not None:
            print('===== %s' % label_str)

    for perf_metric in show_perf_metrics:
        if perf_metric == 'mse':
            print('  squared error  %6.2f' % mse)
        elif perf_metric == 'mae':
            print('  absolute error %6.2f' % mae)
    return dict(mse=mse, mae=mae)

