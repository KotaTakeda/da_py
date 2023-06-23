import itertools
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
from params import prod_params_kv


def gen_assim_func(system, fixed_params, y, dt):
    def assimilate(params):
        instance = system(**{**fixed_params, **params})
        for y_obs in y:
            instance.forecast(dt)
            instance.update(y_obs)
        return instance.x

    return assimilate


def gen_eval_func(system, fixed_params, y, dt, eval):
    def eval_func(params):
        instance = system(**{**fixed_params, **params})
        for y_obs in y:
            instance.forecast(dt)
            instance.update(y_obs)
        return eval(instance.x)

    return eval_func


def grid_search2(
    evaluate, param1_name, params1, param2_name, params2, n_multi=1, multi_type="thread"
):
    params_kv = prod_params_kv(param1_name, params1, param2_name, params2)
    if n_multi > 0:
        if multi_type == "process":
            errors = exec_mp(evaluate, params_kv, n_multi)
        else:
            errors = exec_mt(evaluate, params_kv, n_multi)
    else:
        errors = list(map(evaluate, params_kv))
    errors = np.array(errors).reshape(len(params1), len(params2))
    df = pd.DataFrame(errors, index=params1, columns=params2)
    return df


def exec_mp(func, params, n_process):
    with get_context("fork").Pool(n_process) as pl:
        results = np.array(pl.map(func, params))
        pl.close()
        pl.join()
    return results


def exec_mt(func, params, n_workers):
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(func, params)
    return results
