import itertools


def prod_params_kv(param1_name, params1, param2_name, params2):
    params = itertools.product(params1, params2)
    params_kv = [{param1_name: param[0], param2_name: param[1]} for param in params]
    return params_kv
