import torch

def pattern(in_0):
    tmp_0 = 0.0625 * in_0
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def simple_scale_triton(x):
        return 0.0625 * x
    return simple_scale_triton