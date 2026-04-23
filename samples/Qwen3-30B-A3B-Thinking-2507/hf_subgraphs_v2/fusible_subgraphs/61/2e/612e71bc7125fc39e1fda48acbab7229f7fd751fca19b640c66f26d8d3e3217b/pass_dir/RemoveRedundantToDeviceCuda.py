import torch

def pattern(in_0):
    t = in_0.t()
    to_cuda = t.to(device(type='cuda'))
    return to_cuda

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def remove_redundant_to_cuda(in_0):
        return in_0.t()
    return remove_redundant_to_cuda