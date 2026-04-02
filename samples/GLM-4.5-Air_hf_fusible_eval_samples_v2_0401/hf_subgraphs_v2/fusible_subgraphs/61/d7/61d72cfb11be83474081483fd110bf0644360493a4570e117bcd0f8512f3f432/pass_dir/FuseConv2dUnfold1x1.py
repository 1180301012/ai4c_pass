import torch

def pattern(x):
    return x.reshape(1, 128, 4, -1)

def replacement_args(x):
    return (x,)

def replacement_func():
    def optimized_function(x):
        return x.reshape(1, 128, 4, -1)
    return optimized_function