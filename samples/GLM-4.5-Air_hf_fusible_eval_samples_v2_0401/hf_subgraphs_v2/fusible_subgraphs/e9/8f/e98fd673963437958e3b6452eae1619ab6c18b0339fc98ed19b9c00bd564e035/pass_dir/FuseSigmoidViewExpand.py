import torch

def pattern(in_2, in_1):
    # Match the sequence: sigmoid -> view -> expand_as
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    return tmp_2

def replacement_args(in_2, in_1):
    return (in_2, in_1)

def optimized_sigmoid_broadcast(in_2, in_1):
    """
    Simple optimized direct computation.
    """
    # Direct sigmoid and reshape for broadcasting
    return in_2.sigmoid().view(1, -1, 1, 1)

def replacement_func():
    return optimized_sigmoid_broadcast