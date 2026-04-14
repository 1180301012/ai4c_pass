import torch

def pattern(in_0, in_1, in_2, in_3):
    return (in_2 + in_3,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    def wrapper(in_0, in_1, in_2, in_3):
        return (in_2 + in_3,)
    return wrapper