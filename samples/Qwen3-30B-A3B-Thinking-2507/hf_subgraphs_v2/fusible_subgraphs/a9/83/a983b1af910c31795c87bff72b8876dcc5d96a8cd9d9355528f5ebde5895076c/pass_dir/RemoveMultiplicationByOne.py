import torch


def pattern(x):
    y = x * 1.0
    return y

def replacement_args(x):
    return (x,)

def replacement_func():
    def replacement(x):
        return x
    return replacement