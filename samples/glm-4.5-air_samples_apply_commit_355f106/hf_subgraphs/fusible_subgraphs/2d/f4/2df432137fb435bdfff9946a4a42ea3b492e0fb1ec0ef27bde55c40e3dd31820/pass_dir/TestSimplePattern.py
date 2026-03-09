import torch

def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    def add_function(x, y):
        return x + y
    return add_function