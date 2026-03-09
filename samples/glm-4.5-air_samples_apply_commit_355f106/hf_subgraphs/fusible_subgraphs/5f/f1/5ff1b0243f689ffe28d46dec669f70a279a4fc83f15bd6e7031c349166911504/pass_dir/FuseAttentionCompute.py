import torch

def pattern(in_0, in_1):
    # Simple pattern - just add the two inputs
    return in_0 + in_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    # Simple replacement function that adds the inputs without Triton
    def simple_add(in_0, in_1):
        return in_0 + in_1
    return simple_add