import torch

def pattern(in_0, in_1, in_2, in_3):
    # Simple addition test
    return (in_2 + in_3,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    def simple_add(in_0, in_1, in_2, in_3):
        return (in_2 + in_3,)
    return simple_add