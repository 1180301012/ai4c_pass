import torch

# Simple test pattern: division (handles different constants)
def pattern(in_0):
    tmp_0 = in_0 / 8.0
    return tmp_0

# Alternative pattern for different division constant
def pattern_v2(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def simple_division(in_0):
        return in_0 / 8.0
    
    return simple_division