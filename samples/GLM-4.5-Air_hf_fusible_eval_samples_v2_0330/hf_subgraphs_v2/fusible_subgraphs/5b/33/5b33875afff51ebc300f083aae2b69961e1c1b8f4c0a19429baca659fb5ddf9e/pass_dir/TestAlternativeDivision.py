import torch

# Pattern for alternative division constant
def pattern(in_0):
    tmp_0 = in_0 / 2.8284271247461903
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    def alternative_division(in_0):
        return in_0 / 2.8284271247461903
    
    return alternative_division