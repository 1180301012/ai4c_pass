import torch

def pattern(tensor):
    """Simple tensor indexing pattern that should exist in any graph"""
    return tensor[0]

def replacement_args(tensor):
    return (tensor,)

def replacement_func():
    def simple_index(tensor):
        return tensor[0]
    return simple_index