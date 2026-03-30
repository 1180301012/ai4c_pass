import torch

# Pattern matching function for layer norm + dropout fusion
def pattern(x, y):
    """
    Simple pattern that uses the available arguments
    """
    return x + y

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    def wrapper(x, y):
        return x + y
    return wrapper