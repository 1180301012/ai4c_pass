import torch

# Very simple pattern to test the basic mechanism
def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)

def replacement_args(bias, weight, input_tensor, sequence_output):
    return (input_tensor, weight, bias)

def replacement_func():
    def simple_linear(x, weight, bias):
        # Use allowed operations: matrix multiplication + addition
        return x @ weight.T + bias
    return simple_linear