import torch

def pattern(in_0, in_1, tmp_2):
    # Match the sequence: multiply -> add -> relu
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0  # Note: original uses += but pattern matching should match the dataflow
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    return tmp_5

def replacement_args(in_0, in_1, tmp_2):
    return (in_0, in_1, tmp_2)

@torch.fx.wrap
def fused_multiply_add_relu(in_0, in_1, sigmoid_broadcasted):
    """
    Fuse multiply + add operations to eliminate intermediate tensors.
    The broadcasting from sigmoid_broadcasted enables efficient computation.
    """
    # Direct fusion: in_0 + in_1 * sigmoid_broadcasted
    # The broadcasting happens automatically during multiplication
    # This eliminates the intermediate 'multiplied' and 'added' tensors from the original
    return in_0 + in_1 * sigmoid_broadcasted

def replacement_func():
    return fused_multiply_add_relu