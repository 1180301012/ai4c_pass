import torch
import triton
import triton.language as tl


def pattern(seq_output):
    """
    Pattern matching the mean(-2) computation.
    """
    mean_out = seq_output.mean(-2)
    return mean_out


def replacement_args(seq_output):
    return (seq_output,)


@torch.fx.wrap
def optimized_mean(seq_output):
    """
    Optimized mean using einsum for potential fusion benefits.
    """
    # seq_output shape: [N, R, K] -> [N, K]
    # einsum computes sum and we divide manually
    N, R, K = seq_output.shape
    # Alternative: use reshape and matmul
    # Reshape to [N*K, R], multiply with ones(R)/R, reshape back to [N, K]
    result = seq_output.sum(dim=1)
    result.mul_(1.0 / R)
    return result


def replacement_func():
    return optimized_mean