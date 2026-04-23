import torch
import triton
import triton.language as tl


@triton.jit
def triton_batch_norm_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    B, C,
    eps: tl.constexpr,
):
    """
    Fused BatchNorm kernel for inference mode.
    Uses running mean/var instead of batch statistics.
    
    Formula: output = (input - mean) / sqrt(var + eps) * weight + bias
    
    - input: [B, C]
    - mean: [C] (running mean)
    - var: [C] (running variance)
    - weight: [C] (learnable scale)
    - bias: [C] (learnable bias)
    - output: [B, C]
    """
    # Grid: one program per output element
    pid = tl.program_id(0)
    
    # Calculate batch and channel indices
    c_idx = pid % C
    b_idx = pid // C
    
    # Bounds checking
    if b_idx >= B or c_idx >= C:
        return
    
    # Load running statistics for this channel
    mean = tl.load(mean_ptr + c_idx)
    var = tl.load(var_ptr + c_idx)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Load affine parameters
    weight = tl.load(weight_ptr + c_idx)
    bias = tl.load(bias_ptr + c_idx)
    
    # Load input
    x = tl.load(input_ptr + b_idx * C + c_idx)
    
    # Normalize and apply affine transform
    x_norm = (x - mean) * inv_std
    y = x_norm * weight + bias
    
    # Store result
    tl.store(output_ptr + b_idx * C + c_idx, y)


@torch.fx.wrap
def triton_batch_norm(
    input: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-05
) -> torch.Tensor:
    """
    Fused batch norm operation for inference mode.
    Uses running statistics instead of batch statistics.
    """
    B, C = input.shape
    
    # Allocate output
    output = torch.empty_like(input)
    
    # Grid: B * C programs (one per output element)
    grid = (B * C,)
    
    triton_batch_norm_kernel[grid](
        input, running_mean, running_var, weight, bias, output,
        B, C,
        eps,
    )
    
    return output


def pattern(in_7: torch.Tensor, in_0: torch.Tensor, in_1: torch.Tensor, in_3: torch.Tensor, in_2: torch.Tensor):
    """
    Match the batch_norm operation:
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    Args:
    - in_7: input [B, 384]
    - in_0: running_mean [384]
    - in_1: running_var [384]
    - in_3: weight [384]
    - in_2: bias [384]
    - False: training flag
    - 0.1: momentum (ignored for inference)
    - 1e-05: eps
    """
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_7


def replacement_args(in_7: torch.Tensor, in_0: torch.Tensor, in_1: torch.Tensor, in_3: torch.Tensor, in_2: torch.Tensor):
    return (in_7, in_0, in_1, in_3, in_2)


def replacement_func():
    return triton_batch_norm