import torch
import triton
import triton.language as tl


def pattern(in_5, in_1, in_2, in_4, in_3):
    """
    Match the pattern: adaptive_avg_pool2d -> batch_norm -> relu
    The batch_norm parameters are: running_mean, running_var, weight, bias
    followed by training=False, momentum=0.1, eps=1e-05
    """
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(in_5, (1, 1))
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=True)
    return tmp_8


def replacement_args(in_5, in_1, in_2, in_4, in_3):
    return (in_5, in_1, in_2, in_4, in_3)


@triton.jit
def fused_pool_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for adaptive_avg_pool2d + batch_norm + relu.
    
    adaptive_avg_pool2d reduces [N, C, H, W] -> [N, C, 1, 1]
    So for input [N, C, H, W], we compute:
    1. Average pool: avg over H*W elements for each N,C
    2. Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    3. ReLU: max(0, x)
    
    The pooled tensor [N, C, 1, 1] is flattened to [N*C] as:
    pos[n,c] = n * C + c
    """
    # Get program ID (each program handles one batch element)
    pid = tl.program_id(0)
    
    # Batch index
    n = pid
    
    # Process all channels for this batch element
    for c in range(C):
        idx = n * C + c
        
        # Load BN parameters for this channel
        mean = tl.load(mean_ptr + c)
        var = tl.load(var_ptr + c)
        weight = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        
        # Compute std with epsilon for numerical stability
        std = tl.sqrt(var + eps)
        
        # Load the pooled value at position [n, c]
        x = tl.load(x_ptr + idx)
        
        # Batch normalization with fused parameters
        x_norm = (x - mean) / std * weight + bias
        
        # ReLU activation
        x_out = tl.maximum(x_norm, 0.0)
        
        # Store result
        tl.store(out_ptr + idx, x_out)


@torch.fx.wrap
def fused_pool_bn_relu(x, mean, var, weight, bias, eps=1e-05, momentum=0.1):
    """
    Fused implementation of adaptive_avg_pool2d + batch_norm + relu.
    
    Args:
        x: Input tensor [N, C, H, W]
        mean: BN running mean [C]
        var: BN running var [C]
        weight: BN weight [C]
        bias: BN bias [C]
        eps: Batch norm epsilon (default: 1e-05)
        momentum: Batch norm momentum (default: 0.1)
    
    Returns:
        Output after adaptive_avg_pool2d + BN + ReLU: [N, C, 1, 1]
    """
    N, C, H, W = x.shape
    
    # Step 1: Adaptive Average Pooling to [N, C, 1, 1]
    # Using PyTorch's optimized implementation
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    
    # Flatten to [N*C] for efficient per-batch processing
    # After view(N*C): position [n, c, 0, 0] -> n*C + c
    pooled_flat = pooled.view(N * C)
    
    # Allocate output
    out_flat = torch.empty_like(pooled_flat)
    
    # Launch one program per batch element (N programs)
    grid = (N,)
    
    BLOCK_SIZE = 128
    
    fused_pool_bn_relu_kernel[grid](
        x_ptr=pooled_flat,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out_flat,
        N=N,
        C=C,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to [N, C, 1, 1]
    out = out_flat.view(N, C, 1, 1)
    
    return out


def replacement_func():
    return fused_pool_bn_relu