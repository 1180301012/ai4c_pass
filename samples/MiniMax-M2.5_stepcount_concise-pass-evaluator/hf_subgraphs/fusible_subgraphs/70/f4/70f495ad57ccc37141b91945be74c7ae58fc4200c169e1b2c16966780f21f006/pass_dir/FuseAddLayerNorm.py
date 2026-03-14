import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Fuse Add + LayerNorm
    - in_0: bias ([1280])
    - in_1: weight ([1280]) 
    - in_2: input tensor 1 ([B, 192, 1280])
    - in_3: input tensor 2 ([B, 192, 1280])
    
    Operations:
    1. tmp_2 = in_3 + in_2  # element-wise add
    2. tmp_3 = layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    
    Note: tmp_2 is set to None right after layer_norm, so only tmp_3 is observable.
    """
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1280,), in_1, in_0, 1e-06)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments needed for the fused kernel.
    - in_0: bias for layer norm
    - in_1: weight for layer norm
    - in_2: input tensor 1
    - in_3: input tensor 2
    """
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 512, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8}),
    ],
    key=['N'],
)
@triton.jit
def fused_add_ln_kernel(
    input1_ptr, input2_ptr, output_ptr, 
    weight_ptr, bias_ptr,
    B, H, N,  # B=batch, H=192, N=1280 (N is the normalized dim)
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Fused Add + LayerNorm kernel.
    
    LayerNorm with normalized_shape=(N,) on input [B, H, N] computes
    mean and variance per H position, i.e., for each of the H vectors of size N.
    """
    # Grid: (B, H) - each program handles one batch and one H position
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    
    row_offset = (batch_idx * H + h_idx) * N
    
    # Compute sum for this specific H position
    sum_acc = 0.0
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        
        x1 = tl.load(input1_ptr + row_offset + offs, mask=mask, other=0.0)
        x2 = tl.load(input2_ptr + row_offset + offs, mask=mask, other=0.0)
        x = x1 + x2
        sum_acc += tl.sum(x, axis=0)
    
    mean = sum_acc / N
    
    # Compute variance for this specific H position
    var_acc = 0.0
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        
        x1 = tl.load(input1_ptr + row_offset + offs, mask=mask, other=0.0)
        x2 = tl.load(input2_ptr + row_offset + offs, mask=mask, other=0.0)
        x = x1 + x2
        diff = x - mean
        var_acc += tl.sum(diff * diff, axis=0)
    
    # Compute std with eps
    var = var_acc / N
    std = tl.sqrt(var + eps)
    inv_std = 1.0 / std
    
    # Normalize and apply weight + bias
    for n in range(0, N, BLOCK_SIZE):
        offs = n + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        
        x1 = tl.load(input1_ptr + row_offset + offs, mask=mask, other=0.0)
        x2 = tl.load(input2_ptr + row_offset + offs, mask=mask, other=0.0)
        x = x1 + x2
        
        # Normalize: (x - mean) / std
        normalized = (x - mean) * inv_std
        
        # Load weight and bias
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        
        # Apply affine transform: normalized * weight + bias
        out = normalized * w + b
        
        tl.store(output_ptr + row_offset + offs, out, mask=mask)


def fused_add_ln(input1, input2, weight, bias, eps=1e-6):
    """
    Wrapper function to launch the fused Add + LayerNorm kernel.
    """
    B, H, N = input1.shape
    assert input2.shape == (B, H, N), "Input shapes must match"
    assert weight.shape == (N,), "Weight shape must match normalized dim"
    assert bias.shape == (N,), "Bias shape must match normalized dim"
    
    output = torch.empty_like(input1)
    
    # Grid: (B, H) - each program handles one batch element and one H position
    grid = (B, H)
    
    fused_add_ln_kernel[grid](
        input1, input2, output,
        weight, bias,
        B, H, N,
        eps,
    )
    
    return output


@torch.fx.wrap
def fused_add_ln_wrapper(in_0, in_1, in_2, in_3):
    """
    FX wrap for the fused kernel.
    in_0: bias ([N])
    in_1: weight ([N])
    in_2: input1 ([B, H, N])
    in_3: input2 ([B, H, N])
    
    Returns: layer_norm_result only (tmp_3 is what gets passed forward)
    """
    ln_result = fused_add_ln(in_2, in_3, in_1, in_0, eps=1e-6)
    
    return ln_result


def replacement_func():
    """
    Returns the replacement function for the fused Add + LayerNorm operation.
    """
    return fused_add_ln_wrapper