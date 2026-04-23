import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the dataflow exactly without cleanup statements
@torch.fx.wrap
def pattern(in_2, in_3, in_0, in_1):
    # Match the exact operation sequence in model.py
    tmp = in_2 + in_3
    reshaped = tmp.reshape(-1, 768)
    out = torch.nn.functional.layer_norm(reshaped, (768,), in_1, in_0, 1e-05)
    return reshaped, out


# Argument extraction function
# Returns arguments required for replacement
def replacement_args(in_2, in_3, in_0, in_1):
    return (in_2, in_3, in_0, in_1)


# Optimized Triton kernel
@triton.jit
@torch.fx.wrap
def fused_layer_norm_kernel(
    in_2_ptr, in_3_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, eps, BLOCK_SIZE: tl.constexpr
):
    
    # Compute mean and variance for each row in the N dimension
    block_row = tl.program_id(0)
    
    # First pass: compute row sum (mean)
    total = 0.0
    for col in range(C):
        idx = block_row * C + col
        in2 = tl.load(in_2_ptr + idx, mask=(col < C), other=0.0)
        in3 = tl.load(in_3_ptr + idx, mask=(col < C), other=0.0)
        total += in2 + in3
    
    mean = total / C
    
    # Second pass: compute variance
    variance = 0.0
    for col in range(C):
        idx = block_row * C + col
        in2 = tl.load(in_2_ptr + idx, mask=(col < C), other=0.0)
        in3 = tl.load(in_3_ptr + idx, mask=(col < C), other=0.0)
        x = in2 + in3
        diff = x - mean
        variance += diff * diff
    variance = variance / C + eps
    
    # Third pass: apply normalization
    for col in range(C):
        idx = block_row * C + col
        in2 = tl.load(in_2_ptr + idx, mask=(col < C), other=0.0)
        in3 = tl.load(in_3_ptr + idx, mask=(col < C), other=0.0)
        x = in2 + in3
        normalized = (x - mean) * tl.rsqrt(variance)
        weight = tl.load(weight_ptr + col)
        bias = tl.load(bias_ptr + col)
        out = normalized * weight + bias
        tl.store(out_ptr + idx, out, mask=(col < C))


# Kernel wrapper with proper memory handling
@torch.fx.wrap
def fused_layer_norm(in_2, in_3, in_0, in_1):
    N = in_2.numel() // 768  # Compute total elements / channel size
    C = 768
    eps = 1e-05
    
    # Create output tensor (will be the same shape as in_2)
    out = torch.empty_like(in_2)
    
    # Determine grid configuration
    grid = (N, )
    BLOCK_SIZE = 128  # Tile size for channel dimension
    
    # Launch kernel
    fused_layer_norm_kernel[grid](
        in_2, in_3, in_1, in_0,
        out,
        N, C, eps, BLOCK_SIZE
    )
    
    # Return the reshaped tensor (tmp_3) and normalized tensor (tmp_4)
    # Note: The reshape happens automatically via our output tensor's shape
    return out.view(N, C), out


# Replacement function
def replacement_func():
    return fused_layer_norm