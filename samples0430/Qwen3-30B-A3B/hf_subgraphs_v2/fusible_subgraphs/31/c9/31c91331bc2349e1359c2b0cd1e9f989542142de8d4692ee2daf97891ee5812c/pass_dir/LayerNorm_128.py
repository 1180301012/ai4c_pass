import torch
import triton
import triton.language as tl

def pattern(input_tensor, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(input_tensor, normalized_shape, weight, bias, eps)

def replacement_args(input_tensor, normalized_shape, weight, bias, eps):
    return (input_tensor, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    H,
    W,
    C,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    c_start = c_block * BLOCK_SIZE_C
    c_mask = c_start + tl.arange(0, BLOCK_SIZE_C) < C
    
    # Load weight and bias for this channel block
    weight = tl.load(weight_ptr + c_start, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_start, mask=c_mask, other=0.0)
    
    # Compute mean and variance for channel block
    mean = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                idx = b * H * W * C + h * W * C + w * C + c_start
                val = tl.load(input_ptr + idx, mask=c_mask, other=0.0)
                mean += val
    mean /= (B * H * W)

    var = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                idx = b * H * W * C + h * W * C + w * C + c_start
                val = tl.load(input_ptr + idx, mask=c_mask, other=0.0)
                var += (val - mean) * (val - mean)
    var /= (B * H * W)
    
    # Normalize and scale
    for b in range(B):
        for h in range(H):
            for w in range(W):
                idx = b * H * W * C + h * W * C + w * C + c_start
                val = tl.load(input_ptr + idx, mask=c_mask, other=0.0)
                normalized = (val - mean) / tl.sqrt(var + eps)
                output_val = normalized * weight + bias
                tl.store(output_ptr + idx, output_val, mask=c_mask)

@torch.fx.wrap
def layer_norm_wrapper(input_tensor, weight, bias, eps):
    # Extract dimensions
    B, H, W, C = input_tensor.shape
    eps_val = eps.item() if isinstance(eps, torch.Tensor) else eps
    
    # Choose block size based on C (for optimal occupancy)
    BLOCK_SIZE_C = 128
    num_blocks = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    layer_norm_kernel[(num_blocks,)](
        input_tensor,
        weight,
        bias,
        output,
        B,
        H,
        W,
        C,
        eps_val,
        BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return layer_norm_wrapper