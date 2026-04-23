import torch
import triton
import triton.language as tl

# Pattern: cat -> batch_norm -> silu -> mean(keepdim=True)
# Matches the entire post-conv2d pipeline

def pattern(running_mean, running_var, bias, weight, cat_input):
    bn_out = torch.nn.functional.batch_norm(cat_input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_out = torch.nn.functional.silu(bn_out, inplace=True)
    mean_out = silu_out.mean((2, 3), keepdim=True)
    return silu_out, mean_out

def replacement_args(running_mean, running_var, bias, weight, cat_input):
    return (running_mean, running_var, bias, weight, cat_input, "fuse_bn_silu_mean")

@triton.jit
def bn_silu_mean_kernel(
    input_ptr,          # [N, C, H, W]
    running_mean_ptr,   # [C]
    running_var_ptr,    # [C]
    weight_ptr,         # [C]
    bias_ptr,           # [C]
    silu_out_ptr,       # [N, C, H, W] output
    mean_out_ptr,       # [N, C, 1, 1] output
    N, C, H, W,
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # This kernel processes batches of channels
    pid_c = tl.program_id(0)  # which channel block
    pid_n = tl.program_id(1)  # which batch block
    
    c_start = pid_c * BLOCK_C
    n_start = pid_n * BLOCK_N
    
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    # Load BN parameters for this channel block
    c_mask = c_offsets < C
    rm = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    rv = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=1.0)
    w = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0)
    b = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    
    # Compute BN scale and offset: (x - mean) / sqrt(var + eps) * weight + bias
    # = x * (weight / sqrt(var + eps)) + (bias - mean * weight / sqrt(var + eps))
    inv_std = w / tl.sqrt(rv + eps)
    bn_offset = b - rm * inv_std
    
    # Compute mean accumulator for each (n, c) pair
    # mean over HW dimensions
    hw_size = H * W
    
    n_mask = n_offsets < N
    
    mean_acc = tl.zeros((BLOCK_N, BLOCK_C), dtype=tl.float32)
    
    # Iterate over HW in blocks
    num_hw_blocks = (hw_size + BLOCK_HW - 1) // BLOCK_HW
    for hw_block in range(num_hw_blocks):
        hw_start = hw_block * BLOCK_HW
        hw_offsets_local = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets_local < hw_size
        
        # Compute flat indices for input: n*C*H*W + c*H*W + h*W + w
        # h = hw_offset // W, w = hw_offset % W
        # We need to load input[n, c, h, w] for all n, c in block and hw in sub-block
        
        # input index: n_offsets[:, None, None] * C*H*W + c_offsets[None, :, None] * H*W + hw_offsets_local[None, None, :]
        input_idx = (n_offsets[:, None, None] * (C * hw_size) +
                     c_offsets[None, :, None] * hw_size +
                     hw_offsets_local[None, None, :])
        
        full_mask = (n_mask[:, None, None] & c_mask[None, :, None] & hw_mask[None, None, :])
        
        x = tl.load(input_ptr + input_idx, mask=full_mask, other=0.0)
        
        # Apply BN: x * inv_std + bn_offset
        bn_x = x * inv_std[None, :, None] + bn_offset[None, :, None]
        
        # Apply SiLU: bn_x * sigmoid(bn_x) = bn_x / (1 + exp(-bn_x))
        silu_x = bn_x * tl.sigmoid(bn_x)
        
        # Store silu output
        tl.store(silu_out_ptr + input_idx, silu_x.to(input_ptr.dtype.element_ty), mask=full_mask)
        
        # Accumulate for mean
        mean_acc += tl.sum(silu_x, axis=2)  # sum over HW
    
    # Compute mean = sum / (H*W)
    mean_val = mean_acc / hw_size
    
    # Store mean output: [N, C, 1, 1] -> mean_out[n*C + c]
    mean_idx = n_offsets[:, None] * C + c_offsets[None, :]
    mean_mask_2d = n_mask[:, None] & c_mask[None, :]
    tl.store(mean_out_ptr + mean_idx, mean_val.to(input_ptr.dtype.element_ty), mask=mean_mask_2d)


@torch.fx.wrap
def fused_bn_silu_mean(running_mean, running_var, bias, weight, cat_input, route=""):
    if route != "fuse_bn_silu_mean":
        # This shouldn't happen for this pass, but handle gracefully
        # Fall back to original implementation
        bn_out = torch.nn.functional.batch_norm(cat_input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
        silu_out = torch.nn.functional.silu(bn_out, inplace=True)
        mean_out = silu_out.mean((2, 3), keepdim=True)
        return silu_out, mean_out
    
    N, C, H, W = cat_input.shape
    eps = 1e-05
    
    silu_out = torch.empty_like(cat_input)
    mean_out = torch.empty((N, C, 1, 1), dtype=cat_input.dtype, device=cat_input.device)
    
    BLOCK_C = 16
    BLOCK_N = 1
    BLOCK_HW = 256
    
    grid_c = (C + BLOCK_C - 1) // BLOCK_C
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    bn_silu_mean_kernel[(grid_c, grid_n)](
        cat_input, running_mean, running_var, weight, bias,
        silu_out, mean_out,
        N, C, H, W, eps,
        BLOCK_N=BLOCK_N, BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C,
    )
    
    return silu_out, mean_out

def replacement_func():
    return fused_bn_silu_mean