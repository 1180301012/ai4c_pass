import torch
import triton
import triton.language as tl

# Pattern: conv2d(in_5) with padding (1,1) + batch_norm + avg_pool2d(in_6)
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    tmp_5 = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.avg_pool2d(in_6, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

# Triton kernel for batch normalization (inference mode)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
)
@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index for each element
    c = (offsets // HW) % C
    
    # Load BN parameters for the channel
    mean = tl.load(mean_ptr + c, mask=mask)
    var = tl.load(var_ptr + c, mask=mask)
    gamma = tl.load(weight_ptr + c, mask=mask)
    beta = tl.load(bias_ptr + c, mask=mask)
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Batch norm computation: (x - mean) / sqrt(var + eps) * gamma + beta
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * gamma + beta
    
    # Store output
    tl.store(output_ptr + offsets, out, mask=mask)

# Triton kernel for 2x2 average pooling with stride 2, ceil_mode=True
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['n_out_elements'],
)
@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W, H_out, W_out,
    n_out_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out_elements
    
    # Compute output coordinates
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)
    
    # Compute input coordinates (top-left of 2x2 window)
    h_in = h_out * 2
    w_in = w_out * 2
    
    # Base index for this (n, c) slice
    base_idx = n * C * H * W + c * H * W
    
    # Handle boundary conditions for ceil_mode
    h0_valid = h_in < H
    h1_valid = (h_in + 1) < H
    w0_valid = w_in < W
    w1_valid = (w_in + 1) < W
    
    # Compute indices for 2x2 window
    idx00 = base_idx + h_in * W + w_in
    idx01 = base_idx + h_in * W + (w_in + 1)
    idx10 = base_idx + (h_in + 1) * W + w_in
    idx11 = base_idx + (h_in + 1) * W + (w_in + 1)
    
    # Load values with boundary checking
    v00 = tl.load(input_ptr + idx00, mask=mask & h0_valid & w0_valid, other=0.0)
    v01 = tl.load(input_ptr + idx01, mask=mask & h0_valid & w1_valid, other=0.0)
    v10 = tl.load(input_ptr + idx10, mask=mask & h1_valid & w0_valid, other=0.0)
    v11 = tl.load(input_ptr + idx11, mask=mask & h1_valid & w1_valid, other=0.0)
    
    # Count valid elements (count_include_pad=False)
    count = (h0_valid & w0_valid).to(tl.float32) + (h0_valid & w1_valid).to(tl.float32) + \
            (h1_valid & w0_valid).to(tl.float32) + (h1_valid & w1_valid).to(tl.float32)
    count = tl.maximum(count, 1.0)  # Avoid division by zero
    
    out = (v00 + v01 + v10 + v11) / count
    
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # in_0: running_mean [C]
    # in_1: running_var [C]
    # in_2: bias [C]
    # in_3: weight [C]
    # in_4: conv weights
    # in_5: conv input [N, C, H, W]
    # in_6: avg_pool input [N, C, H, W]
    
    # Conv2d - use cuDNN (highly optimized)
    conv_out = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    
    # Batch norm using Triton
    N, C, H, W = conv_out.shape
    bn_out = torch.empty_like(conv_out)
    
    n_elements = N * C * H * W
    HW = H * W
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    batch_norm_kernel[grid](
        conv_out, bn_out,
        in_0, in_1, in_3, in_2,  # Note order: mean, var, weight, bias
        n_elements, C, HW,
        eps=1e-05,
    )
    
    # Avg pool2d using Triton
    N_pool, C_pool, H_pool, W_pool = in_6.shape
    # ceil_mode=True: H_out = ceil(H/2)
    H_out = (H_pool + 1) // 2
    W_out = (W_pool + 1) // 2
    
    pool_out = torch.empty((N_pool, C_pool, H_out, W_out), device=in_6.device, dtype=in_6.dtype)
    
    n_out_elements = N_pool * C_pool * H_out * W_out
    grid_pool = lambda meta: ((n_out_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    avg_pool2d_kernel[grid_pool](
        in_6, pool_out,
        N_pool, C_pool, H_pool, W_pool, H_out, W_out,
        n_out_elements,
    )
    
    return (pool_out, bn_out)

def replacement_func():
    return optimized_forward