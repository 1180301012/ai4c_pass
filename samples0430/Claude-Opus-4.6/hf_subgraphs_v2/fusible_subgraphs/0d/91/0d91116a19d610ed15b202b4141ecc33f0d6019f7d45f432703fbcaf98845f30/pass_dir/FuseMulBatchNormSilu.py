import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5 * in_4
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
    ],
    key=['HW'],
)
@triton.jit
def fused_mul_bn_silu_kernel(
    x_ptr,
    sigmoid_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (num_spatial_blocks * B * C,)
    # Each program handles one spatial block for one (batch, channel) pair
    pid = tl.program_id(0)
    
    # Compute number of spatial blocks per (b,c) pair
    num_spatial_blocks = tl.cdiv(HW, BLOCK_SIZE)
    
    # Determine which (b,c) pair and which spatial block
    bc_id = pid // num_spatial_blocks
    spatial_block_id = pid % num_spatial_blocks
    
    c = bc_id % C
    
    # Load per-channel parameters (in float32 for accuracy)
    mean = tl.load(running_mean_ptr + c).to(tl.float32)
    var = tl.load(running_var_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    bi = tl.load(bias_ptr + c).to(tl.float32)
    
    # Load sigmoid value for this (b, c) pair
    sig = tl.load(sigmoid_ptr + bc_id).to(tl.float32)
    
    # Precompute batch norm affine transform
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale = w * inv_std
    shift = bi - mean * scale
    
    # Combined: x * sigmoid * bn_scale + bn_shift
    combined_scale = sig * scale
    
    # Process spatial elements
    offsets = spatial_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    
    # Base offset for this (b, c) in x (NCHW layout, contiguous)
    base = bc_id * HW
    
    # Load x values
    x = tl.load(x_ptr + base + offsets, mask=mask).to(tl.float32)
    
    # Fused computation: mul + batch_norm + silu
    normed = x * combined_scale + shift
    
    # SiLU: x * sigmoid(x)
    result = normed * tl.sigmoid(normed)
    
    # Store (auto-cast back to output dtype)
    tl.store(out_ptr + base + offsets, result, mask=mask)


@torch.fx.wrap
def fused_mul_bn_silu(running_mean, running_var, bias, weight, sigmoid, x):
    B = x.shape[0]
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    
    # Ensure params are on same device as x
    device = x.device
    running_mean = running_mean.to(device)
    running_var = running_var.to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    
    # Ensure contiguous layout
    x = x.contiguous()
    sigmoid = sigmoid.contiguous()
    
    out = torch.empty_like(x)
    
    # Grid: total blocks = num_spatial_blocks * B * C
    grid = lambda meta: ((((HW + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']) * B * C),)
    
    fused_mul_bn_silu_kernel[grid](
        x, sigmoid, running_mean, running_var, weight, bias, out,
        C, HW,
    )
    
    return out


def replacement_func():
    return fused_mul_bn_silu