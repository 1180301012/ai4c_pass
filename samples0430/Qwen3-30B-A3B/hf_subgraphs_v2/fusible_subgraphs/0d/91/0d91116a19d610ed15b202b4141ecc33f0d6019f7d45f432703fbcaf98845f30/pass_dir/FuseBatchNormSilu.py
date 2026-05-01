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

@triton.jit
def fused_bn_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    total_elements = B * C * H * W
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, total_elements)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < block_end
    
    element_indices = block_start + tl.arange(0, BLOCK_SIZE)
    channels = element_indices // (B * H * W)
    
    x = tl.load(x_ptr + offsets, mask=mask)
    running_mean = tl.load(running_mean_ptr + channels, mask=mask)
    running_var = tl.load(running_var_ptr + channels, mask=mask)
    weight = tl.load(weight_ptr + channels, mask=mask)
    bias = tl.load(bias_ptr + channels, mask=mask)
    
    norm_val = (x - running_mean) / tl.sqrt(running_var + eps)
    norm_val = norm_val * weight + bias
    sigmoid_val = 1.0 / (1.0 + tl.exp(-norm_val))
    result = norm_val * sigmoid_val
    
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_bn_silu(in_0, in_1, in_2, in_3, in_4, in_5):
    x = in_5 * in_4
    B, C, H, W = x.shape
    total_elements = B * C * H * W
    out = torch.empty_like(x)
    BLOCK_SIZE = 128
    eps = 1e-05
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_bn_silu_kernel[(num_blocks,)](
        x_ptr=x,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

@torch.fx.wrap
def wrapper(*args):
    return fused_bn_silu(*args)

def replacement_func():
    return wrapper