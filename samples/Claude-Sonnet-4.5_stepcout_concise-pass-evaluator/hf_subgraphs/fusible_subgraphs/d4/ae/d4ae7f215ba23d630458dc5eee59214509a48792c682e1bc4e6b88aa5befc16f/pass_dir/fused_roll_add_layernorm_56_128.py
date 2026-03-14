import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern for 56x56 with 128 channels and shift (3,3)
    """
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 56, 56, 128)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 3136, 128)
    tmp_6 = in_2 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (128,), in_1, in_0, 1e-05)
    return tmp_6, tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_roll_add_layernorm_kernel_56_128(
    input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    output_add_ptr,
    output_ln_ptr,
    N, C,
    H: tl.constexpr,
    W: tl.constexpr,
    shift_h: tl.constexpr,
    shift_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_idx = pid
    
    if seq_idx >= N:
        return
    
    # Compute rolled position
    h_idx = seq_idx // W
    w_idx = seq_idx % W
    h_rolled = (h_idx + shift_h) % H
    w_rolled = (w_idx + shift_w) % W
    rolled_idx = h_rolled * W + w_rolled
    
    # Load all channels for this position
    c_offsets = tl.arange(0, BLOCK_SIZE)
    c_mask = c_offsets < C
    
    input_idx = rolled_idx * C + c_offsets
    output_idx = seq_idx * C + c_offsets
    
    rolled_data = tl.load(input_ptr + input_idx, mask=c_mask, other=0.0)
    residual_data = tl.load(residual_ptr + output_idx, mask=c_mask, other=0.0)
    
    # Add
    add_result = residual_data + rolled_data
    tl.store(output_add_ptr + output_idx, add_result, mask=c_mask)
    
    # Layernorm: compute mean
    mean_val = tl.sum(add_result, axis=0) / C
    
    # Compute variance
    diff = add_result - mean_val
    var_val = tl.sum(diff * diff, axis=0) / C
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var_val + eps)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
    normalized = (add_result - mean_val) * rstd * weight + bias
    tl.store(output_ln_ptr + output_idx, normalized, mask=c_mask)

@torch.fx.wrap
def fused_roll_add_layernorm_wrapper_56_128(in_0, in_1, in_2, in_3):
    B, seq_len, C = in_2.shape
    H, W = 56, 56
    shift_h, shift_w = 3, 3
    
    in_3_reshaped = in_3.contiguous().view(-1, H, W, C)
    in_3_flat = in_3_reshaped.reshape(-1, C)
    
    output_add = torch.empty_like(in_2)
    output_ln = torch.empty_like(in_2)
    
    N = H * W
    BLOCK_SIZE = 128
    grid = (N,)
    
    fused_roll_add_layernorm_kernel_56_128[grid](
        in_3_flat,
        in_2,
        in_1,
        in_0,
        output_add,
        output_ln,
        N, C,
        H, W,
        shift_h, shift_w,
        1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (output_add, output_ln)

def replacement_func():
    return fused_roll_add_layernorm_wrapper_56_128