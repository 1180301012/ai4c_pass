import torch
import triton
import triton.language as tl


@triton.jit
def window_roll_ln_add_kernel_32x768(
    # Input pointer (in_3 with shape [1, 4, 8, 4, 8, 768])
    in_3_ptr,
    # Residual pointer (in_2 with shape [1, 1024, 768])
    residual_ptr,
    # Weight and bias pointers
    weight_ptr,
    bias_ptr,
    # Output pointer
    output_ptr,
    eps: tl.constexpr,
):
    """
    Fused kernel for 32x32 window with 768 channels.
    Input: in_3 with shape [1, 4, 8, 4, 8, 768]
    Applies: view->roll(4,4)->view->layer_norm->add residual
    Output: [1, 1024, 768]
    """
    pid = tl.program_id(0)
    if pid >= 1024:
        return
    
    # Channel offsets - use 1024 (power of 2) for arange, mask to 768
    c_offsets = tl.arange(0, 1024)
    c_mask = c_offsets < 768
    
    # Load weight and bias
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    
    # Compute h, w from pid (output position in [1, 1024, 768])
    h = pid // 32
    w = pid % 32
    
    # Accumulate mean and variance
    sum_vals = tl.zeros((1024,), dtype=tl.float32)
    sum_sq = tl.zeros((1024,), dtype=tl.float32)
    
    # For each (h, w) in the window (32x32 = 1024 elements)
    for i in range(1024):
        local_h = i // 32
        local_w = i % 32
        # Apply roll: shift by (4, 4)
        rolled_local_h = (local_h + 28) % 32
        rolled_local_w = (local_w + 28) % 32
        
        # Source index computation for [1, 4, 8, 4, 8, 768] -> [1, 32, 32, 768]
        src_local_d1 = rolled_local_h // 8
        src_local_d3 = rolled_local_h % 8
        src_local_d2 = rolled_local_w // 8
        src_local_d4 = rolled_local_w % 8
        
        # Flat index in original tensor
        src_local_flat = src_local_d1 * 196608 + src_local_d2 * 24576 + src_local_d3 * 6144 + src_local_d4 * 768
        
        # Load from input
        x_vals = tl.load(in_3_ptr + src_local_flat + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        # Load from residual
        residual_vals = tl.load(residual_ptr + pid * 768 + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        
        sum_vals += x_vals + residual_vals
        sum_sq += (x_vals + residual_vals) * (x_vals + residual_vals)
    
    # Compute mean and variance
    mean = sum_vals / 1024.0
    var = (sum_sq / 1024.0) - (mean * mean)
    inv_std = tl.rsqrt(var + eps)
    
    # Final normalization - reload values
    sum_final = tl.zeros((1024,), dtype=tl.float32)
    for i in range(1024):
        local_h = i // 32
        local_w = i % 32
        rolled_local_h = (local_h + 28) % 32
        rolled_local_w = (local_w + 28) % 32
        
        src_local_d1 = rolled_local_h // 8
        src_local_d3 = rolled_local_h % 8
        src_local_d2 = rolled_local_w // 8
        src_local_d4 = rolled_local_w % 8
        
        src_local_flat = src_local_d1 * 196608 + src_local_d2 * 24576 + src_local_d3 * 6144 + src_local_d4 * 768
        
        x_vals = tl.load(in_3_ptr + src_local_flat + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        residual_vals = tl.load(residual_ptr + pid * 768 + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        sum_final += x_vals + residual_vals
    
    normalized = (sum_final / 1024.0 - mean) * inv_std
    output = normalized * weight + bias_val
    
    tl.store(output_ptr + pid * 768 + c_offsets, output, mask=c_mask)


@triton.jit
def window_roll_ln_add_kernel_64x384(
    # Input pointer (in_3 with shape [1, 8, 8, 8, 8, 384])
    in_3_ptr,
    # Residual pointer (in_2 with shape [1, 4096, 384])
    residual_ptr,
    # Weight and bias pointers
    weight_ptr,
    bias_ptr,
    # Output pointer
    output_ptr,
    eps: tl.constexpr,
):
    """
    Fused kernel for 64x64 window with 384 channels.
    Input: in_3 with shape [1, 8, 8, 8, 8, 384]
    Applies: view->roll(4,4)->view->layer_norm->add residual
    Output: [1, 4096, 384]
    """
    pid = tl.program_id(0)
    if pid >= 4096:
        return
    
    # Channel offsets - use 512 (power of 2) for arange, mask to 384
    c_offsets = tl.arange(0, 512)
    c_mask = c_offsets < 384
    
    # Load weight and bias
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    
    # Compute h, w from pid (output position in [1, 4096, 384])
    h = pid // 64
    w = pid % 64
    
    # Accumulate mean and variance (64x64 = 4096 elements)
    sum_vals = tl.zeros((512,), dtype=tl.float32)
    sum_sq = tl.zeros((512,), dtype=tl.float32)
    
    for i in range(4096):
        local_h = i // 64
        local_w = i % 64
        # Apply roll: shift by (4, 4)
        rolled_local_h = (local_h + 60) % 64
        rolled_local_w = (local_w + 60) % 64
        
        # Source index computation for [1, 8, 8, 8, 8, 384] -> [1, 64, 64, 384]
        src_local_d1 = rolled_local_h // 8
        src_local_d3 = rolled_local_h % 8
        src_local_d2 = rolled_local_w // 8
        src_local_d4 = rolled_local_w % 8
        
        # Flat index in original tensor
        src_local_flat = src_local_d1 * 196608 + src_local_d2 * 24576 + src_local_d3 * 3072 + src_local_d4 * 384
        
        x_vals = tl.load(in_3_ptr + src_local_flat + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        residual_vals = tl.load(residual_ptr + pid * 384 + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        
        sum_vals += x_vals + residual_vals
        sum_sq += (x_vals + residual_vals) * (x_vals + residual_vals)
    
    mean = sum_vals / 4096.0
    var = (sum_sq / 4096.0) - (mean * mean)
    inv_std = tl.rsqrt(var + eps)
    
    # Final normalization
    sum_final = tl.zeros((512,), dtype=tl.float32)
    for i in range(4096):
        local_h = i // 64
        local_w = i % 64
        rolled_local_h = (local_h + 60) % 64
        rolled_local_w = (local_w + 60) % 64
        
        src_local_d1 = rolled_local_h // 8
        src_local_d3 = rolled_local_h % 8
        src_local_d2 = rolled_local_w // 8
        src_local_d4 = rolled_local_w % 8
        
        src_local_flat = src_local_d1 * 196608 + src_local_d2 * 24576 + src_local_d3 * 3072 + src_local_d4 * 384
        
        x_vals = tl.load(in_3_ptr + src_local_flat + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        residual_vals = tl.load(residual_ptr + pid * 384 + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        sum_final += x_vals + residual_vals
    
    normalized = (sum_final / 4096.0 - mean) * inv_std
    output = normalized * weight + bias_val
    
    tl.store(output_ptr + pid * 384 + c_offsets, output, mask=c_mask)


@torch.fx.wrap
def fused_window_roll_ln_add_32x768(in_3, in_2, in_1, in_0):
    """
    Wrapper for 32x32 window with 768 channels.
    Input shapes: in_3=[1,4,8,4,8,768], in_2=[1,1024,768], in_1=[768], in_0=[768]
    """
    output = torch.empty_like(in_2)
    grid = (1024,)
    
    window_roll_ln_add_kernel_32x768[grid](
        in_3, in_2, in_1, in_0, output, 1e-05
    )
    return output


@torch.fx.wrap
def fused_window_roll_ln_add_64x384(in_3, in_2, in_1, in_0):
    """
    Wrapper for 64x64 window with 384 channels.
    Input shapes: in_3=[1,8,8,8,8,384], in_2=[1,4096,384], in_1=[384], in_0=[384]
    """
    output = torch.empty_like(in_2)
    grid = (4096,)
    
    window_roll_ln_add_kernel_64x384[grid](
        in_3, in_2, in_1, in_0, output, 1e-05
    )
    return output


def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching: contiguous -> view -> roll -> view -> layer_norm -> add
    Matches the 32x32, 768 channel variant
    """
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_window_roll_ln_add_32x768