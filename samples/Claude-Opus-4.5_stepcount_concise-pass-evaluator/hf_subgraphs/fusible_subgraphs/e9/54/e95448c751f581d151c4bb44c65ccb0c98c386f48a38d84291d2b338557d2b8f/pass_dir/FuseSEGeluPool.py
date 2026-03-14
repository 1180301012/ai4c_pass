import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match SE + GELU + AvgPool computation.
    in_0 = bias [1024]
    in_1 = weight [1024, 64, 1, 1]
    in_2 = feature map [B, 1024, H, W]
    in_3 = SE input [B, 64, 1, 1]
    """
    conv_out = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    sigmoid_out = conv_out.sigmoid()
    mul_out = in_2 * sigmoid_out
    gelu_out = torch.nn.functional.gelu(mul_out, approximate='none')
    pool_out = torch.nn.functional.adaptive_avg_pool2d(gelu_out, 1)
    flat_out = pool_out.flatten(1, -1)
    dropout_out = torch.nn.functional.dropout(flat_out, 0.0, False, False)
    return dropout_out


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def se_gelu_pool_kernel_small(
    bias_ptr, weight_ptr, in_2_ptr, in_3_ptr, out_ptr,
    C_out, C_in, HW,
    stride_in2_c, stride_in2_hw,
    stride_in3_c,
    stride_w_cout, stride_w_cin,
    BLOCK_HW: tl.constexpr,
    C_IN_BLOCK: tl.constexpr,
):
    """
    Optimized kernel for small batch (B=1).
    Grid: (C_out,) - one program per output channel.
    """
    pid_c = tl.program_id(0)
    
    # ===== Step 1: Compute SE scale =====
    scale = tl.load(bias_ptr + pid_c)
    
    cin_range = tl.arange(0, C_IN_BLOCK)
    in3_ptrs = in_3_ptr + cin_range * stride_in3_c
    w_ptrs = weight_ptr + pid_c * stride_w_cout + cin_range * stride_w_cin
    
    in3_vals = tl.load(in3_ptrs)
    w_vals = tl.load(w_ptrs)
    
    scale += tl.sum(in3_vals * w_vals)
    scale = tl.sigmoid(scale)
    
    # ===== Step 2: Mul + GELU + Avg Pool =====
    hw_range = tl.arange(0, BLOCK_HW)
    hw_mask = hw_range < HW
    
    in2_ptrs = in_2_ptr + pid_c * stride_in2_c + hw_range * stride_in2_hw
    in2_vals = tl.load(in2_ptrs, mask=hw_mask, other=0.0)
    
    scaled_vals = in2_vals * scale
    gelu_vals = scaled_vals * 0.5 * (1.0 + tl.math.erf(scaled_vals * 0.7071067811865476))
    
    masked_gelu = tl.where(hw_mask, gelu_vals, 0.0)
    result = tl.sum(masked_gelu) / HW
    
    tl.store(out_ptr + pid_c, result)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BC': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_BC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BC': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BC': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_BC': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_BC': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_BC': 256}, num_warps=4, num_stages=3),
    ],
    key=['total_bc', 'HW'],
)
@triton.jit
def se_gelu_pool_kernel_large(
    bias_ptr, weight_ptr, in_2_ptr, in_3_ptr, out_ptr,
    B, C_out, C_in, HW, total_bc,
    stride_in2_b, stride_in2_c, stride_in2_hw,
    stride_in3_b, stride_in3_c,
    stride_w_cout, stride_w_cin,
    BLOCK_BC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    C_IN_BLOCK: tl.constexpr,
):
    """
    Fused kernel for SE attention + GELU + avg pool.
    Grid: 1D - each block handles BLOCK_BC (batch, channel) pairs.
    """
    pid = tl.program_id(0)
    
    bc_offsets = pid * BLOCK_BC + tl.arange(0, BLOCK_BC)
    bc_mask = bc_offsets < total_bc
    
    b_idx = bc_offsets // C_out
    c_idx = bc_offsets % C_out
    
    # ===== Step 1: Compute SE scale =====
    bias_vals = tl.load(bias_ptr + c_idx, mask=bc_mask, other=0.0)
    
    dot_acc = tl.zeros([BLOCK_BC], dtype=tl.float32)
    
    for cin in range(C_IN_BLOCK):
        in3_ptrs = in_3_ptr + b_idx * stride_in3_b + cin * stride_in3_c
        in3_vals = tl.load(in3_ptrs, mask=bc_mask, other=0.0)
        
        w_ptrs = weight_ptr + c_idx * stride_w_cout + cin * stride_w_cin
        w_vals = tl.load(w_ptrs, mask=bc_mask, other=0.0)
        
        dot_acc += in3_vals * w_vals
    
    scale = tl.sigmoid(bias_vals + dot_acc)
    
    # ===== Step 2: Mul + GELU + Avg Pool =====
    pool_acc = tl.zeros([BLOCK_BC], dtype=tl.float32)
    
    for hw in range(HW):
        in2_offset = b_idx * stride_in2_b + c_idx * stride_in2_c + hw * stride_in2_hw
        in2_val = tl.load(in_2_ptr + in2_offset, mask=bc_mask, other=0.0)
        
        scaled_val = in2_val * scale
        gelu_val = scaled_val * 0.5 * (1.0 + tl.math.erf(scaled_val * 0.7071067811865476))
        pool_acc += gelu_val
    
    result = pool_acc / HW
    tl.store(out_ptr + bc_offsets, result, mask=bc_mask)


@torch.fx.wrap
def se_gelu_pool_fused(in_0, in_1, in_2, in_3):
    """
    Fused implementation of SE attention + GELU + global avg pool.
    """
    B = in_2.shape[0]
    C_out = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    C_in = in_3.shape[1]
    HW = H * W
    total_bc = B * C_out
    
    # Reshape for contiguous access
    in_2_flat = in_2.view(B, C_out, HW)
    in_3_sq = in_3.view(B, C_in)
    weight_sq = in_1.view(C_out, C_in)
    
    # Output shape [B, C_out]
    out = torch.empty((B, C_out), device=in_2.device, dtype=in_2.dtype)
    
    BLOCK_HW = 64
    C_IN_BLOCK = 64
    
    # Use unified kernel with autotuning for all batch sizes
    out_flat = out.view(-1)
    grid = lambda meta: (triton.cdiv(total_bc, meta['BLOCK_BC']),)
    se_gelu_pool_kernel_large[grid](
        in_0, weight_sq, in_2_flat, in_3_sq, out_flat,
        B, C_out, C_in, HW, total_bc,
        in_2_flat.stride(0), in_2_flat.stride(1), in_2_flat.stride(2),
        in_3_sq.stride(0), in_3_sq.stride(1),
        weight_sq.stride(0), weight_sq.stride(1),
        BLOCK_HW=BLOCK_HW,
        C_IN_BLOCK=C_IN_BLOCK,
    )
    
    return out


def replacement_func():
    return se_gelu_pool_fused