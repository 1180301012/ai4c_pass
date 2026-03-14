import torch
import triton
import triton.language as tl


# Pattern matching function - must mirror model.py exactly
def pattern(bias, weight, x_input, x_se):
    """
    Match the computation pattern:
    conv2d -> hardsigmoid -> mul -> adaptive_avg_pool2d -> flatten -> dropout
    """
    conv_out = torch.conv2d(x_se, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    hsigmoid = torch.nn.functional.hardsigmoid(conv_out, False)
    mul_out = x_input * hsigmoid
    pool_out = torch.nn.functional.adaptive_avg_pool2d(mul_out, 1)
    flat_out = pool_out.flatten(1, -1)
    dropout_out = torch.nn.functional.dropout(flat_out, 0.0, False, False)
    return dropout_out


# Argument extraction function
def replacement_args(bias, weight, x_input, x_se):
    return (bias, weight, x_input, x_se)


@triton.jit
def fully_fused_kernel(
    x_se_ptr,       # [B, Cin]
    weight_ptr,     # [Cout, Cin]
    bias_ptr,       # [Cout]
    x_input_ptr,    # [B, Cout, HW]
    out_ptr,        # [B, Cout]
    B,
    Cout,
    Cin,
    HW,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """
    Fully fused kernel: conv1x1 + hardsigmoid + avg_pool + multiply
    One program per (batch, channel) pair
    
    output[b, c] = hardsigmoid(dot(x_se[b,:], weight[c,:]) + bias[c]) * mean(x_input[b,c,:])
    """
    pid = tl.program_id(0)
    batch_idx = pid // Cout
    cout_idx = pid % Cout
    
    # 1. Compute conv1x1 (dot product)
    x_se_base = batch_idx * Cin
    weight_base = cout_idx * Cin
    
    conv_acc = 0.0
    for k_start in range(0, Cin, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < Cin
        x_se_vals = tl.load(x_se_ptr + x_se_base + k_offsets, mask=k_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + weight_base + k_offsets, mask=k_mask, other=0.0)
        conv_acc += tl.sum(x_se_vals * weight_vals, axis=0)
    
    bias_val = tl.load(bias_ptr + cout_idx)
    conv_out = conv_acc + bias_val
    
    # 2. Apply hardsigmoid
    hsig = (conv_out + 3.0) / 6.0
    hsig = tl.maximum(0.0, tl.minimum(1.0, hsig))
    
    # 3. Compute avg pool of x_input
    x_input_base = batch_idx * Cout * HW + cout_idx * HW
    pool_acc = 0.0
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        x_vals = tl.load(x_input_ptr + x_input_base + hw_offsets, mask=hw_mask, other=0.0)
        pool_acc += tl.sum(x_vals, axis=0)
    
    mean_val = pool_acc / HW
    
    # 4. Final multiply
    result = hsig * mean_val
    
    out_offset = batch_idx * Cout + cout_idx
    tl.store(out_ptr + out_offset, result)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['B', 'Cout', 'Cin'],
)
@triton.jit  
def matmul_hardsigmoid_mul_kernel(
    x_se_ptr,       # [B, Cin]
    weight_ptr,     # [Cout, Cin] 
    bias_ptr,       # [Cout]
    x_mean_ptr,     # [B, Cout] - pre-computed mean
    out_ptr,        # [B, Cout]
    B,
    Cout,
    Cin,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled matmul: out = hardsigmoid(x_se @ weight.T + bias) * x_mean
    Using standard tiled matrix multiplication approach.
    BLOCK_M, BLOCK_N, BLOCK_K must all be >= 16 for tl.dot
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(B, BLOCK_M)
    num_pid_n = tl.cdiv(Cout, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Tiled matmul loop
    for k_start in range(0, Cin, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load x_se tile [BLOCK_M, BLOCK_K]
        x_se_ptrs = x_se_ptr + offs_m[:, None] * Cin + k_offs[None, :]
        x_se_mask = (offs_m[:, None] < B) & (k_offs[None, :] < Cin)
        x_se_tile = tl.load(x_se_ptrs, mask=x_se_mask, other=0.0)
        
        # Load weight tile [BLOCK_K, BLOCK_N] for matmul
        weight_ptrs = weight_ptr + offs_n[None, :] * Cin + k_offs[:, None]
        weight_mask = (offs_n[None, :] < Cout) & (k_offs[:, None] < Cin)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(x_se_tile, weight_tile)
    
    # Add bias [BLOCK_N]
    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < Cout, other=0.0)
    acc = acc + bias_vals[None, :]
    
    # Apply hardsigmoid: clamp((x + 3) / 6, 0, 1)
    hsig = (acc + 3.0) / 6.0
    hsig = tl.maximum(0.0, tl.minimum(1.0, hsig))
    
    # Load x_mean and multiply
    x_mean_ptrs = x_mean_ptr + offs_m[:, None] * Cout + offs_n[None, :]
    x_mean_mask = (offs_m[:, None] < B) & (offs_n[None, :] < Cout)
    x_mean_vals = tl.load(x_mean_ptrs, mask=x_mean_mask, other=0.0)
    
    result = hsig * x_mean_vals
    
    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * Cout + offs_n[None, :]
    out_mask = (offs_m[:, None] < B) & (offs_n[None, :] < Cout)
    tl.store(out_ptrs, result, mask=out_mask)


@triton.jit
def avg_pool_kernel(
    x_ptr,      # Input [B, C, HW]
    out_ptr,    # Output [B, C]
    B,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """Compute average pooling over spatial dimension."""
    pid = tl.program_id(0)
    batch_idx = pid // C
    c_idx = pid % C
    
    x_base = batch_idx * C * HW + c_idx * HW
    
    acc = 0.0
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        x_vals = tl.load(x_ptr + x_base + hw_offsets, mask=hw_mask, other=0.0)
        acc += tl.sum(x_vals, axis=0)
    
    mean_val = acc / HW
    out_offset = batch_idx * C + c_idx
    tl.store(out_ptr + out_offset, mean_val)


@torch.fx.wrap
def fused_conv1x1_hardsigmoid_mul_pool_wrapper(bias, weight, x_input, x_se):
    """
    Optimized implementation with adaptive strategy based on batch size.
    """
    B = x_input.shape[0]
    Cout = x_input.shape[1]
    H = x_input.shape[2]
    W = x_input.shape[3]
    HW = H * W
    Cin = x_se.shape[1]
    
    x_se_flat = x_se.view(B, Cin).contiguous()
    weight_flat = weight.view(Cout, Cin).contiguous()
    x_input_flat = x_input.view(B, Cout, HW).contiguous()
    bias_contig = bias.contiguous()
    
    out = torch.empty((B, Cout), device=x_input.device, dtype=x_input.dtype)
    
    # For small batches, use fully fused kernel
    # For large batches, use tiled matmul with separate avg pool
    if B < 16:
        # Fully fused kernel - single pass for small batches
        num_programs = B * Cout
        BLOCK_K = 256
        BLOCK_HW = 64
        
        fully_fused_kernel[(num_programs,)](
            x_se_flat,
            weight_flat,
            bias_contig,
            x_input_flat,
            out,
            B=B,
            Cout=Cout,
            Cin=Cin,
            HW=HW,
            BLOCK_K=BLOCK_K,
            BLOCK_HW=BLOCK_HW,
        )
    else:
        # Two-pass: avg_pool then tiled matmul for large batches
        x_mean = torch.empty((B, Cout), device=x_input.device, dtype=x_input.dtype)
        
        # Avg pool kernel
        BLOCK_HW = 64
        num_programs_pool = B * Cout
        avg_pool_kernel[(num_programs_pool,)](
            x_input_flat,
            x_mean,
            B=B,
            C=Cout,
            HW=HW,
            BLOCK_HW=BLOCK_HW,
        )
        
        # Tiled matmul with autotuning
        def grid(META):
            return (
                triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(Cout, META['BLOCK_N']),
            )
        
        matmul_hardsigmoid_mul_kernel[grid](
            x_se_flat,
            weight_flat,
            bias_contig,
            x_mean,
            out,
            B=B,
            Cout=Cout,
            Cin=Cin,
        )
    
    return out


def replacement_func():
    return fused_conv1x1_hardsigmoid_mul_pool_wrapper