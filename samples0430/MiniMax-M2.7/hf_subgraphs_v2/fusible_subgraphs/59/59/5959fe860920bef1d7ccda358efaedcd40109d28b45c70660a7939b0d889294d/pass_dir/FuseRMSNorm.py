import torch
import triton
import triton.language as tl

# Auto-tuning configuration for the fused kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['hidden_dim'],
)
@triton.jit
def fused_rmsnorm_kernel(
    x_ptr,           # Input tensor (inputs_embeds)
    weight_ptr,      # Weight tensor (in_0)
    out_ptr,         # Output tensor
    hidden_dim,      # Hidden dimension size
    n_rows,          # Total number of rows (batch * seq_len)
    n_elements,      # Total number of elements
    eps,             # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)
    
    # Compute row offsets
    row_offset = row_idx * hidden_dim
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data - convert to float32
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Compute sum of squares for this row
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute RMS: rsqrt(sum(x^2)/n + eps)
    rms = tl.rsqrt(sum_sq / hidden_dim + eps)
    
    # Compute output: x * rms
    out = x * rms
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def multiply_and_convert_kernel(in_ptr, weight_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + (offsets % 2048), mask=mask, other=0.0).to(tl.float32)
    out = (x * w).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


def rmsnorm_wrapper(in_0, in_1, in_2, eps, output_dtype):
    """Module-level wrapper function for RMSNorm pass"""
    # Fused RMSNorm computation using Triton kernels
    batch, seq_len, hidden_dim = in_2.shape
    
    # Flatten to 2D for easier processing
    x_flat = in_2.reshape(-1, hidden_dim)
    n_rows = x_flat.shape[0]
    n_elements = n_rows * hidden_dim
    
    # Create output tensor
    out = torch.empty([n_rows, hidden_dim], dtype=torch.float32, device=in_2.device)
    
    # Launch kernel - one block per row
    grid = (n_rows,)
    fused_rmsnorm_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=in_0,
        out_ptr=out,
        hidden_dim=hidden_dim,
        n_rows=n_rows,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=hidden_dim,
    )
    
    # Reshape back to original shape
    out = out.reshape(batch, seq_len, hidden_dim)
    
    # Create result tensor
    result = torch.empty([batch, seq_len, hidden_dim], dtype=output_dtype, device=in_2.device)
    
    n = batch * seq_len * hidden_dim
    BLOCK_SIZE = 1024
    num_programs = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    multiply_and_convert_kernel[(num_programs,)](
        in_ptr=out,
        weight_ptr=in_0,
        out_ptr=result,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result


def pattern(in_0, in_1, in_2):
    """
    Match the RMSNorm pattern:
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + epsilon
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(target_dtype)
    tmp_17 = in_0 * tmp_16
    """
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    
    return tmp_17


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1e-06, torch.bfloat16)


def replacement_func():
    return rmsnorm_wrapper