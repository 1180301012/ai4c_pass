"""
Fused optimization pass for RMSNorm computation.
Handles both eps=1e-06 (SmolLM) and eps=1e-05 (TinyLlama) variants.
"""
import torch
import triton
import triton.language as tl


# Autotune configurations for RMSNorm kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['n_last_dim'],
)
@triton.jit
def fused_rmsnorm_kernel(
    x_ptr,
    eps,
    output_ptr,
    n_elements: tl.constexpr,
    n_last_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel with autotuning.
    Computes: output = x * rsqrt(mean(x^2, dim=-1) + eps)
    """
    # Calculate program IDs for 1D grid
    pid = tl.program_id(0)
    
    # Calculate row offset for this program
    row_offset = pid * BLOCK_SIZE
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements * n_last_dim
    
    # Load x values for this row
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # x^2
    x_sq = x * x
    
    # Reshape for reduction over last dim
    # Each program processes BLOCK_SIZE elements, we need to reduce across n_last_dim
    n_elements_per_row = n_last_dim
    row_block_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute sum of squares for this row
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute normalization factor
    norm_factor = tl.rsqrt(sum_sq / n_elements_per_row + eps)
    
    # Multiply by norm factor and store
    output = x * norm_factor
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_rmsnorm_kernel_wrapper(x: torch.Tensor, eps: float, dtype_out: torch.dtype) -> torch.Tensor:
    """
    Wrapper for fused RMSNorm kernel.
    Computes: output = x * rsqrt(mean(x^2, dim=-1) + eps)
    """
    x_shape = x.shape
    n_last_dim = x_shape[-1]
    n_rows = x.numel() // n_last_dim
    
    # Allocate output
    output = torch.empty_like(x, dtype=dtype_out)
    
    # Configure kernel - use 1D grid with rows
    grid = (n_rows,)
    
    fused_rmsnorm_kernel[grid](
        x_ptr=x,
        eps=eps,
        output_ptr=output,
        n_elements=n_rows,
        n_last_dim=n_last_dim,
    )
    
    return output


def pattern_1e_06(in_0, in_1, in_2):
    """
    Match the computation pattern with eps=1e-06 (SmolLM models).
    """
    # Path A: sin/cos with concat
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    
    # Path B: RMSNorm-style computation with eps=1e-06
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    
    return (tmp_6, tmp_17, tmp_7)


def pattern_1e_05(in_0, in_1, in_2):
    """
    Match the computation pattern with eps=1e-05 (TinyLlama models).
    """
    # Path A: sin/cos with concat
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.float32)
    tmp_7 = tmp_5.to(dtype=torch.float32)
    
    # Path B: RMSNorm-style computation with eps=1e-05
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-05
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.float32)
    tmp_17 = in_0 * tmp_16
    
    return (tmp_6, tmp_17, tmp_7)


def replacement_args_1e_06(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1e-06, "smollm")


def replacement_args_1e_05(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1e-05, "tinyllama")


def replacement_func():
    """
    Returns a dispatch function that handles both eps variants.
    """
    def dispatch(in_0, in_1, in_2, eps, route=""):
        # Determine output dtype based on route
        if route == "tinyllama":
            output_dtype = torch.float32
        else:
            output_dtype = torch.bfloat16
        
        # Path A: sin/cos computation
        tmp_1 = torch.cat((in_1, in_1), dim=-1)
        tmp_1_f32 = tmp_1.to(torch.float32)
        tmp_2 = tmp_1_f32.cos()
        tmp_4 = tmp_1_f32.sin()
        tmp_6 = tmp_2.to(dtype=output_dtype)
        tmp_7 = tmp_4.to(dtype=output_dtype)
        
        # Path B: RMSNorm computation - use fused kernel
        tmp_16 = fused_rmsnorm_kernel_wrapper(in_2, eps, output_dtype)
        tmp_17 = in_0 * tmp_16
        
        return (tmp_6, tmp_17, tmp_7)
    
    return dispatch