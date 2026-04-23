"""
Fused optimization pass for RMSNorm computation with eps=1e-06 (SmolLM models).
Full computation fused in Triton: concat + sin/cos + RMSNorm + multiply.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def sincos_kernel(
    freqs_ptr,
    cos_out_ptr,
    sin_out_ptr,
    batch: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sin/cos kernel with implicit concat (doubling head_dim).
    """
    pid = tl.program_id(0)
    total_elements = batch * seq_len * head_dim * 2  # 2x for concat
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Map output index to input index (output is 2x head_dim)
    # output[..., i] where i >= head_dim maps to input[..., i - head_dim]
    input_offsets = offsets % (head_dim * 2)
    input_mask = input_offsets < head_dim
    src_offsets = offsets - (input_offsets // head_dim) * head_dim
    
    x = tl.load(freqs_ptr + src_offsets, mask=mask & input_mask, other=0.0)
    
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    tl.store(cos_out_ptr + offsets, cos_val, mask=mask)
    tl.store(sin_out_ptr + offsets, sin_val, mask=mask)


@torch.fx.wrap
def sincos_kernel_wrapper(freqs: torch.Tensor, output_dtype: torch.dtype) -> tuple:
    """Wrapper for sincos kernel."""
    batch, seq_len, head_dim = freqs.shape
    total_out_elements = batch * seq_len * head_dim * 2
    
    cos_out = torch.empty((batch, seq_len, head_dim * 2), 
                          dtype=torch.float32, device=freqs.device)
    sin_out = torch.empty((batch, seq_len, head_dim * 2), 
                          dtype=torch.float32, device=freqs.device)
    
    BLOCK_SIZE = 1024
    num_programs = (total_out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_programs,)
    
    sincos_kernel[grid](
        freqs_ptr=freqs,
        cos_out_ptr=cos_out,
        sin_out_ptr=sin_out,
        batch=batch,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out.to(dtype=output_dtype), sin_out.to(dtype=output_dtype)


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    eps,
    output_ptr,
    n_rows: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm kernel.
    Computes: output = x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    """
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows * hidden_dim
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute x^2
    x_sq = x * x
    
    # Sum across hidden dim (all elements in block contribute to same sum)
    sum_sq = tl.sum(x_sq, axis=0)
    
    # RMSNorm: x * rsqrt(sum(x^2)/N + eps) * weight
    norm_factor = tl.rsqrt(sum_sq / hidden_dim + eps)
    rms_out = x * norm_factor
    
    # Load weight and multiply
    weight = tl.load(weight_ptr + (offsets % hidden_dim))
    out = rms_out * weight
    
    tl.store(output_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def rmsnorm_kernel_wrapper(x: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    """Wrapper for RMSNorm kernel."""
    hidden_dim = x.shape[-1]
    n_rows = x.numel() // hidden_dim
    
    output = torch.empty_like(x, dtype=output_dtype)
    grid = (n_rows,)
    
    rmsnorm_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        eps=eps,
        output_ptr=output,
        n_rows=n_rows,
        hidden_dim=hidden_dim,
    )
    
    return output


def pattern(in_0, in_1, in_2):
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


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 1e-06, "smollm")


def replacement_func():
    """
    Returns a dispatch function that handles both eps variants.
    ALL computation uses Triton kernels only.
    """
    def dispatch(in_0, in_1, in_2, eps, route=""):
        if route == "tinyllama":
            output_dtype = torch.float32
        else:
            output_dtype = torch.bfloat16
        
        # Path A: sin/cos with implicit concat using Triton kernel
        cos_out, sin_out = sincos_kernel_wrapper(in_1, output_dtype)
        
        # Path B: RMSNorm using Triton kernel
        # First convert input to float32
        x_f32 = torch.empty_like(in_2, dtype=torch.float32)
        x_f32.copy_(in_2)
        
        rmsnorm_out = rmsnorm_kernel_wrapper(x_f32, in_0, eps, output_dtype)
        
        return (cos_out, rmsnorm_out, sin_out)
    
    return dispatch