import torch
import triton
import triton.language as tl


@triton.jit
def fused_rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    hidden_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMS Normalization kernel.
    Input shape: [batch, seq_len, hidden_dim]
    Weight shape: [hidden_dim]
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    # Compute squared sum for this sequence element
    row_offset = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim
    squared_sum = 0.0

    # Process in blocks
    for i in range(0, hidden_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        # Load input values
        x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
        # Accumulate squared values
        squared_sum += tl.sum(x * x, axis=0)

    # Compute normalization factor: 1/sqrt(mean(x^2) + eps)
    # For RMS norm: normalize by sqrt(sum(x^2)/hidden_dim + eps)
    norm_factor = tl.rsqrt(squared_sum / hidden_dim + eps)

    # Second pass: normalize and apply weight
    for i in range(0, hidden_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        # Load original input
        x = tl.load(input_ptr + row_offset + offsets, mask=mask, other=0.0)
        # Load weight
        w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

        # Normalize and scale
        normalized = x * norm_factor
        out = normalized * w

        # Store result
        tl.store(output_ptr + row_offset + offsets, out, mask=mask)


@torch.fx.wrap
def fused_rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    Fused RMS Normalization.
    Input: [batch, seq_len, hidden_dim]
    Weight: [hidden_dim]
    Output: [batch, seq_len, hidden_dim] in bfloat16
    """
    # Ensure input is float32 for computation
    input_fp32 = input_tensor.to(torch.float32)

    # Get dimensions
    batch = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    hidden_dim = input_tensor.size(2)

    # Output in bfloat16
    output = torch.empty_like(input_tensor, dtype=torch.bfloat16)

    # Choose block size based on hidden dim
    BLOCK_SIZE = min(1024, hidden_dim)
    # Grid: (batch, seq_len)
    grid = (batch, seq_len)

    # Launch kernel
    fused_rms_norm_kernel[grid](
        input_ptr=input_fp32,
        weight_ptr=weight,
        output_ptr=output,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


# Pattern matching function - this matches the RMS normalization computation
def pattern(in_0, in_2):
    """
    Pattern: RMS normalization + linear transformation
    This matches:
      tmp_10 = in_2.to(torch.float32)
      tmp_11 = tmp_10.pow(2)
      tmp_12 = tmp_11.mean(-1, keepdim=True)
      tmp_13 = tmp_12 + 1e-06
      tmp_14 = torch.rsqrt(tmp_13)
      tmp_15 = tmp_10 * tmp_14
      tmp_16 = tmp_15.to(torch.bfloat16)
      tmp_17 = tmp_0 * tmp_16  (where tmp_0 = in_0)
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


# Argument extraction function
def replacement_args(in_0, in_2):
    return (in_2, in_0)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_rms_norm