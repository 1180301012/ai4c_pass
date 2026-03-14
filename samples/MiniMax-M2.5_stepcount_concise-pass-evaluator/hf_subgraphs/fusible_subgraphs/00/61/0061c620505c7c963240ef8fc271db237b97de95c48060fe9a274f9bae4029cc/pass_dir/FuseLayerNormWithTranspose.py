import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match LayerNorm followed by transpose(-1, -2)
    in_0: bias (shape [768])
    in_1: weight (shape [768])
    in_2: input tensor (shape [B, N, C] = [B, 196, 768])
    """
    # LayerNorm with normalized_shape=(768,), weight=in_1, bias=in_0, eps=1e-05
    tmp_2 = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05)
    # Transpose last two dimensions
    tmp_3 = tmp_2.transpose(-1, -2)
    return tmp_3


def replacement_args(in_0, in_1, in_2):
    """Extract arguments needed for the replacement kernel"""
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        # Try different block sizes for the normalization dimension
        triton.Config({'BLOCK_SIZE_C': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 768}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 1024}, num_stages=3, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def fused_layernorm_transpose_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, C,  # input dimensions
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused LayerNorm + Transpose kernel
    
    Input: [B, N, C]
    Output: [B, C, N] (transposed)
    
    Each program handles one element in the output [B, C, N]
    """
    # Calculate position in output tensor [B, C, N]
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Total programs should be [B, C, NBLOCKS]
    # Where NBLOCKS = ceil(N / BLOCK_SIZE_C)
    
    # Offset for input at batch pid_b
    input_offset = pid_b * N * C
    output_offset = pid_b * C * N + pid_c * N + pid_n
    
    # Load weight and bias for channel pid_c
    weight_val = tl.load(weight_ptr + pid_c)
    bias_val = tl.load(bias_ptr + pid_c)
    
    # Compute mean and variance for the feature dimension at position [pid_b, pid_n, :]
    # LayerNorm normalizes over C dimension
    input_base = input_offset + pid_n * C
    
    # Compute sum for mean
    sum_val = 0.0
    sum_sq = 0.0
    for i in range(0, C, BLOCK_SIZE_C):
        offs = i + tl.arange(0, BLOCK_SIZE_C)
        mask = offs < C
        
        # Load C elements for this row
        vals = tl.load(input_ptr + input_base + offs, mask=mask, other=0.0)
        
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)
    
    # Calculate mean and variance
    mean = sum_val / C
    var = (sum_sq / C) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Compute normalized and transposed output
    # We need to load all C elements again to compute the normalized output
    # Then store at position [pid_b, pid_c, pid_n] (transposed)
    result = 0.0
    for i in range(0, C, BLOCK_SIZE_C):
        offs = i + tl.arange(0, BLOCK_SIZE_C)
        mask = offs < C
        
        vals = tl.load(input_ptr + input_base + offs, mask=mask, other=0.0)
        
        # LayerNorm: (x - mean) * weight * inv_std + bias
        normalized = (vals - mean) * inv_std
        transformed = normalized * weight_val + bias_val
        
        # We only need the value at position pid_c
        # Since we're processing all C values, we need to handle this differently
        
    # Actually, let's rethink this - we want to compute output[pid_b, pid_c, pid_n]
    # which corresponds to input[pid_b, pid_n, pid_c] after LayerNorm
    # That means we need to normalize input[pid_b, pid_n, :] first, then select index pid_c
    
    # This approach is inefficient. Let me reconsider.
    
    # Better approach: each program handles one (b, n) pair, computes the full LayerNorm,
    # then we transpose. But that's also not great for GPU parallelism.
    
    # Actually, let's use a simpler approach: 
    # Each program computes one output element output[b, c, n]
    # This corresponds to LayerNorm(input[b, n, :])[c]
    # We need mean and variance for row (b, n), then compute element c
    
    pass  # Placeholder - will rewrite below


# Let's use a more straightforward approach - compute LayerNorm first, then transpose
# Actually, a better fusion strategy is to compute LayerNorm in a more efficient way
# and avoid the explicit transpose

# Let me think about this more carefully...
# The key insight is that LayerNorm normalizes over C dimension for each (b, n)
# Then transpose just swaps (n, c) -> (c, n)

# For efficiency, let's process each (b, n) row independently, computing all C outputs
# Then we have the full [b, n, c] result, and we can write to transposed positions

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 8}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_layernorm_transpose_kernel_v2(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, C, eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused LayerNorm + Transpose kernel
    
    Input: [B, N, C] - B=batch, N=196, C=768
    Output: [B, C, N] - transposed
    
    Each program handles one batch element b
    Each program processes BLOCK_SIZE_N rows (n positions)
    """
    pid_b = tl.program_id(0)
    row_offsets = pid_b * N * C + tl.arange(0, BLOCK_SIZE_N)[:, None] * C + tl.arange(0, C)[None, :]
    
    # Actually, let's do it simpler: each program handles one (b, n) pair
    # and writes C elements
    
    # Get program ids
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_b >= B or pid_n >= N:
        return
    
    # Calculate input offset for row [pid_b, pid_n, :]
    input_base = pid_b * N * C + pid_n * C
    
    # Compute mean and variance
    # Load all C elements
    x = tl.load(input_ptr + input_base + tl.arange(0, C), mask=tl.arange(0, C) < C)
    
    mean = tl.sum(x, axis=0) / C
    var = tl.sum(x * x, axis=0) / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Apply LayerNorm: (x - mean) * inv_std * weight + bias
    normalized = (x - mean) * inv_std
    weight = tl.load(weight_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    bias = tl.load(bias_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    
    out = normalized * weight + bias
    
    # Write to transposed output [B, C, N]
    # output[b, c, n] -> flat offset = b * C * N + c * N + n
    output_base = pid_b * C * N + pid_n  # Start at column pid_n
    # We need to stride by N to go to next channel
    for i in range(0, C):
        tl.store(output_ptr + pid_b * C * N + i * N + pid_n, out[i])


# LayerNorm kernel - with BLOCK_SIZE as constexpr parameter
@triton.jit
def layernorm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, N, C, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm kernel - computes normalization over C dimension for each (b, n)
    
    Grid: (B * N,) - each program handles one (b, n) pair
    """
    # Grid: (B * N,)
    pid = tl.program_id(0)
    pid_b = pid // N
    pid_n = pid % N
    
    if pid_b >= B or pid_n >= N:
        return
    
    # Create range for BLOCK_SIZE elements
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    
    # Load input row [b, n, :]
    input_base = pid_b * N * C + pid_n * C
    x = tl.load(input_ptr + input_base + offs, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / C
    var = tl.sum(x * x, axis=0) / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    normalized = (x - mean) * inv_std
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    
    # Apply affine transformation
    out = normalized * weight + bias
    
    # Store to output - same layout as input [B, N, C]
    # Use vectorized store
    tl.store(output_ptr + input_base + offs, out, mask=mask)


@torch.fx.wrap
def fused_layernorm_transpose(bias: torch.Tensor, weight: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """
    Fused LayerNorm + Transpose kernel
    
    Args:
        bias: LayerNorm bias (shape [C])
        weight: LayerNorm weight (shape [C])
        input: Input tensor (shape [B, N, C])
    
    Output shape: [B, C, N]
    """
    B, N, C = input.shape
    
    # First compute LayerNorm using Triton
    ln_output = torch.empty_like(input)
    
    # Grid: one program per (b, n) pair
    grid = (B * N, )
    
    # BLOCK_SIZE must be a power of 2 and >= max C (768)
    BLOCK_SIZE = 1024
    
    layernorm_kernel[grid](
        input, weight, bias, ln_output,
        B, N, C, 
        1e-5,  # eps
        BLOCK_SIZE,
    )
    
    # Now transpose the last two dimensions
    return ln_output.transpose(-1, -2)


def replacement_func():
    """Return the fused kernel function"""
    return fused_layernorm_transpose