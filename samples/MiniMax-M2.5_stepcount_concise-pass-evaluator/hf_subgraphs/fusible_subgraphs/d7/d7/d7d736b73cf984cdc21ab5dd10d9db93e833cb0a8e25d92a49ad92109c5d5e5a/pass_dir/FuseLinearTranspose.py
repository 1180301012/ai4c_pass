import torch
import triton
import triton.language as tl


def pattern(bias, weight, input):
    """
    Pattern: linear(input, weight, bias) followed by transpose(-1, -2)
    This matches: torch.nn.functional.linear(in_2, tmp_1, tmp_0) then .transpose(-1, -2)
    """
    # Linear: input [B, D, M] @ weight [M, M].T + bias [M] -> [B, D, M]
    linear_out = torch.nn.functional.linear(input, weight, bias)
    # Transpose: [B, D, M] -> [B, M, D]
    transposed_out = linear_out.transpose(-1, -2)
    return transposed_out


def replacement_args(bias, weight, input):
    """
    Extract arguments needed for the replacement kernel.
    """
    return (bias, weight, input)


# Use a fixed block size - must be constexpr for Triton
# Try smaller block size for better occupancy
BLOCK_SIZE = tl.constexpr(32)


@triton.jit
def fused_linear_transpose_kernel_large_batch(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, D, M, N,  # Input shape: [B, D, M], Weight: [M, N], Output: [B, N, D]
    stride_input_b, stride_input_d, stride_input_m,
    stride_weight_m, stride_weight_n,
    stride_output_b, stride_output_n, stride_output_d,
):
    """
    Kernel for large batch sizes.
    Each program handles one batch element and a tile of N dimension.
    
    Grid: (B, cdiv(N, BLOCK_SIZE))
    """
    # Get batch index
    batch_idx = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate output offsets
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, BLOCK_SIZE)  # Full D dimension for this batch
    
    # Offsets for M dimension
    offs_m = tl.arange(0, BLOCK_SIZE)
    
    # Load bias
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over M dimension
    for m in range(0, M, BLOCK_SIZE):
        offs_m = m + tl.arange(0, BLOCK_SIZE)
        
        # Load weight[n, m]
        weight_offsets = (
            offs_n[:, None] * stride_weight_n + 
            offs_m[None, :] * stride_weight_m
        )
        weight = tl.load(
            weight_ptr + weight_offsets,
            mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
            other=0.0
        )
        
        # Load input[batch, d, m]
        input_base = batch_idx * stride_input_b
        input_offsets = (
            input_base + 
            offs_d[None, :] * stride_input_d + 
            offs_m[:, None] * stride_input_m
        )
        input_vals = tl.load(
            input_ptr + input_offsets,
            mask=(offs_d[None, :] < D) & (offs_m[:, None] < M),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(weight, input_vals)
    
    # Add bias
    acc += bias[:, None]
    
    # Store result
    output_offsets = (
        batch_idx * stride_output_b +
        offs_n[:, None] * stride_output_n +
        offs_d[None, :] * stride_output_d
    )
    tl.store(
        output_ptr + output_offsets,
        acc,
        mask=(offs_n[:, None] < N) & (offs_d[None, :] < D)
    )


@triton.jit
def fused_linear_transpose_kernel_small_batch(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, D, M, N,  # Input shape: [B, D, M], Weight: [M, N], Output: [B, N, D]
    stride_input_b, stride_input_d, stride_input_m,
    stride_weight_m, stride_weight_n,
    stride_output_b, stride_output_n, stride_output_d,
):
    """
    Kernel for small batch sizes.
    Uses 2D grid (n_blocks, d_blocks) to maximize parallelism.
    Each program handles all batch elements for its tile.
    
    Grid: (cdiv(N, BLOCK_SIZE), cdiv(D, BLOCK_SIZE))
    """
    # Get program indices
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Calculate output offsets for this program
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Offsets for M dimension
    offs_m = tl.arange(0, BLOCK_SIZE)
    
    # Load bias
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    
    # Process each batch element
    for b in range(B):
        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
        
        # Iterate over M dimension
        for m in range(0, M, BLOCK_SIZE):
            offs_m = m + tl.arange(0, BLOCK_SIZE)
            
            # Load weight[n, m]
            weight_offsets = (
                offs_n[:, None] * stride_weight_n + 
                offs_m[None, :] * stride_weight_m
            )
            weight = tl.load(
                weight_ptr + weight_offsets,
                mask=(offs_n[:, None] < N) & (offs_m[None, :] < M),
                other=0.0
            )
            
            # Load input[b, d, m]
            input_base = b * stride_input_b
            input_offsets = (
                input_base + 
                offs_d[None, :] * stride_input_d + 
                offs_m[:, None] * stride_input_m
            )
            input_vals = tl.load(
                input_ptr + input_offsets,
                mask=(offs_d[None, :] < D) & (offs_m[:, None] < M),
                other=0.0
            )
            
            # Accumulate
            acc += tl.dot(weight, input_vals)
        
        # Add bias
        acc += bias[:, None]
        
        # Store result
        output_offsets = (
            b * stride_output_b +
            offs_n[:, None] * stride_output_n +
            offs_d[None, :] * stride_output_d
        )
        tl.store(
            output_ptr + output_offsets,
            acc,
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D)
        )


@torch.fx.wrap
def fused_linear_transpose_wrapper(bias, weight, input):
    """
    Wrapper function that launches the fused kernel.
    
    Input shape: [B, D, M]
    Weight shape: [M, M]
    Bias shape: [M]
    Output shape: [B, M, D]
    """
    B, D, M = input.shape
    N = weight.shape[0]  # M
    
    # Allocate output
    output = torch.empty((B, N, D), dtype=input.dtype, device=input.device)
    
    # Use the large batch kernel for all cases - it's more efficient
    # Grid: (B, cdiv(N, BLOCK_SIZE))
    grid = (B, triton.cdiv(N, BLOCK_SIZE))
    fused_linear_transpose_kernel_large_batch[grid](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        B=B, D=D, M=M, N=N,
        stride_input_b=input.stride(0),
        stride_input_d=input.stride(1),
        stride_input_m=input.stride(2),
        stride_weight_m=weight.stride(0),
        stride_weight_n=weight.stride(1),
        stride_output_b=output.stride(0),
        stride_output_n=output.stride(1),
        stride_output_d=output.stride(2),
    )
    
    return output


def replacement_func():
    """
    Return the replacement function.
    """
    return fused_linear_transpose_wrapper