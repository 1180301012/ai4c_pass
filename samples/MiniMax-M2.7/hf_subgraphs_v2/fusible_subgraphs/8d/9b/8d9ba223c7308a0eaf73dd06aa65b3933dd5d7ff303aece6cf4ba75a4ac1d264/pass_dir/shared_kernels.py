import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


# ========== Sigmoid Scale Kernel ==========

@triton.jit
def sigmoid_scale_kernel(
    x_ptr,
    scale_val,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused sigmoid + scale kernel.
    Computes: output = scale_val * sigmoid(input)
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    exp_neg_x = tl.exp(-x)
    sigmoid = 1.0 / (1.0 + exp_neg_x)
    
    # Scale
    result = scale_val * sigmoid
    
    # Store
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def sigmoid_scale_wrapper(x: torch.Tensor, scale_val: float):
    """
    Wrapper for fused sigmoid + scale operation.
    Handles bfloat16/float16 by converting to float32 and back.
    """
    # Unwrap tensor if it's wrapped
    x = unwrap_tensor(x)
    
    # Store original dtype
    orig_dtype = x.dtype
    
    # Convert to float32 for Triton computation
    x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x
    
    n_elements = x_fp32.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x_fp32)
    
    sigmoid_scale_kernel[(num_programs,)](
        x_ptr=x_fp32,
        scale_val=scale_val,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert back to original dtype
    return output.to(orig_dtype) if orig_dtype != torch.float32 else output


# ========== Softmax Kernel ==========

@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    n_elements: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softmax kernel along the last dimension.
    """
    # Calculate row id (each row is independently softmaxed)
    row_id = tl.program_id(0)
    row_start = x_ptr + row_id * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = col_offsets < n_cols
    
    # Load the row
    x = tl.load(row_start + col_offsets, mask=mask, other=0.0)
    
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    
    # Compute exp(x)
    exp_x = tl.exp(x)
    
    # Sum exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Compute softmax
    softmax = exp_x / sum_exp_x
    
    # Store result
    tl.store(output_ptr + row_id * n_cols + col_offsets, softmax, mask=mask)


@torch.fx.wrap
def softmax_wrapper(x: torch.Tensor, dim: int):
    """
    Custom softmax implementation using Triton.
    Currently only supports dim=-1 (last dimension).
    """
    # Unwrap tensor if it's wrapped
    x = unwrap_tensor(x)
    
    # Only optimize for last-dimension softmax
    if dim != -1:
        # Fallback to native implementation for non-last-dim
        # Note: This won't be called for our patterns since dim=-1 always
        return x.softmax(dim=dim)
    
    # Store original dtype
    orig_dtype = x.dtype
    
    # Convert to float32 for Triton computation
    x_fp32 = x.to(torch.float32) if x.dtype != torch.float32 else x
    
    # Flatten all dimensions except the last
    x_2d = x_fp32.view(-1, x_fp32.shape[-1])
    n_rows, n_cols = x_2d.shape
    
    # Use 1D grid where each program handles one row
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    
    output = torch.empty_like(x_2d)
    
    # Launch kernel with n_rows programs
    softmax_kernel[(n_rows,)](
        x_ptr=x_2d,
        output_ptr=output,
        n_elements=n_rows * n_cols,
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape and convert back to original dtype
    result = output.view(x.shape)
    return result.to(orig_dtype) if orig_dtype != torch.float32 else result


# ========== Bias Add Kernel ==========

@triton.jit
def bias_add_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for fused unsqueeze(1) unsqueeze(0) add.
    Broadcasts bias to match x shape and adds them.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Add
    result = x + bias
    
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def bias_add_1_wrapper(in_2: torch.Tensor, in_3: torch.Tensor):
    """
    Fused unsqueeze(1) unsqueeze(0) add wrapper.
    in_2: [1, 64, 12, 64, 64] or [1, 16, 24, 64, 64]
    in_3: [64, 64, 64] or [16, 64, 64]
    """
    # Unwrap tensors if they're wrapped
    in_2 = unwrap_tensor(in_2)
    in_3 = unwrap_tensor(in_3)
    
    # Store original dtype
    orig_dtype = in_2.dtype
    
    # Convert to float32 for computation
    in_2_fp32 = in_2.to(torch.float32) if in_2.dtype != torch.float32 else in_2
    in_3_fp32 = in_3.to(torch.float32) if in_3.dtype != torch.float32 else in_3
    
    # Unsqueeze operations: in_3.unsqueeze(1).unsqueeze(0)
    # in_3: [B, H, W] -> [B, 1, H, W] -> [1, B, 1, H, W]
    in_3_unsqueezed = in_3_fp32.unsqueeze(1).unsqueeze(0)
    
    # Now add: in_2 + in_3_unsqueezed (broadcasting should handle this)
    result = in_2_fp32 + in_3_unsqueezed
    
    # Convert back to original dtype
    return result.to(orig_dtype) if orig_dtype != torch.float32 else result


# ========== Dispatch Wrapper ==========

@torch.fx.wrap
def dispatch_wrapper(*args):
    """
    Dispatch wrapper that routes to the appropriate implementation.
    Route string is always the last argument.
    """
    if len(args) == 2:
        # sigmoid_scale: (x, "sigmoid")
        x, route = args
        if route == "sigmoid":
            return sigmoid_scale_wrapper(x, 16.0)
    elif len(args) == 3:
        # softmax: (x, dim, "softmax")
        x, dim, route = args
        if route == "softmax":
            return softmax_wrapper(x, dim)
        # bias_add: (in_2, in_3, "bias_add_1")
        in_2, in_3, route = args
        if route == "bias_add_1":
            return bias_add_1_wrapper(in_2, in_3)
    
    # Fallback - should not reach here
    raise ValueError(f"Unknown route: {args}")