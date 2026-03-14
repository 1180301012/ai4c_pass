import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Different tile sizes for different workload sizes
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=1),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=8),
    ],
    key=['B', 'C', 'H_out', 'W_out'],
)
@triton.jit
def interpolate_mul_kernel(
    input_ptr, other_ptr, output_ptr,
    B, C, H_in, W_in, H_out, W_out,
    scale_h, scale_w,
    stride_input_b, stride_input_c, stride_input_h, stride_input_w,
    stride_other_b, stride_other_c, stride_other_h, stride_other_w,
    stride_output_b, stride_output_c, stride_output_h, stride_output_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused interpolate (nearest neighbor) + multiply kernel.
    
    For nearest neighbor interpolation:
    output[b, c, h, w] = input[b, c, h * scale_h, w * scale_w] * other[b, c, h, w]
    """
    # Get program id
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(H_out, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(W_out, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, H_out - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_b = tl.program_id(1)
    offs_c = tl.arange(0, C)
    
    # Create masks
    mask_m = offs_m < H_out
    mask_n = offs_n < W_out
    
    # Compute input strides for broadcasting
    # Input: [B, C, H_in, W_in], we need to compute nearest neighbor indices
    # For nearest neighbor: input_h = output_h * scale_h, input_w = output_w * scale_w
    
    # Load other tensor: [B, C, H_out, W_out]
    # Reshape for broadcasting
    other_ptrs = (
        other_ptr
        + offs_b * stride_other_b
        + offs_c[:, None, None] * stride_other_c
        + offs_m[None, :, None] * stride_other_h
        + offs_n[None, None, :] * stride_other_w
    )
    
    # For input, we need to compute nearest neighbor indices
    # input_h = floor(offs_m * scale_h), input_w = floor(offs_n * scale_w)
    input_offs_h = (offs_m * scale_h).to(tl.int64)
    input_offs_w = (offs_n * scale_w).to(tl.int64)
    
    # Clamp to valid range
    input_offs_h = tl.minimum(tl.maximum(input_offs_h, 0), H_in - 1)
    input_offs_w = tl.minimum(tl.maximum(input_offs_w, 0), W_in - 1)
    
    # Load input tensor with nearest neighbor indices
    input_ptrs = (
        input_ptr
        + offs_b * stride_input_b
        + offs_c[:, None, None] * stride_input_c
        + input_offs_h[None, :, None] * stride_input_h
        + input_offs_w[None, None, :] * stride_input_w
    )
    
    # Load and compute
    # Iterate over channels
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for c in range(0, C):
        # Load input value
        input_val = tl.load(
            input_ptrs + c * stride_input_c,
            mask=mask_m[None, :, None] & mask_n[None, None, :],
            other=0.0
        )
        
        # Load other value
        other_val = tl.load(
            other_ptrs + c * stride_other_c,
            mask=mask_m[None, :, None] & mask_n[None, None, :],
            other=0.0
        )
        
        # Multiply
        result = input_val * other_val
        
        # Store result (accumulate across channels if needed, but here we replace)
        # Actually for this case, we compute one channel at a time and store
        output_ptrs = (
            output_ptr
            + offs_b * stride_output_b
            + c * stride_output_c
            + offs_m[:, None] * stride_output_h
            + offs_n[None, :] * stride_output_w
        )
        tl.store(output_ptrs, result, mask=mask_m[:, None] & mask_n[None, :])


def interpolate_mul_triton(input_tensor, other_tensor, size_h, size_w):
    """
    Fused interpolate (nearest) + multiply operation.
    
    Args:
        input_tensor: Input tensor [B, C, H_in, W_in]
        other_tensor: Other tensor [B, C, H_out, W_out] 
        size_h: Output height
        size_w: Output width
    """
    B, C, H_in, W_in = input_tensor.shape
    _, _, H_out, W_out = other_tensor.shape
    
    # Compute scale factors for nearest neighbor
    scale_h = H_in / H_out
    scale_w = W_in / W_out
    
    # Allocate output
    output = torch.empty((B, C, H_out, W_out), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel
    # Grid: (H_out * W_out, B) - each output element gets a thread
    grid = (H_out * W_out, B)
    
    interpolate_mul_kernel[grid](
        input_tensor, other_tensor, output,
        B, C, H_in, W_in, H_out, W_out,
        scale_h, scale_w,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        other_tensor.stride(0), other_tensor.stride(1), other_tensor.stride(2), other_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
    )
    
    return output


@torch.fx.wrap
def interpolate_mul_wrapper(input_tensor, other_tensor, size_h, size_w):
    return interpolate_mul_triton(input_tensor, other_tensor, size_h, size_w)


def pattern(in_0, in_1, in_2, in_3):
    """
    Match Pattern A: interpolate(in_0, (64, 48)) * in_2, interpolate(in_1, (32, 24)) * in_3
    Returns both multiplication results.
    """
    tmp_0 = torch.nn.functional.interpolate(in_0, size=(64, 48), mode='nearest')
    tmp_1 = in_2 * tmp_0
    
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_3 = in_3 * tmp_2
    
    return tmp_1, tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    # First branch: interpolate in_0 to (64, 48), multiply with in_2
    # Second branch: interpolate in_1 to (32, 24), multiply with in_3
    return (in_0, in_2, 64, 48, in_1, in_3, 32, 24)


def replacement_func():
    return interpolate_mul_wrapper