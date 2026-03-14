import torch
import triton
import triton.language as tl


def pattern(k_tensor):
    """
    Match the permute + transpose pattern for K tensor.
    Input: k_tensor [batch, seq, head, d_k] from split
    Output: K^T [batch, head, d_k, seq]
    """
    tmp_10 = k_tensor.permute(0, 2, 1, 3)
    tmp_13 = tmp_10.transpose(-2, -1)
    return tmp_13


def replacement_args(k_tensor):
    return (k_tensor,)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=8),
    ],
    key=['batch', 'head', 'seq', 'd'],
)
@triton.jit
def permute_transpose_kernel(
    input_ptr,   # [batch, seq, head, d] - non-contiguous from split
    output_ptr,  # [batch, head, d, seq] - contiguous output
    batch,
    head: tl.constexpr,
    seq: tl.constexpr,
    d: tl.constexpr,
    input_stride_b,  # stride for batch dim
    input_stride_s,  # stride for seq dim
    input_stride_h,  # stride for head dim  
    input_stride_d,  # stride for d dim
):
    """
    Fused permute + transpose: [batch, seq, head, d] -> [batch, head, d, seq]
    Handles non-contiguous input from split operation.
    """
    # Grid: (batch * head * seq,)
    pid = tl.program_id(0)
    
    # Decompose pid into (b, h, s)
    b = pid // (head * seq)
    remainder = pid % (head * seq)
    h = remainder // seq
    s = remainder % seq
    
    # Input offset using actual strides (handles non-contiguous input)
    input_base = b * input_stride_b + s * input_stride_s + h * input_stride_h
    
    # Output: [batch, head, d, seq] - contiguous
    output_stride_b = head * d * seq
    output_stride_h = d * seq
    output_base = b * output_stride_b + h * output_stride_h
    
    # Load from non-contiguous input and store to transposed contiguous output
    d_offsets = tl.arange(0, d)
    
    # Load with input strides
    input_offsets = input_base + d_offsets * input_stride_d
    data = tl.load(input_ptr + input_offsets)
    
    # Store to transposed positions
    output_offsets = output_base + d_offsets * seq + s
    tl.store(output_ptr + output_offsets, data)


@torch.fx.wrap
def fused_permute_transpose(k_tensor):
    """
    Fused permute + transpose for K tensor.
    
    Input: k_tensor [batch, seq, head, d] (may be non-contiguous from split)
    Output: K^T [batch, head, d, seq] (contiguous for large batches, view for small)
    """
    batch = k_tensor.shape[0]
    
    # For small batches, use PyTorch's native view operations (zero copy)
    if batch <= 8:
        return k_tensor.permute(0, 2, 1, 3).transpose(-2, -1)
    
    seq = k_tensor.shape[1]
    head = k_tensor.shape[2]
    d = k_tensor.shape[3]
    
    # Get actual strides (important for non-contiguous tensors from split)
    strides = k_tensor.stride()
    input_stride_b = strides[0]
    input_stride_s = strides[1]
    input_stride_h = strides[2]
    input_stride_d = strides[3]
    
    # Allocate contiguous output tensor
    kt_out = torch.empty((batch, head, d, seq), device=k_tensor.device, dtype=k_tensor.dtype)
    
    grid = (batch * head * seq,)
    
    permute_transpose_kernel[grid](
        k_tensor, kt_out,
        batch, head, seq, d,
        input_stride_b, input_stride_s, input_stride_h, input_stride_d,
    )
    
    return kt_out


def replacement_func():
    return fused_permute_transpose