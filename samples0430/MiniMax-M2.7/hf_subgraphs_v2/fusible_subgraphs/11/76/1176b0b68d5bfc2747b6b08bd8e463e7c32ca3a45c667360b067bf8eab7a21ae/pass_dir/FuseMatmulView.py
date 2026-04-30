import torch
import triton
import triton.language as tl


@triton.jit
def matmul_view_kernel(
    a_ptr, b_ptr, c_ptr,
    # a: [B, C_a, H_a, W_a] where last two dims are matrix
    # b: [B, C_b, H_b, W_b] where last two dims are matrix
    # Output: [B_out, C_out, H_out, W_out]
    B, C_a, H_a, W_a,
    B2, C_b, H_b, W_b,
    C_out, H_out, W_out,
    stride_a_0, stride_a_1, stride_a_2, stride_a_3,
    stride_b_0, stride_b_1, stride_b_2, stride_b_3,
    stride_c_0, stride_c_1, stride_c_2, stride_c_3,
    K: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused matmul + view kernel.
    Computes C = A @ B (batch matmul on last 2 dims) and stores in view layout.
    """
    # Get position
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # The output position is [pid_b, pid_c, pid_h, pid_w]
    # We need to compute the dot product for this position
    
    # For 4D tensors A[B, C_a, H_a, W_a] @ B[B, C_b, H_b, W_b]:
    # The matmul is performed on the last 2 dimensions
    # A @ B gives [B, C_a, H_a, W_b]
    # For our specific case: W_a == H_b (the K dimension)
    
    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)
    
    # The matmul computes: sum over k of A[..., pid_h, k] * B[..., k, pid_w]
    # Where pid_h corresponds to the H dimension and pid_w to the W dimension of output
    
    for k in range(K):
        # A offset: [pid_b, pid_c, pid_h, k]
        offs_a = (pid_b * stride_a_0 + pid_c * stride_a_1 + 
                  pid_h * stride_a_2 + k * stride_a_3)
        
        # B offset: [pid_b, pid_c, k, pid_w]
        offs_b = (pid_b * stride_b_0 + pid_c * stride_b_1 + 
                  k * stride_b_2 + pid_w * stride_b_3)
        
        a = tl.load(a_ptr + offs_a)
        b = tl.load(b_ptr + offs_b)
        
        acc += a * b
    
    # Convert to output dtype
    c = acc.to(tl.float16)
    
    # Store at output position
    offs_c = (pid_b * stride_c_0 + pid_c * stride_c_1 + 
              pid_h * stride_c_2 + pid_w * stride_c_3)
    tl.store(c_ptr + offs_c, c)


@torch.fx.wrap
def fused_matmul_view(a, b, view_b, view_c, view_h, view_w):
    """
    Fused matmul + view operation.
    
    Args:
        a: Input tensor [B, C_a, H, W] 
        b: Input tensor [B, C_b, H, W]
        view_b, view_c, view_h, view_w: Target view shape
    
    Returns:
        Tensor of shape [view_b, view_c, view_h, view_w]
    """
    B, C_a, H_a, W_a = a.shape
    B2, C_b, H_b, W_b = b.shape
    K = W_a  # W_a == H_b for valid matmul
    
    output_shape = (view_b, view_c, view_h, view_w)
    c = torch.empty(output_shape, device=a.device, dtype=a.dtype)
    
    # Grid: (B, C_out, H_out, W_out)
    grid = (view_b, view_c, view_h, view_w)
    
    matmul_view_kernel[grid](
        a, b, c,
        B, C_a, H_a, W_a,
        B2, C_b, H_b, W_b,
        view_c, view_h, view_w,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        K,
        BLOCK_SIZE=1,
    )
    
    return c


def pattern(in_0, in_1):
    """
    Match the pattern: matmul followed by view
    Pattern: matmul = in_1 @ in_0; tmp = matmul.view(...)
    """
    matmul = in_1 @ in_0
    tmp_1 = matmul.view(24, 128, 20, 20)
    return tmp_1


def replacement_args(in_0, in_1):
    """
    Extract arguments needed for the replacement function.
    """
    # For this specific pattern: view(24, 128, 20, 20)
    return (in_0, in_1, 24, 128, 20, 20)


def replacement_func():
    """
    Returns the fused matmul + view function.
    """
    return fused_matmul_view