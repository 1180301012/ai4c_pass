import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern to match: unfold -> reshape
    """
    tmp_2 = torch.nn.functional.unfold(input_tensor, (2, 2), 1, 0, (2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return (tmp_3,)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=5, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def unfold_reshape_kernel(
    input_ptr, output_ptr,
    B, C, H, W,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_op, stride_ow,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized unfold (2x2, stride 2) + reshape kernel
    Input: (B, C, H, W)
    After unfold: (B, C*4, H//2*W//2)
    After reshape: (B, C, 4, H//2*W//2)
    """
    pid = tl.program_id(0)
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Decode linear index to (b, c, p, patch_idx)
    # where p in [0, 3] is position within 2x2 patch
    mask = offsets < N
    
    patch_idx = offsets % num_patches
    temp = offsets // num_patches
    p = temp % 4
    temp = temp // 4
    c = temp % C
    b = temp // C
    
    # Decode patch position
    patch_h = patch_idx // W_out
    patch_w = patch_idx % W_out
    
    # Decode p to position within 2x2 patch
    p_h = p // 2
    p_w = p % 2
    
    # Actual position in input
    h = patch_h * 2 + p_h
    w = patch_w * 2 + p_w
    
    # Load from input
    input_ptrs = (input_ptr + b * stride_ib + c * stride_ic + 
                  h * stride_ih + w * stride_iw)
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Store to output: (B, C, 4, num_patches)
    output_ptrs = (output_ptr + b * stride_ob + c * stride_oc + 
                   p * stride_op + patch_idx * stride_ow)
    tl.store(output_ptrs, data, mask=mask)


@torch.fx.wrap
def optimized_unfold_reshape(input_tensor):
    """
    Optimized implementation of unfold + reshape
    """
    B, C, H, W = input_tensor.shape
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Output shape: (B, C, 4, num_patches)
    output = torch.empty((B, C, 4, num_patches), device=input_tensor.device, dtype=input_tensor.dtype)
    
    N = B * C * 4 * num_patches
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    unfold_reshape_kernel[grid](
        input_tensor, output,
        B, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_unfold_reshape