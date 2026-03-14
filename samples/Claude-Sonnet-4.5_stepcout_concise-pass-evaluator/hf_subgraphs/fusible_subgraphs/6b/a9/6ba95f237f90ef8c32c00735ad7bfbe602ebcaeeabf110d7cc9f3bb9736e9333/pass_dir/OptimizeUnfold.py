import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern to match: unfold only
    """
    tmp_2 = torch.nn.functional.unfold(input_tensor, kernel_size=(2, 2), stride=(2, 2))
    return (tmp_2,)

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
def unfold_kernel(
    input_ptr, output_ptr,
    B, C, H, W,
    stride_ib, stride_ic, stride_ih, stride_iw,
    stride_ob, stride_oc, stride_ow,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized unfold (2x2, stride 2) kernel
    Input: (B, C, H, W)
    Output: (B, C*4, H//2*W//2)
    """
    pid = tl.program_id(0)
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < N
    
    # Decode linear index to (b, c_out, patch_idx)
    # c_out = c * 4 + p
    patch_idx = offsets % num_patches
    temp = offsets // num_patches
    c_out = temp % (C * 4)
    b = temp // (C * 4)
    
    # Decode c_out to c and p
    c = c_out // 4
    p = c_out % 4
    
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
    
    # Store to output: (B, C*4, num_patches)
    output_ptrs = (output_ptr + b * stride_ob + c_out * stride_oc + 
                   patch_idx * stride_ow)
    tl.store(output_ptrs, data, mask=mask)


@torch.fx.wrap
def optimized_unfold(input_tensor):
    """
    Optimized implementation of unfold
    """
    B, C, H, W = input_tensor.shape
    
    H_out = H // 2
    W_out = W // 2
    num_patches = H_out * W_out
    
    # Output shape: (B, C*4, num_patches)
    output = torch.empty((B, C * 4, num_patches), device=input_tensor.device, dtype=input_tensor.dtype)
    
    N = B * C * 4 * num_patches
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    unfold_kernel[grid](
        input_tensor, output,
        B, C, H, W,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_unfold