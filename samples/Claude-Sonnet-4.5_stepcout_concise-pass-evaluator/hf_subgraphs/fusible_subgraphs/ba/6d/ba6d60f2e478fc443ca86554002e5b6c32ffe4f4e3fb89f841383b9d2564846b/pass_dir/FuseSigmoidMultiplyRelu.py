import torch
import triton
import triton.language as tl

def pattern(conv_out, x_main):
    """
    Pattern to match: sigmoid -> multiply
    """
    attn = conv_out.sigmoid()
    mult = x_main * attn
    return mult

def replacement_args(conv_out, x_main):
    return (conv_out, x_main)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_se_kernel(
    conv_out_ptr,
    x_main_ptr,
    out_ptr,
    n_elements,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel for sigmoid + broadcast multiply
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for broadcasting
    CHW = C * H * W
    HW = H * W
    
    b_idx = offsets // CHW
    rem = offsets % CHW
    c_idx = rem // HW
    
    # Map to conv_out indices
    se_offsets = b_idx * C + c_idx
    
    # Load values
    conv_out = tl.load(conv_out_ptr + se_offsets, mask=mask, other=0.0)
    x_main = tl.load(x_main_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operations: sigmoid -> multiply
    attn = tl.sigmoid(conv_out)
    mult = x_main * attn
    
    # Store result
    tl.store(out_ptr + offsets, mult, mask=mask)

@torch.fx.wrap
def fused_se(conv_out, x_main):
    """
    Wrapper function for fused SE attention
    """
    B, C, H, W = x_main.shape
    out = torch.empty_like(x_main)
    n_elements = out.numel()
    
    # Squeeze conv_out from [B, C, 1, 1] to [B, C] for efficient indexing
    conv_out_squeezed = conv_out.squeeze(-1).squeeze(-1).contiguous()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_se_kernel[grid](
        conv_out_squeezed, x_main, out, n_elements, B, C, H, W
    )
    
    return out

def replacement_func():
    return fused_se